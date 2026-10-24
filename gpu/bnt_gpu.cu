// bnt_gpu.cu - CUDA Argon2id engine of BntMiner (Blocknet PoW: Argon2id v0x13, m = 2 GiB,
// t = 1, p = 1).  Target: NVIDIA P104-100 (Pascal, sm_61), CUDA 12.9.  Apache-2.0.
//
// One lane (one hash) = one warp of 32 threads = m_blocks KiB of GPU memory.  The 1 KiB
// blocks are filled in order: block i = G(block i-1, block ref(i)).  Every thread keeps 4 of
// the 128 words of the current block in registers ("g layout": words 2t, 2t+1, 64+2t, 65+2t,
// loaded and stored as two 16-byte accesses, coalesced over the warp).  The BlaMka permutation
// works in shared memory (padded, 2 transactions per access): row step, then column step,
// each thread computing one of the 32 quarter-rounds, diagonals done with warp shuffles.
//
// Argon2i part (first half): reference indices from a fixed table (host-computed), the
// reference block of i+1 is loaded while block i is computed.  Argon2d part: the index comes
// from the first word of block i-1, so the load latency cannot be hidden inside a lane.
// Each thread only reads words it wrote itself (same g layout), so no extra memory fence is
// needed between iterations.
#define BGPU_BUILD
#include "bnt_gpu.h"
#include "../src/argon_ref.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

#define FULL_MASK 0xffffffffu
#define SP(w) ((w) + (((w) >> 4) << 2))      // padded shared-memory index of word w
#define SMEM_WORDS (128 + 32)

__device__ __forceinline__ uint64_t d_rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
__device__ __forceinline__ uint64_t d_rotr32(uint64_t x)
{
    return ((uint64_t)(uint32_t)x << 32) | (x >> 32);
}
__device__ __forceinline__ uint64_t d_fBlaMka(uint64_t x, uint64_t y)
{
    uint64_t m = (uint64_t)(uint32_t)x * (uint64_t)(uint32_t)y;
    return x + y + 2 * m;
}
__device__ __forceinline__ void d_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d)
{
    a = d_fBlaMka(a, b);
    d = d_rotr32(d ^ a);
    c = d_fBlaMka(c, d);
    b = d_rotr(b ^ c, 24);
    a = d_fBlaMka(a, b);
    d = d_rotr(d ^ a, 16);
    c = d_fBlaMka(c, d);
    b = d_rotr(b ^ c, 63);
}

// one BLAKE2 round on the 16 words held by a group of 4 threads (a,b,c,d = column k):
// column step, then diagonal step (b,c,d rotated across the group by shuffles)
__device__ __forceinline__ void d_round4(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, int gbase, int k)
{
    d_G(a, b, c, d);
    b = __shfl_sync(FULL_MASK, b, gbase | ((k + 1) & 3));
    c = __shfl_sync(FULL_MASK, c, gbase | ((k + 2) & 3));
    d = __shfl_sync(FULL_MASK, d, gbase | ((k + 3) & 3));
    d_G(a, b, c, d);
    b = __shfl_sync(FULL_MASK, b, gbase | ((k + 3) & 3));
    c = __shfl_sync(FULL_MASK, c, gbase | ((k + 2) & 3));
    d = __shfl_sync(FULL_MASK, d, gbase | ((k + 1) & 3));
}

__device__ __forceinline__ uint32_t d_ref_index(uint32_t i, uint32_t j1)
{
    uint32_t area = i - 1;
    uint32_t x = __umulhi(j1, j1);
    uint32_t y = __umulhi(area, x);
    return area - 1 - y;
}

__device__ __forceinline__ ulonglong2 xor2(ulonglong2 a, ulonglong2 b)
{
    ulonglong2 r;
    r.x = a.x ^ b.x;
    r.y = a.y ^ b.y;
    return r;
}

// blocks [start, end) of every lane (blockIdx.x = lane); start >= 2
__global__ void __launch_bounds__(32) fill_kernel(uint64_t* const* __restrict__ lanes, const uint32_t* __restrict__ table, uint32_t half,
                                                  uint32_t start, uint32_t end)
{
    __shared__ __align__(16) uint64_t s[SMEM_WORDS];
    const int t = threadIdx.x;
    ulonglong2* m2 = (ulonglong2*)lanes[blockIdx.x];
    const int row = t >> 2, k = t & 3, gbase = t & ~3;
    const int rw = 16 * row + k;                          // row step: words rw + 4j
    const int cw = 2 * row + (k & 1) + 16 * (k >> 1);     // column step: words cw + 32j
    const int sr0 = SP(rw), sr1 = SP(rw + 4), sr2 = SP(rw + 8), sr3 = SP(rw + 12);
    const int sc0 = SP(cw), sc1 = SP(cw + 32), sc2 = SP(cw + 64), sc3 = SP(cw + 96);
    ulonglong2* s2 = (ulonglong2*)s;
    const int sg0 = SP(2 * t) >> 1, sg1 = SP(64 + 2 * t) >> 1;   // g layout in 16-byte units

    ulonglong2 pa = m2[(size_t)(start - 1) * 64 + t];
    ulonglong2 pb = m2[(size_t)(start - 1) * 64 + 32 + t];
    ulonglong2 fa, fb;                                    // prefetched reference block (i-part)
    fa.x = fa.y = fb.x = fb.y = 0;
    if (start < half) {
        uint32_t r = __ldg(table + start);
        fa = m2[(size_t)r * 64 + t];
        fb = m2[(size_t)r * 64 + 32 + t];
    }
    for (uint32_t i = start; i < end; i++) {
        ulonglong2 qa, qb;
        if (i < half) {
            qa = fa;
            qb = fb;
            if (i + 1 < half && i + 1 < end) {
                uint32_t r = __ldg(table + i + 1);
                fa = m2[(size_t)r * 64 + t];
                fb = m2[(size_t)r * 64 + 32 + t];
            }
        } else {
            uint64_t w0 = __shfl_sync(FULL_MASK, pa.x, 0);
            uint32_t r = d_ref_index(i, (uint32_t)w0);
            qa = m2[(size_t)r * 64 + t];
            qb = m2[(size_t)r * 64 + 32 + t];
        }
        const ulonglong2 ra = xor2(pa, qa), rb = xor2(pb, qb);
        __syncwarp();
        s2[sg0] = ra;
        s2[sg1] = rb;
        __syncwarp();
        uint64_t a = s[sr0], b = s[sr1], c = s[sr2], d = s[sr3];
        d_round4(a, b, c, d, gbase, k);
        s[sr0] = a; s[sr1] = b; s[sr2] = c; s[sr3] = d;
        __syncwarp();
        a = s[sc0]; b = s[sc1]; c = s[sc2]; d = s[sc3];
        d_round4(a, b, c, d, gbase, k);
        s[sc0] = a; s[sc1] = b; s[sc2] = c; s[sc3] = d;
        __syncwarp();
        pa = xor2(s2[sg0], ra);
        pb = xor2(s2[sg1], rb);
        m2[(size_t)i * 64 + t] = pa;
        m2[(size_t)i * 64 + 32 + t] = pb;
    }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
static const char* KERNEL_INFO = "warp/lane, smem pad4, i-part prefetch";

struct BgCtx {
    int dev = -1;
    uint32_t m_blocks = 0;
    int lanes = 0;
    int chunk = 32768;
    std::vector<uint64_t*> lane_mem;
    uint64_t** d_lanes = nullptr;
    uint32_t* d_table = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    std::string name, err;
    int cc = 0, sms = 0;
    double mem_total = 0, mem_used = 0;
};

static std::string g_create_err;

static bool ck(BgCtx* c, cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return true;
    char b[256];
    snprintf(b, sizeof b, "%s: %s (%d)", what, cudaGetErrorString(e), (int)e);
    if (c) c->err = b; else g_create_err = b;
    return false;
}

BG_API int bg_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

BG_API void bg_destroy(void* h)
{
    BgCtx* c = (BgCtx*)h;
    if (!c) return;
    cudaSetDevice(c->dev);
    for (auto p : c->lane_mem) cudaFree(p);
    if (c->d_lanes) cudaFree(c->d_lanes);
    if (c->d_table) cudaFree(c->d_table);
    if (c->ev0) cudaEventDestroy(c->ev0);
    if (c->ev1) cudaEventDestroy(c->ev1);
    if (c->stream) cudaStreamDestroy(c->stream);
    delete c;
}

BG_API void* bg_create(int device, uint32_t m_blocks, int lanes, int reserve_mb, int chunk)
{
    g_create_err.clear();
    if (m_blocks < 8 || m_blocks % 8) { g_create_err = "m_blocks must be a multiple of 8"; return nullptr; }
    BgCtx* c = new BgCtx();
    c->dev = device;
    c->m_blocks = m_blocks;
    if (chunk > 0) c->chunk = chunk;
    cudaDeviceProp p;
    if (!ck(nullptr, cudaSetDevice(device), "cudaSetDevice") || !ck(nullptr, cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties")) {
        delete c;
        return nullptr;
    }
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    c->name = p.name;
    c->cc = p.major * 10 + p.minor;
    c->sms = p.multiProcessorCount;
    c->mem_total = (double)p.totalGlobalMem;
    if (!ck(nullptr, cudaFree(0), "cuda context") || !ck(nullptr, cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream") ||
        !ck(nullptr, cudaEventCreate(&c->ev0), "event") || !ck(nullptr, cudaEventCreate(&c->ev1), "event")) {
        bg_destroy(c);
        return nullptr;
    }
    // reference table of the Argon2i half
    std::vector<uint32_t> table(m_blocks / 2);
    bnt::ref_table(m_blocks, table.data());
    if (!ck(nullptr, cudaMalloc(&c->d_table, table.size() * 4), "cudaMalloc table") ||
        !ck(nullptr, cudaMemcpy(c->d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "copy table")) {
        bg_destroy(c);
        return nullptr;
    }
    size_t lane_bytes = (size_t)m_blocks * 1024;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    size_t reserve = (size_t)(reserve_mb > 0 ? reserve_mb : 0) * 1048576;
    int maxl = fr > reserve ? (int)((fr - reserve) / lane_bytes) : 0;
    if (lanes <= 0 || lanes > maxl) lanes = lanes <= 0 ? maxl : lanes;
    if (lanes > 64) lanes = 64;
    for (int l = 0; l < lanes; l++) {
        uint64_t* m = nullptr;
        if (cudaMalloc(&m, lane_bytes) != cudaSuccess) { cudaGetLastError(); break; }
        c->lane_mem.push_back(m);
    }
    c->lanes = (int)c->lane_mem.size();
    if (c->lanes == 0) {
        char b[160];
        snprintf(b, sizeof b, "not enough GPU memory for one lane (%.0f MB free, %.0f MB per lane)", fr / 1048576.0, lane_bytes / 1048576.0);
        g_create_err = b;
        bg_destroy(c);
        return nullptr;
    }
    if (!ck(nullptr, cudaMalloc(&c->d_lanes, sizeof(uint64_t*) * c->lanes), "cudaMalloc lanes") ||
        !ck(nullptr, cudaMemcpy(c->d_lanes, c->lane_mem.data(), sizeof(uint64_t*) * c->lanes, cudaMemcpyHostToDevice), "copy lanes")) {
        bg_destroy(c);
        return nullptr;
    }
    c->mem_used = (double)lane_bytes * c->lanes + table.size() * 4.0;
    return c;
}

BG_API int bg_lanes(void* h) { return h ? ((BgCtx*)h)->lanes : 0; }
BG_API const char* bg_name(void* h) { return h ? ((BgCtx*)h)->name.c_str() : ""; }
BG_API int bg_compute_capability(void* h) { return h ? ((BgCtx*)h)->cc : 0; }
BG_API int bg_sm_count(void* h) { return h ? ((BgCtx*)h)->sms : 0; }
BG_API double bg_memory_total(void* h) { return h ? ((BgCtx*)h)->mem_total : 0; }
BG_API double bg_memory_used(void* h) { return h ? ((BgCtx*)h)->mem_used : 0; }
BG_API const char* bg_last_error(void* h) { return h ? ((BgCtx*)h)->err.c_str() : g_create_err.c_str(); }
BG_API const char* bg_kernel_info(void) { return KERNEL_INFO; }

static int run_range(BgCtx* c, int nlanes, uint32_t start, uint32_t end, volatile int* abort)
{
    const uint32_t half = c->m_blocks / 2;
    while (start < end) {
        uint32_t e = start + (uint32_t)c->chunk;
        if (e > end || e < start) e = end;
        fill_kernel<<<nlanes, 32, 0, c->stream>>>(c->d_lanes, c->d_table, half, start, e);
        if (!ck(c, cudaGetLastError(), "fill_kernel launch")) return -1;
        if (!ck(c, cudaStreamSynchronize(c->stream), "fill_kernel")) return -1;
        start = e;
        if (abort && *abort) return 1;
    }
    return 0;
}

BG_API int bg_hash(void* h, const uint64_t* init, uint64_t* last, int nlanes, volatile int* abort, double* ms)
{
    BgCtx* c = (BgCtx*)h;
    if (!c || nlanes < 1 || nlanes > c->lanes) return -1;
    auto t0 = std::chrono::steady_clock::now();
    cudaSetDevice(c->dev);
    for (int l = 0; l < nlanes; l++)
        if (!ck(c, cudaMemcpyAsync(c->lane_mem[l], init + (size_t)l * 256, 2048, cudaMemcpyHostToDevice, c->stream), "copy init")) return -1;
    int r = run_range(c, nlanes, 2, c->m_blocks, abort);
    if (r) return r;
    for (int l = 0; l < nlanes; l++)
        if (!ck(c, cudaMemcpyAsync(last + (size_t)l * 128, c->lane_mem[l] + (size_t)(c->m_blocks - 1) * 128, 1024, cudaMemcpyDeviceToHost, c->stream),
                "copy last"))
            return -1;
    if (!ck(c, cudaStreamSynchronize(c->stream), "sync")) return -1;
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

BG_API int bg_profile(void* h, int nlanes, double* ms_i_part, double* ms_d_part)
{
    BgCtx* c = (BgCtx*)h;
    if (!c || nlanes < 1 || nlanes > c->lanes) return -1;
    cudaSetDevice(c->dev);
    std::vector<uint64_t> init((size_t)nlanes * 256);
    for (size_t i = 0; i < init.size(); i++) init[i] = 0x9e3779b97f4a7c15ULL * (i + 1);
    for (int l = 0; l < nlanes; l++)
        if (!ck(c, cudaMemcpy(c->lane_mem[l], init.data() + (size_t)l * 256, 2048, cudaMemcpyHostToDevice), "copy init")) return -1;
    const uint32_t half = c->m_blocks / 2;
    auto t0 = std::chrono::steady_clock::now();
    if (run_range(c, nlanes, 2, half, nullptr)) return -1;
    auto t1 = std::chrono::steady_clock::now();
    if (run_range(c, nlanes, half, c->m_blocks, nullptr)) return -1;
    auto t2 = std::chrono::steady_clock::now();
    if (ms_i_part) *ms_i_part = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms_d_part) *ms_d_part = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return 0;
}
