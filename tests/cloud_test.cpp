// cloud_test.cpp - checks of the reference code (argon_ref.h) against the Rust argon2 crate
// vectors (tests/vectors.txt, made by tools/bntref = the call of blocknet core), and a CPU
// emulation of the CUDA kernel's thread mapping (g layout, padded shared memory, shuffles).
//   g++ -O2 -std=c++17 -Isrc tests/cloud_test.cpp -o cloud_test && ./cloud_test tests/vectors.txt [full]
#include "argon_ref.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

using namespace bnt;

static bool hex2(const std::string& h, std::vector<uint8_t>& out)
{
    out.clear();
    for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back((uint8_t)strtoul(h.substr(i, 2).c_str(), nullptr, 16));
    return true;
}
static std::string tohex(const uint8_t* p, size_t n)
{
    std::string s;
    char b[3];
    for (size_t i = 0; i < n; i++) { snprintf(b, 3, "%02x", p[i]); s += b; }
    return s;
}

// ---- emulation of fill_kernel's compression for one block, 32 "threads" ----
#define SP(w) ((w) + (((w) >> 4) << 2))
static void G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d) { BNT_GB(a, b, c, d); }
static void emu_round4(uint64_t a[32], uint64_t b[32], uint64_t c[32], uint64_t d[32])
{
    for (int t = 0; t < 32; t++) G(a[t], b[t], c[t], d[t]);
    uint64_t nb[32], nc[32], nd[32];
    for (int t = 0; t < 32; t++) {
        int g = t & ~3, k = t & 3;
        nb[t] = b[g | ((k + 1) & 3)];
        nc[t] = c[g | ((k + 2) & 3)];
        nd[t] = d[g | ((k + 3) & 3)];
    }
    for (int t = 0; t < 32; t++) G(a[t], nb[t], nc[t], nd[t]);
    for (int t = 0; t < 32; t++) {
        int g = t & ~3, k = t & 3;
        b[t] = nb[g | ((k + 3) & 3)];
        c[t] = nc[g | ((k + 2) & 3)];
        d[t] = nd[g | ((k + 1) & 3)];
    }
}
static void emu_compress(const uint64_t* prev, const uint64_t* ref, uint64_t* next)
{
    uint64_t s[160];
    memset(s, 0xcc, sizeof s);
    uint64_t r[32][4];
    for (int t = 0; t < 32; t++) {
        int w[4] = { 2 * t, 2 * t + 1, 64 + 2 * t, 65 + 2 * t };
        for (int j = 0; j < 4; j++) { r[t][j] = prev[w[j]] ^ ref[w[j]]; s[SP(w[j])] = r[t][j]; }
    }
    uint64_t a[32], b[32], c[32], d[32];
    for (int t = 0; t < 32; t++) {
        int row = t >> 2, k = t & 3, rw = 16 * row + k;
        a[t] = s[SP(rw)]; b[t] = s[SP(rw + 4)]; c[t] = s[SP(rw + 8)]; d[t] = s[SP(rw + 12)];
    }
    emu_round4(a, b, c, d);
    for (int t = 0; t < 32; t++) {
        int row = t >> 2, k = t & 3, rw = 16 * row + k;
        s[SP(rw)] = a[t]; s[SP(rw + 4)] = b[t]; s[SP(rw + 8)] = c[t]; s[SP(rw + 12)] = d[t];
    }
    for (int t = 0; t < 32; t++) {
        int row = t >> 2, k = t & 3, cw = 2 * row + (k & 1) + 16 * (k >> 1);
        a[t] = s[SP(cw)]; b[t] = s[SP(cw + 32)]; c[t] = s[SP(cw + 64)]; d[t] = s[SP(cw + 96)];
    }
    emu_round4(a, b, c, d);
    for (int t = 0; t < 32; t++) {
        int row = t >> 2, k = t & 3, cw = 2 * row + (k & 1) + 16 * (k >> 1);
        s[SP(cw)] = a[t]; s[SP(cw + 32)] = b[t]; s[SP(cw + 64)] = c[t]; s[SP(cw + 96)] = d[t];
    }
    for (int t = 0; t < 32; t++) {
        int w[4] = { 2 * t, 2 * t + 1, 64 + 2 * t, 65 + 2 * t };
        for (int j = 0; j < 4; j++) next[w[j]] = s[SP(w[j])] ^ r[t][j];
    }
}
// padded indices must be distinct and fit
static bool check_padding()
{
    std::vector<int> seen(160, 0);
    for (int w = 0; w < 128; w++) { int p = SP(w); if (p >= 160 || seen[p]++) return false; }
    // 16-byte alignment of the g layout pairs
    for (int t = 0; t < 32; t++) if ((SP(2 * t) & 1) || SP(2 * t + 1) != SP(2 * t) + 1 || (SP(64 + 2 * t) & 1)) return false;
    return true;
}
// device index formula (32-bit __umulhi) == reference
static uint32_t dev_ref_index(uint32_t i, uint32_t j1)
{
    uint32_t area = i - 1;
    uint32_t x = (uint32_t)(((uint64_t)j1 * j1) >> 32);
    uint32_t y = (uint32_t)(((uint64_t)area * x) >> 32);
    return area - 1 - y;
}

static bool emu_hash(const uint8_t* header, uint64_t nonce, uint32_t m, uint8_t out[32])
{
    std::vector<uint64_t> mem((size_t)m * 128);
    std::vector<uint32_t> table(m / 2);
    ref_table(m, table.data());
    initial_blocks(header, 100, nonce, m, mem.data(), mem.data() + 128);
    for (uint32_t i = 2; i < m; i++) {
        const uint64_t* prev = &mem[(size_t)(i - 1) * 128];
        uint32_t r = i < m / 2 ? table[i] : dev_ref_index(i, (uint32_t)prev[0]);
        if (r >= i - 1) { printf("bad ref %u for %u\n", r, i); return false; }
        emu_compress(prev, &mem[(size_t)r * 128], &mem[(size_t)i * 128]);
    }
    final_hash(&mem[(size_t)(m - 1) * 128], out);
    return true;
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "tests/vectors.txt";
    bool full = argc > 2 && std::string(argv[2]) == "full";
    int fails = 0;
    // BLAKE2b known answers ("abc" and "" from RFC 7693 / hashlib)
    {
        uint8_t o[64];
        blake2b(o, 64, "abc", 3);
        if (tohex(o, 64) != "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923") fails++;
        blake2b(o, 32, "", 0);
        if (tohex(o, 32) != "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8") fails++;
        std::vector<uint8_t> big(300);
        for (int i = 0; i < 300; i++) big[i] = (uint8_t)i;
        blake2b(o, 64, big.data(), 256);   // exactly two blocks
        Blake2b S;
        b2_init(S, 64);
        b2_update(S, big.data(), 100);
        b2_update(S, big.data() + 100, 156);
        uint8_t o2[64];
        b2_final(S, o2);
        if (memcmp(o, o2, 64)) fails++;
        printf("BLAKE2b known answers + streaming: %s\n", fails ? "FAILED" : "OK");
    }
    if (!check_padding()) { printf("padding FAILED\n"); fails++; }
    // ref index formula
    {
        uint64_t x = 1;
        int bad = 0;
        for (int n = 0; n < 1000000; n++) {
            x = x * 6364136223846793005ULL + 1442695040888963407ULL;
            uint32_t i = 2 + (uint32_t)((x >> 20) % (2097152 - 2)), j1 = (uint32_t)(x >> 7);
            if (i < 3) continue;
            if (ref_index(i, j1) != dev_ref_index(i, j1)) bad++;
        }
        printf("ref index 32-bit formula: %s\n", bad ? "FAILED" : "OK");
        fails += bad;
    }
    // random compress: emulation == reference
    {
        uint64_t p[128], r[128], o1[128], o2[128], x = 99;
        int bad = 0;
        for (int n = 0; n < 200; n++) {
            for (int i = 0; i < 128; i++) { x = x * 6364136223846793005ULL + 1; p[i] = x; x = x * 6364136223846793005ULL + 1; r[i] = x; }
            compress(p, r, o1);
            emu_compress(p, r, o2);
            if (memcmp(o1, o2, 1024)) bad++;
        }
        printf("kernel mapping emulation of G: %s\n", bad ? "FAILED" : "OK");
        fails += bad;
    }
    FILE* f = fopen(path, "r");
    if (!f) { printf("no vectors file %s\n", path); return 1; }
    char line[1024];
    int nv = 0;
    while (fgets(line, sizeof line, f)) {
        char hh[512], oh[128];
        unsigned m;
        unsigned long long nonce;
        if (sscanf(line, "%u %511s %llu %127s", &m, hh, &nonce, oh) != 4) continue;
        std::vector<uint8_t> header, expect;
        hex2(hh, header);
        hex2(oh, expect);
        if (m == POW_M_KIB && !full && nv > 0) continue;
        uint8_t out[32], out2[32];
        auto t0 = std::chrono::steady_clock::now();
        bool ok1 = pow_hash_cpu(header.data(), header.size(), nonce, m, out) && !memcmp(out, expect.data(), 32);
        double s1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool ok2 = true;
        if (m <= 65536 || full) ok2 = emu_hash(header.data(), nonce, m, out2) && !memcmp(out2, expect.data(), 32);
        printf("m=%u nonce=%llu: reference %s (%.2f s), kernel emulation %s\n", m, nonce, ok1 ? "OK" : "FAILED", s1, ok2 ? "OK" : "FAILED");
        if (!ok1 || !ok2) { fails++; printf("   expected %s got %s\n", oh, tohex(out, 32).c_str()); }
        nv++;
    }
    fclose(f);
    printf("RESULT: %s (%d vectors)\n", fails ? "FAILED" : "ALL OK", nv);
    return fails ? 1 : 0;
}
