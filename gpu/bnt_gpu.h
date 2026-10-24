// bnt_gpu.h - C API of bgpu.dll, the CUDA Argon2id engine of BntMiner (Blocknet PoW).
// One "lane" = one hash in progress = m_blocks KiB of GPU memory (2 GiB for Blocknet),
// computed by one warp of 32 threads.  All lanes of a device advance together.
#pragma once
#include <stdint.h>

#ifdef _WIN32
#ifdef BGPU_BUILD
#define BG_API extern "C" __declspec(dllexport)
#else
#define BG_API extern "C" __declspec(dllimport)
#endif
#else
#define BG_API extern "C"
#endif

#define BG_BLOCK_WORDS 128

BG_API int bg_device_count(void);
// m_blocks: lane size in 1 KiB blocks (Blocknet: 2097152), multiple of 8.
// lanes: wanted lanes, 0 = as many as the free memory allows (minus reserve_mb).
// chunk: blocks per kernel launch (0 = default).
BG_API void* bg_create(int device, uint32_t m_blocks, int lanes, int reserve_mb, int chunk);
BG_API void bg_destroy(void* h);
BG_API int bg_lanes(void* h);
BG_API const char* bg_name(void* h);
BG_API int bg_compute_capability(void* h);   // 61 for sm_61
BG_API int bg_sm_count(void* h);
BG_API double bg_memory_total(void* h);      // bytes
BG_API double bg_memory_used(void* h);       // bytes allocated by the engine
BG_API const char* bg_last_error(void* h);   // h may be null (create errors)
BG_API const char* bg_kernel_info(void);
// One hash per lane (nlanes <= bg_lanes):
//   init: nlanes x 2 blocks (blocks 0 and 1 of every lane, 256 words per lane)
//   last: nlanes x 1 block (last block of every lane, 128 words per lane)
//   abort: polled between launches; if *abort != 0 the run stops and 1 is returned.
// Returns 0 on success, 1 if aborted, -1 on a CUDA error (bg_last_error).
BG_API int bg_hash(void* h, const uint64_t* init, uint64_t* last, int nlanes, volatile int* abort, double* ms);
// Same with a time limit per kernel launch reported (profiling): ms_kernel = summed GPU time.
BG_API int bg_profile(void* h, int nlanes, double* ms_i_part, double* ms_d_part);
