// bg_stub.cpp - CPU stand-in for bgpu.dll (cloud build checks of the host code only).
#include "../gpu/bnt_gpu.h"
#include "../src/argon_ref.h"
#include <vector>
#include <string>
#include <chrono>
struct Stub { uint32_t m; int lanes; std::vector<uint32_t> table; std::vector<std::vector<uint64_t>> mem; };
extern "C" {
int bg_device_count(void) { return 1; }
void* bg_create(int, uint32_t m, int lanes, int, int) { Stub* s = new Stub(); s->m = m; s->lanes = lanes > 0 ? lanes : 1; if (s->lanes > 4) s->lanes = 4;
  s->table.resize(m / 2); bnt::ref_table(m, s->table.data()); s->mem.resize(s->lanes); return s; }
void bg_destroy(void* h) { delete (Stub*)h; }
int bg_lanes(void* h) { return ((Stub*)h)->lanes; }
const char* bg_name(void*) { return "CPU stub"; }
int bg_compute_capability(void*) { return 0; }
int bg_sm_count(void*) { return 0; }
double bg_memory_total(void*) { return 0; }
double bg_memory_used(void*) { return 0; }
const char* bg_last_error(void*) { return ""; }
const char* bg_kernel_info(void) { return "cpu stub"; }
int bg_hash(void* h, const uint64_t* init, uint64_t* last, int n, volatile int* abort, double* ms) {
  Stub* s = (Stub*)h; auto t0 = std::chrono::steady_clock::now();
  for (int l = 0; l < n; l++) { auto& mem = s->mem[l]; mem.assign((size_t)s->m * 128, 0); memcpy(mem.data(), init + l * 256, 2048);
    for (uint32_t i = 2; i < s->m; i++) { const uint64_t* p = &mem[(size_t)(i - 1) * 128];
      uint32_t r = i < s->m / 2 ? s->table[i] : bnt::ref_index(i, (uint32_t)p[0]); bnt::compress(p, &mem[(size_t)r * 128], &mem[(size_t)i * 128]);
      if (abort && *abort && (i & 65535) == 0) return 1; }
    memcpy(last + l * 128, &mem[(size_t)(s->m - 1) * 128], 1024); std::vector<uint64_t>().swap(mem); }
  if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); return 0; }
int bg_profile(void*, int, double* a, double* b) { *a = *b = 1; return 0; }
}
