// argon_ref.h - Blocknet (BNT) proof of work, reference code (header only, portable C++:
// MSVC for the GPU DLL, MinGW g++ for the miner, g++ in the cloud tests).
//
// Blocknet PoW (github.com/blocknetprivacy/core, crypto-rs/src/pow.rs):
//   Argon2id, version 0x13, m = 2 GiB (2,097,152 KiB = 2^21 blocks of 1 KiB), t = 1, p = 1,
//   output 32 bytes, password = nonce (u64 little-endian), salt = block header without the
//   nonce (100 bytes), no secret, no associated data.  hash <= share target (big-endian bytes).
//
// With t = 1 and p = 1 there is a single lane and a single pass:
//   slices 0 and 1 (blocks 0 .. m/2-1)  : Argon2i addressing (independent of the data:
//                                          the reference indices are a fixed table for a given m)
//   slices 2 and 3 (blocks m/2 .. m-1)  : Argon2d addressing (J1 = low 32 bits of block i-1)
//   reference index of block i (pass 0, one lane): area = i - 1, x = J1*J1 >> 32,
//   ref = area - 1 - (area * x >> 32).
//   block i = G(block i-1, block ref) (pass 0: no XOR with the old content).
#pragma once
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

namespace bnt {

static const uint32_t POW_M_KIB = 2u * 1024u * 1024u;   // 2 GiB
static const int HEADER_BYTES = 100;                     // header without nonce
static const int BLOCK_WORDS = 128;                      // 1 KiB block

// ---------------------------------------------------------------------------
// BLAKE2b (RFC 7693), unkeyed, variable output length
// ---------------------------------------------------------------------------
static inline uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

static const uint64_t B2_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL };
static const uint8_t B2_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }, { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 }, { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 }, { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 }, { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 } };

struct Blake2b {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t buf[128];
    size_t buflen;
    size_t outlen;
};

static inline uint64_t load64(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static inline void store64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static inline void store32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

static inline void b2_compress(Blake2b& S, const uint8_t* block, bool last)
{
    uint64_t m[16], v[16];
    for (int i = 0; i < 16; i++) m[i] = load64(block + 8 * i);
    for (int i = 0; i < 8; i++) { v[i] = S.h[i]; v[i + 8] = B2_IV[i]; }
    v[12] ^= S.t[0];
    v[13] ^= S.t[1];
    if (last) v[14] = ~v[14];
#define B2G(r, i, a, b, c, d)                          \
    do {                                               \
        a = a + b + m[B2_SIGMA[r][2 * i + 0]];         \
        d = rotr64(d ^ a, 32);                         \
        c = c + d;                                     \
        b = rotr64(b ^ c, 24);                         \
        a = a + b + m[B2_SIGMA[r][2 * i + 1]];         \
        d = rotr64(d ^ a, 16);                         \
        c = c + d;                                     \
        b = rotr64(b ^ c, 63);                         \
    } while (0)
    for (int r = 0; r < 12; r++) {
        B2G(r, 0, v[0], v[4], v[8], v[12]);
        B2G(r, 1, v[1], v[5], v[9], v[13]);
        B2G(r, 2, v[2], v[6], v[10], v[14]);
        B2G(r, 3, v[3], v[7], v[11], v[15]);
        B2G(r, 4, v[0], v[5], v[10], v[15]);
        B2G(r, 5, v[1], v[6], v[11], v[12]);
        B2G(r, 6, v[2], v[7], v[8], v[13]);
        B2G(r, 7, v[3], v[4], v[9], v[14]);
    }
#undef B2G
    for (int i = 0; i < 8; i++) S.h[i] ^= v[i] ^ v[i + 8];
}

static inline void b2_init(Blake2b& S, size_t outlen)
{
    for (int i = 0; i < 8; i++) S.h[i] = B2_IV[i];
    S.h[0] ^= 0x01010000ULL ^ (uint64_t)outlen;
    S.t[0] = S.t[1] = 0;
    S.buflen = 0;
    S.outlen = outlen;
}

static inline void b2_update(Blake2b& S, const void* data, size_t len)
{
    const uint8_t* in = (const uint8_t*)data;
    while (len > 0) {
        if (S.buflen == 128) {   // only compress when more data follows (the last block is final)
            S.t[0] += 128;
            if (S.t[0] < 128) S.t[1]++;
            b2_compress(S, S.buf, false);
            S.buflen = 0;
        }
        size_t take = 128 - S.buflen;
        if (take > len) take = len;
        memcpy(S.buf + S.buflen, in, take);
        S.buflen += take;
        in += take;
        len -= take;
    }
}

static inline void b2_final(Blake2b& S, uint8_t* out)
{
    S.t[0] += S.buflen;
    if (S.t[0] < S.buflen) S.t[1]++;
    memset(S.buf + S.buflen, 0, 128 - S.buflen);
    b2_compress(S, S.buf, true);
    uint8_t full[64];
    for (int i = 0; i < 8; i++) store64(full + 8 * i, S.h[i]);
    memcpy(out, full, S.outlen);
}

static inline void blake2b(uint8_t* out, size_t outlen, const void* in, size_t inlen)
{
    Blake2b S;
    b2_init(S, outlen);
    b2_update(S, in, inlen);
    b2_final(S, out);
}

// Argon2 variable-length hash H' (RFC 9106 section 3.3)
static inline void h_prime(uint8_t* out, uint32_t outlen, const uint8_t* in, size_t inlen)
{
    uint8_t lenbuf[4];
    store32(lenbuf, outlen);
    Blake2b S;
    if (outlen <= 64) {
        b2_init(S, outlen);
        b2_update(S, lenbuf, 4);
        b2_update(S, in, inlen);
        b2_final(S, out);
        return;
    }
    uint8_t v[64];
    b2_init(S, 64);
    b2_update(S, lenbuf, 4);
    b2_update(S, in, inlen);
    b2_final(S, v);
    memcpy(out, v, 32);
    out += 32;
    uint32_t remain = outlen - 32;
    while (remain > 64) {
        uint8_t nv[64];
        blake2b(nv, 64, v, 64);
        memcpy(v, nv, 64);
        memcpy(out, v, 32);
        out += 32;
        remain -= 32;
    }
    uint8_t last[64];
    blake2b(last, remain, v, 64);
    memcpy(out, last, remain);
}

// ---------------------------------------------------------------------------
// Argon2 compression G (BlaMka permutation)
// ---------------------------------------------------------------------------
static inline uint64_t fBlaMka(uint64_t x, uint64_t y)
{
    return x + y + 2 * ((uint64_t)(uint32_t)x * (uint64_t)(uint32_t)y);
}
#define BNT_GB(a, b, c, d)                \
    do {                                  \
        a = fBlaMka(a, b);                \
        d = rotr64(d ^ a, 32);            \
        c = fBlaMka(c, d);                \
        b = rotr64(b ^ c, 24);            \
        a = fBlaMka(a, b);                \
        d = rotr64(d ^ a, 16);            \
        c = fBlaMka(c, d);                \
        b = rotr64(b ^ c, 63);            \
    } while (0)
#define BNT_ROUND(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) \
    do {                                                                                 \
        BNT_GB(v0, v4, v8, v12);                                                         \
        BNT_GB(v1, v5, v9, v13);                                                         \
        BNT_GB(v2, v6, v10, v14);                                                        \
        BNT_GB(v3, v7, v11, v15);                                                        \
        BNT_GB(v0, v5, v10, v15);                                                        \
        BNT_GB(v1, v6, v11, v12);                                                        \
        BNT_GB(v2, v7, v8, v13);                                                         \
        BNT_GB(v3, v4, v9, v14);                                                         \
    } while (0)

// next = P(prev ^ ref) ^ prev ^ ref  (pass 0, no XOR with the old content)
static inline void compress(const uint64_t* prev, const uint64_t* ref, uint64_t* next)
{
    uint64_t R[128], Q[128];
    for (int i = 0; i < 128; i++) Q[i] = R[i] = prev[i] ^ ref[i];
    for (int i = 0; i < 8; i++) {
        uint64_t* v = Q + 16 * i;
        BNT_ROUND(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
    }
    for (int i = 0; i < 8; i++) {
        uint64_t* v = Q + 2 * i;
        BNT_ROUND(v[0], v[1], v[16], v[17], v[32], v[33], v[48], v[49], v[64], v[65], v[80], v[81], v[96], v[97], v[112], v[113]);
    }
    for (int i = 0; i < 128; i++) next[i] = Q[i] ^ R[i];
}

// ---------------------------------------------------------------------------
// Blocknet PoW pieces
// ---------------------------------------------------------------------------
// H0 = BLAKE2b-512(p, T, m, t, v, y, |P|, P, |S|, S, |K|, |X|)
static inline void h0(uint8_t out[64], const uint8_t* header, size_t hlen, uint64_t nonce, uint32_t m_kib)
{
    uint8_t buf[4];
    Blake2b S;
    b2_init(S, 64);
    auto u32 = [&](uint32_t x) { store32(buf, x); b2_update(S, buf, 4); };
    u32(1);          // parallelism
    u32(32);         // tag length
    u32(m_kib);      // memory
    u32(1);          // passes
    u32(0x13);       // version
    u32(2);          // Argon2id
    uint8_t pw[8];
    store64(pw, nonce);
    u32(8);
    b2_update(S, pw, 8);
    u32((uint32_t)hlen);
    b2_update(S, header, hlen);
    u32(0);          // secret
    u32(0);          // associated data
    b2_final(S, out);
}

static inline void bytes_to_block(const uint8_t* b, uint64_t* w) { for (int i = 0; i < 128; i++) w[i] = load64(b + 8 * i); }
static inline void block_to_bytes(const uint64_t* w, uint8_t* b) { for (int i = 0; i < 128; i++) store64(b + 8 * i, w[i]); }

// blocks 0 and 1 of the lane
static inline void initial_blocks(const uint8_t* header, size_t hlen, uint64_t nonce, uint32_t m_kib, uint64_t b0[128], uint64_t b1[128])
{
    uint8_t in[72], blk[1024];
    h0(in, header, hlen, nonce, m_kib);
    store32(in + 64, 0);
    store32(in + 68, 0);   // lane 0
    h_prime(blk, 1024, in, 72);
    bytes_to_block(blk, b0);
    store32(in + 64, 1);
    h_prime(blk, 1024, in, 72);
    bytes_to_block(blk, b1);
}

// tag = H'(32, last block)
static inline void final_hash(const uint64_t last[128], uint8_t out[32])
{
    uint8_t blk[1024];
    block_to_bytes(last, blk);
    h_prime(out, 32, blk, 1024);
}

// reference index of block i (pass 0, one lane) from its 32-bit pseudo-random value J1
static inline uint32_t ref_index(uint32_t i, uint32_t j1)
{
    uint64_t area = (uint64_t)i - 1;
    uint64_t x = ((uint64_t)j1 * (uint64_t)j1) >> 32;
    uint64_t y = (area * x) >> 32;
    return (uint32_t)(area - 1 - y);
}

// Argon2i part (slices 0 and 1): reference index of every block i in [0, m/2), fixed for m.
// Entries 0 and 1 are unused (0).  Address blocks: G(0, G(0, input)), input = (pass 0, lane 0,
// slice, m, t = 1, type 2, counter, 0...), counter restarts at every slice (one input block per
// segment, as in the reference fill_segment).
static inline void ref_table(uint32_t m_blocks, uint32_t* table)
{
    const uint32_t seg = m_blocks / 4;
    uint64_t zero[128], input[128], addr[128], tmp[128];
    memset(zero, 0, sizeof zero);
    table[0] = table[1] = 0;
    for (uint32_t slice = 0; slice < 2; slice++) {
        memset(input, 0, sizeof input);
        input[0] = 0;
        input[1] = 0;
        input[2] = slice;
        input[3] = m_blocks;
        input[4] = 1;
        input[5] = 2;
        uint32_t start = slice == 0 ? 2 : 0;
        auto next_addresses = [&]() {
            input[6]++;
            compress(zero, input, tmp);
            compress(zero, tmp, addr);
        };
        if (slice == 0) next_addresses();   // first block of the first segment: generated before i = 2
        for (uint32_t idx = start; idx < seg; idx++) {
            if (idx % 128 == 0) next_addresses();
            uint64_t pr = addr[idx % 128];
            uint32_t i = slice * seg + idx;
            table[i] = ref_index(i, (uint32_t)pr);
        }
    }
}

// Full CPU reference (m_blocks KiB of memory).  Returns false if the memory cannot be allocated.
static inline bool pow_hash_cpu(const uint8_t* header, size_t hlen, uint64_t nonce, uint32_t m_blocks, uint8_t out[32],
                                uint64_t* last_block = nullptr)
{
    uint64_t* mem = (uint64_t*)malloc((size_t)m_blocks * 1024);
    if (!mem) return false;
    std::vector<uint32_t> table(m_blocks / 2);
    ref_table(m_blocks, table.data());
    initial_blocks(header, hlen, nonce, m_blocks, mem, mem + 128);
    const uint32_t half = m_blocks / 2;
    for (uint32_t i = 2; i < m_blocks; i++) {
        const uint64_t* prev = mem + (size_t)(i - 1) * 128;
        uint32_t r = i < half ? table[i] : ref_index(i, (uint32_t)prev[0]);
        compress(prev, mem + (size_t)r * 128, mem + (size_t)i * 128);
    }
    final_hash(mem + (size_t)(m_blocks - 1) * 128, out);
    if (last_block) memcpy(last_block, mem + (size_t)(m_blocks - 1) * 128, 1024);
    free(mem);
    return true;
}

// hash <= target, both big-endian byte strings (seine / bntpool rule)
static inline bool hash_le_target(const uint8_t h[32], const uint8_t t[32]) { return memcmp(h, t, 32) <= 0; }

}   // namespace bnt
