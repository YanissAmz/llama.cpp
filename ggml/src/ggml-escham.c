// ESCHAM tile decode (CPU scalar) + vector transforms.
//
// Literal C port of tools/escha/escham_cpu.py (normative spec,
// hand-translated from the CUDA PTX, valid to 4.7e-4 rms against the vendor
// wheel). Decodes one atomic 16x16 tile: 16*K int16 -> 256 fp16 weights,
// 16-bit windows sliding K bits over the tile's circular code (tail-biting).
//
// This reproduces reconstruct_code only (raw trellis decode, cbA codebook).
// No Hadamard, no rin/rout here: they are vector transforms around the
// matmul (see BRIEF_M4_CPU.md addendum sec.5).

#include "ggml-escham.h"

#include <string.h>

#define ESCHAM_MASK  0x8FFF8FFFu
#define ESCHAM_MAGIC 0x3B603B60u
#define ESCHAM_MUL1  0xCBAC1FEDu // 3417055213, escham_cpu.MUL[1] (normative)

// exact round-to-nearest-even conversions, no libc/libm dependency
static inline float escham_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x03ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            e = 113; // 127 - 15 + 1
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x03ffu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7f800000u | (m << 13);
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t escham_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t s = (x >> 16) & 0x8000u;
    const uint32_t e = (x >> 23) & 0xffu;
    uint32_t m = x & 0x007fffffu;
    if (e == 0xff) { // inf / nan
        return (uint16_t)(s | 0x7c00u | (m ? 0x0200u | (m >> 13) : 0));
    }
    const int ee = (int)e - 112; // rebias to half exponent
    if (ee >= 31) {
        return (uint16_t)(s | 0x7c00u); // overflow -> inf
    }
    if (ee <= 0) { // subnormal
        if (ee < -10) {
            return (uint16_t)s;
        }
        m |= 0x00800000u;
        const int shift = 14 - ee;
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) {
            r++;
        }
        return (uint16_t)(s | r);
    }
    uint32_t r = ((uint32_t)ee << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
        r++; // carry may propagate into the exponent, that is correct
    }
    return (uint16_t)(s | r);
}

// window value -> fp16 weight, cbA codebook (all 400 projections use cb_id=1)
static uint16_t escham_lut[65536];

// (lane, step) -> (row, col) inside the 16x16 tile, same for K=2 and K=3
static uint8_t escham_tile_row[32 * 8];
static uint8_t escham_tile_col[32 * 8];

// per-lane bit extraction addresses, straight from the PTX prologue
static uint8_t escham_cur[2][32];
static uint8_t escham_prev[2][32];
static uint8_t escham_sh[2][32];

static int escham_ready = 0;

// first call may race across threads; writes are idempotent so it is benign
static void escham_init(void) {
    if (escham_ready) {
        return;
    }
    for (uint32_t w = 0; w < 65536u; w++) {
        const uint32_t t = ((w * ESCHAM_MUL1) & ESCHAM_MASK) ^ ESCHAM_MAGIC;
        const float lo = escham_fp16_to_fp32((uint16_t)(t & 0xffffu));
        const float hi = escham_fp16_to_fp32((uint16_t)(t >> 16));
        escham_lut[w] = escham_fp32_to_fp16(lo + hi);
    }
    static const int off[8][2] = { // indexed by STEP s (windows are extracted
        // at shift sh + K*s); the PTX stores steps high->low in shared memory,
        // hence the reversed display order {9,8} first
        {9, 8}, {8, 8}, {1, 8}, {0, 8}, {9, 0}, {8, 0}, {1, 0}, {0, 0},
    };
    for (int m = 0; m < 32; m++) {
        const int r0 = 2 * (m & 3);
        const int c0 = 2 * ((m >> 3) & 1) + 4 * ((m >> 4) & 1);
        const int d = (m >> 2) & 1;
        for (int s = 0; s < 8; s++) {
            escham_tile_row[m * 8 + s] = (uint8_t)(r0 + off[s][0]);
            escham_tile_col[m * 8 + s] = (uint8_t)(c0 + off[s][1] + d);
        }
    }
    for (int l = 0; l < 32; l++) {
        // K=2
        const int c2 = (l >> 1) & 15;
        escham_cur[0][l]  = (uint8_t)c2;
        escham_prev[0][l] = (uint8_t)((c2 - 1) & 15);
        escham_sh[0][l]   = (l & 1) == 0 ? 16 : 0;
        // K=3
        const int b   = l * 24;
        const int r86 = b + 791;
        const int r87 = r86 & 2016;
        escham_cur[1][l]  = (uint8_t)((((r86 >> 3) & 252) - 96) / 4);
        escham_prev[1][l] = (uint8_t)(l == 0 ? 23 : (((b + 755) >> 5) - 24));
        escham_sh[1][l]   = (uint8_t)(r87 - b - 760);
    }
    escham_ready = 1;
}

static void escham_decode_tile(const uint16_t * codes, uint16_t * out, int k3) {
    const int nw = k3 ? 24 : 16; // 8*K u32 words per tile
    uint32_t words[24];
    for (int i = 0; i < nw; i++) {
        words[i] = (uint32_t)codes[2 * i] | ((uint32_t)codes[2 * i + 1] << 16);
    }
    escham_init();
    const int ki = k3 ? 1 : 0;
    for (int m = 0; m < 32; m++) {
        const uint64_t pair = ((uint64_t)words[escham_prev[ki][m]] << 32)
                            | words[escham_cur[ki][m]];
        const int sh = escham_sh[ki][m];
        for (int s = 0; s < 8; s++) {
            const uint16_t win = (uint16_t)((pair >> (sh + (k3 ? 3 : 2) * s)) & 0xffffu);
            out[escham_tile_row[m * 8 + s] * 16 + escham_tile_col[m * 8 + s]] = escham_lut[win];
        }
    }
}

void escham_decode_tile_k2(const uint16_t * codes, uint16_t * out) {
    escham_decode_tile(codes, out, 0);
}

void escham_decode_tile_k3(const uint16_t * codes, uint16_t * out) {
    escham_decode_tile(codes, out, 1);
}

void escham_vec_had128_f32(float * x, int64_t n) {
    for (int64_t base = 0; base < n; base += 128) {
        float * v = x + base;
        for (int len = 1; len < 128; len <<= 1) {
            for (int i = 0; i < 128; i += 2 * len) {
                for (int j = i; j < i + len; j++) {
                    const float a = v[j];
                    const float b = v[j + len];
                    v[j]       = a + b;
                    v[j + len] = a - b;
                }
            }
        }
        const float sc = 0.08838834764831845f; // 1/sqrt(128)
        for (int i = 0; i < 128; i++) {
            v[i] *= sc;
        }
    }
}
