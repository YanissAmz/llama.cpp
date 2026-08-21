#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Decode one atomic 16x16 ESCHAM tile.
//
// codes: 16*K uint16 of the tile (K=2 -> 32, K=3 -> 48), as stored in the
// checkpoint ((IC//16, OC//16, 16*K) int16 plane, little endian).
// out:   256 fp16 weights, row-major 16x16 (IC x OC inside the tile).
//
// This reproduces escham_cpu.reconstruct_code (raw trellis decode, cbA
// codebook). No Hadamard, no rin/rout: those live around the matmul.
void escham_decode_tile_k2(const uint16_t * codes, uint16_t * out);
void escham_decode_tile_k3(const uint16_t * codes, uint16_t * out);

// Blockwise normalised Hadamard-128 along a fp32 vector, in place.
// n must be a multiple of 128. Matches escha_t128 (T = H/sqrt(128),
// symmetric orthogonal Sylvester transform).
void escham_vec_had128_f32(float * x, int64_t n);

#ifdef __cplusplus
}
#endif
