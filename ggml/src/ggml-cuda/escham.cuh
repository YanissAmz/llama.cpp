#pragma once
#include <stdint.h>
struct ggml_backend_cuda_context;
struct ggml_tensor;

// M5-C ESCHAM CUDA — tuile 16x16 bande-major, miroir exact ggml-escham.c
// Pas de Hadamard ici. Reference: ggml/src/ggml-escham.c (fige) et ggml-cpu.c
// branche ESCHAM mul_mat (band-major, 16*K int16 -> 256 fp16 par tuile).

// Raw pointer dispatch (stream may be cudaStream_t* ; nullptr = default)
void ggml_cuda_escham_mul_mat_raw(const void * src0_data, const void * src1_data, void * dst_data,
                                  int64_t ic, int64_t oc, int64_t nc, int is_k3, void * stream);

// ggml tensor dispatch
void ggml_cuda_escham_mul_mat(struct ggml_backend_cuda_context & ctx,
                              const struct ggml_tensor * src0,
                              const struct ggml_tensor * src1,
                              struct ggml_tensor * dst);

bool ggml_cuda_escham_supports_type(int ggml_type);
