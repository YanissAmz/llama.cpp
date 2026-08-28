#include "common.cuh"

// Chemin de tri accelere par bibliotheque.
// CUDA le prend par CUB. ROCm l'avait perdu: common.cuh exclut GGML_CUDA_USE_CUB
// sous HIP, si bien que supports_op() refusait TOP_K/ARGSORT des que ne[0] > 1024
// et que l'ordonnanceur sortait l'operation sur le CPU. Or hipCUB expose la meme
// API. On declare donc un macro propre au tri, sans toucher a GGML_CUDA_USE_CUB
// qui pilote aussi mean/sum/cumsum.
#if defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP)
#    define GGML_CUDA_SORT_CUB
#endif

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_CUDA_SORT_CUB
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows);
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_SORT_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);
