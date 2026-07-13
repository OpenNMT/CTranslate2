#pragma once

#ifdef __APPLE__

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace mps {

    enum class UnaryOp {
      EXP,
      LOG,
      COS,
      SIN,
      TANH,
      RELU,
      GELU,
      GELU_TANH,
      GELU_SIGMOID,
      SIGMOID,
      SWISH,
    };

    enum class BinaryOp {
      ADD,
      SUB,
      MUL,
      MAX,
      MIN,
    };

    void gemm(DataType dtype,
              bool transpose_a,
              bool transpose_b,
              dim_t m,
              dim_t n,
              dim_t k,
              float alpha,
              const void* a,
              dim_t lda,
              const void* b,
              dim_t ldb,
              float beta,
              void* c,
              dim_t ldc);

    void gemm_batch_strided(DataType dtype,
                            bool transpose_a,
                            bool transpose_b,
                            dim_t m,
                            dim_t n,
                            dim_t k,
                            float alpha,
                            const void* a,
                            dim_t lda,
                            dim_t stridea,
                            const void* b,
                            dim_t ldb,
                            dim_t strideb,
                            float beta,
                            void* c,
                            dim_t ldc,
                            dim_t stridec,
                            dim_t batch_size);

    void gemv(DataType dtype,
              bool transpose_a,
              bool transpose_b,
              dim_t m,
              dim_t k,
              float alpha,
              const void* a,
              dim_t lda,
              dim_t stridea,
              const void* b,
              dim_t ldb,
              dim_t strideb,
              float beta,
              void* c,
              dim_t ldc,
              dim_t stridec,
              dim_t batch_size);

    void gemv_row(DataType dtype,
                  bool transpose_a,
                  bool transpose_b,
                  dim_t n,
                  dim_t k,
                  float alpha,
                  const void* a,
                  dim_t lda,
                  dim_t stridea,
                  const void* b,
                  dim_t ldb,
                  dim_t strideb,
                  float beta,
                  void* c,
                  dim_t ldc,
                  dim_t stridec,
                  dim_t batch_size);

    bool supports_gemm_type(DataType dtype);

    // Drops cached output-major weight metadata before the backing allocation
    // is released. The argument is the opaque MTLBuffer returned by utils.
    void invalidate_packed_weight_cache(void* metal_buffer);

    void fill(DataType dtype, const void* value, void* y, dim_t size);
    void indexed_fill(DataType dtype,
                      const void* value,
                      void* y,
                      const int32_t* indices,
                      dim_t size);
    void prepare_length_mask(const int32_t* lengths,
                             dim_t batch_size,
                             dim_t num_heads,
                             dim_t num_queries,
                             bool mask_future,
                             bool multi_query,
                             int32_t* mask);
    void unary(DataType dtype, UnaryOp op, const void* x, void* y, dim_t size);
    void binary(DataType dtype, BinaryOp op, const void* a, const void* b, void* c, dim_t size);
    void scalar(DataType dtype, BinaryOp op, float a, const void* x, void* y, dim_t size);

    void gather(DataType dtype,
                const void* data,
                const int32_t* indices,
                void* output,
                dim_t copy_size,
                dim_t batch_stride,
                dim_t num_indices,
                dim_t num_indices_per_batch);

    void tile(DataType dtype,
              const void* input,
              void* output,
              dim_t outer_size,
              dim_t inner_size,
              dim_t num_tiles);

    void bias_add(DataType dtype,
                  const void* bias,
                  const void* value,
                  const void* residual,
                  void* output,
                  dim_t bias_size,
                  dim_t value_size,
                  dim_t block,
                  int activation);

    void batch_broadcast(DataType dtype,
                         BinaryOp op,
                         const void* a,
                         const void* b,
                         void* c,
                         dim_t a_size,
                         dim_t b_size);

    void depth_broadcast(DataType dtype,
                         BinaryOp op,
                         const void* a,
                         const void* b,
                         void* c,
                         dim_t a_size,
                         dim_t b_size);

    void block_broadcast(DataType dtype,
                         BinaryOp op,
                         const void* a,
                         const void* b,
                         void* c,
                         dim_t block,
                         dim_t a_size,
                         dim_t b_size);

    void transpose_2d(DataType dtype, const void* a, dim_t rows, dim_t cols, void* b);
    void transpose_3d(DataType dtype, const void* a, const dim_t* dims, const dim_t* perm, void* b);
    void transpose_4d(DataType dtype, const void* a, const dim_t* dims, const dim_t* perm, void* b);

    void softmax(DataType dtype,
                 const void* input,
                 const int32_t* lengths,
                 void* output,
                 dim_t batch_size,
                 dim_t depth,
                 bool log);

    void mean(DataType dtype,
              const void* input,
              void* output,
              dim_t outer_size,
              dim_t axis_size,
              dim_t inner_size,
              bool get_sum);

    void layer_norm(DataType dtype,
                    const void* input,
                    const void* gamma,
                    const void* beta,
                    void* output,
                    dim_t outer_size,
                    dim_t axis_size,
                    dim_t inner_size,
                    float epsilon);

    void rms_norm(DataType dtype,
                  const void* input,
                  const void* gamma,
                  void* output,
                  dim_t batch_size,
                  dim_t depth,
                  float epsilon,
                  bool use_residual);

    void rotary(DataType dtype,
                const void* input,
                const void* sin,
                const void* cos,
                void* output,
                dim_t batch_size,
                dim_t max_time,
                dim_t ndims,
                dim_t depth,
                bool interleave);

    void im2col_conv1d(DataType dtype,
                       const void* input,
                       void* output,
                       dim_t batch_size,
                       dim_t groups,
                       dim_t input_length,
                       dim_t kernel_size,
                       dim_t stride,
                       dim_t padding,
                       dim_t dilation,
                       dim_t output_length,
                       dim_t k,
                       dim_t in_batch_stride,
                       dim_t in_group_stride);

    bool supports_topk(DataType dtype, dim_t k);

    void topk(DataType dtype,
              const void* input,
              void* values,
              int32_t* indices,
              dim_t batch_size,
              dim_t depth,
              dim_t k);

  }
}

#endif
