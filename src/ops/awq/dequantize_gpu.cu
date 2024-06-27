#include <ctranslate2/ops/awq/dequantize_awq.h>
#include "dequantize.cuh"
#include "cuda/helpers.h"

namespace ctranslate2 {
   namespace ops {

     __global__ void __launch_bounds__(64) dequantize_weights(const int* __restrict__ B, // 4096x64    4096 rows    64 cols
                                                              const half * __restrict__ scaling_factors,  // 32x512   32 rows    512 cols
                                                              const int* __restrict__ zeros,  // 32x64    32 rows     64 cols
                                                              half * __restrict__ C, // 4096x512    4096 rows    512 cols
                                                              int G,
                                                              int in_c,
                                                              int out_c)
     {
       if (blockIdx.z > 0) {
         B = B + blockIdx.z * in_c * out_c / 8;
         scaling_factors = scaling_factors + blockIdx.z * in_c * out_c / G;
         zeros = zeros + blockIdx.z * in_c * out_c / G / 8;
         C = C + blockIdx.z * in_c * out_c;
       }
       static constexpr uint32_t ZERO = 0x0;
       half B_shared[32 * (128 + 8)];

       half* B_shared_ptr2 = B_shared;

       int N = blockDim.x * gridDim.x;  // 2
       int col = (blockIdx.x * blockDim.x + threadIdx.x);
       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int index1 = 8 * col + 8 * row * N;  // + i (<8)
       half* C_ptr2 = C + index1;

       int index2 = col + row * N;
       const int* B_ptr2 = B + index2;

       int index3 = col + (int)(row / G) * N;
       const int* zeros_ptr2 = zeros + index3;
       int index4 = 8 * col + (int)(row / G) * N * 8;  // + i (<8)
       const half* scaling_factors_ptr2 = scaling_factors + index4;


       uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
       uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
       uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr2);
       int j=0;

       uint32_t B_loaded = *(uint32_t*)(B_ptr2 + j);
       uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
       asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

       *(uint4*)(B_shared_ptr2 + j) = B_loaded_fp16;

       for (int i=0; i<8; ++i) {
         *(C_ptr2 + i) = B_shared[i];
       }
     }

     template <Device D, typename InT, typename OutT>
     void DequantizeAwq::dequantize(const StorageView& input,
                                 const StorageView& scale,
                                 const StorageView& zero,
                                 StorageView& output) const {
       dim_t in_c = input.rank() == 2 ? input.dim(0) : input.dim(1);
       dim_t qout_c = input.rank() == 2 ? input.dim(1) : input.dim(2);
       int num_experts = input.rank() == 2 ? 1 : input.dim(0);
       int out_c = qout_c * 8;
       int G = in_c / (input.rank() == 2 ? scale.dim(0) : scale.dim(1));

       int x_thread = 0 /*thx*/;
       int y_thread = 0 /*thy*/;

       int x_blocks = 1;
       int y_blocks = 1;
       x_thread = qout_c;
       y_thread = in_c;

       x_thread = 8;
       y_thread = 8;
       x_blocks = (int)(qout_c / 8);
       y_blocks = (int)(in_c / 8);
       if (num_experts == 1) {
         output.resize({in_c, out_c});
       } else {
         output.resize({num_experts, in_c, out_c});
       }

       auto output_data = reinterpret_cast<half*>(output.data<OutT>());
       const auto scale_data = reinterpret_cast<const half*>(scale.data<OutT>());

       dim3 num_blocks(x_blocks, y_blocks, num_experts);
       dim3 threads_per_block(x_thread, y_thread);  //  col, row 64x4096

       dequantize_weights<<<num_blocks, threads_per_block>>>(input.data<InT>(), scale_data,
                                                            zero.data<InT>(), output_data, G, in_c, out_c);
     }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    DequantizeAwq::dequantize<Device::CUDA, int, T>(                    \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)

   }
}
