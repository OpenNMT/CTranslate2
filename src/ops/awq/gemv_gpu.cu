#include "cuda/utils.h"
#include "dequantize.cuh"
#include <cublas_v2.h>
#include <ctranslate2/ops/awq/gemv.h>
#define PACK_FACTOR 8
#define WARP_SIZE 32

namespace ctranslate2 {
  namespace ops {
    template <int G>
    __global__ void __launch_bounds__(128) gemmv2_forward_4bit_cuda_m128n64k32(int split_k_iters, const half* __restrict__ A, const int* __restrict__ B, const half* __restrict__ scaling_factors, const int* zeros, int M, int IC, int OC, half* __restrict__ C)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
      static constexpr uint32_t ZERO = 0x0;
      float C_warp[64];
      __shared__ half A_shared[128 * (32 + 8)];
      __shared__ half B_shared[64 * (32 + 8)];

      // __shared__ half scaling_factors_shared[64];
      // __shared__ half zeros_shared[64];

      int j_factors1 = ((OC + 64 - 1) / 64);

      //int blockIdx_x = 0;
      int blockIdx_y = blockIdx.x % ((M + 128 - 1) / 128 * j_factors1);
      int blockIdx_z = blockIdx.x / ((M + 128 - 1) / 128 * j_factors1);

      half A_shared_warp[32];
      half B_shared_warp[16];
      for (int i_0_3_init = 0; i_0_3_init < 4; ++i_0_3_init) {
        for (int j_0_4_init = 0; j_0_4_init < 2; ++j_0_4_init) {
          for (int i = 0; i < 8; ++i) {
            C_warp[((i_0_3_init * 16) + (j_0_4_init * 8)) + i] = 0.0;
          }
        }
      }

      static constexpr int row_stride_warp = 32 * 8 / 32;
      static constexpr int row_stride_A = 4 * 32 * 8 / 32;
      static constexpr int row_stride = 4 * 32 * 8 / 32;
      const int make_divisible_multipler = 128 / G;
      const int zeros_w = make_divisible(make_divisible(IC / G, 8), make_divisible_multipler) * make_divisible_multipler;
      const int sf_w = zeros_w * 8;

      int ld_A_row = (blockIdx_y / j_factors1 * 128 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32);     // threadIdx.y is warp_id
      // bool wb_C_flag = (threadIdx.x / 4) < M;

      const half* A_ptr = A
                    + (((int)blockIdx_y) / j_factors1 * 128 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
                    + (((int)threadIdx.x) % (32 / 8)) * 8;

      const int* B_ptr = B
                   + ((int)threadIdx.y) * (IC / 8) * 8
                   + (((int)threadIdx.x) / (32 / 8)) * (IC / 8)
                   + (((int)blockIdx_y) % j_factors1) * 64 * (IC / 8)
                   + (((int)threadIdx.x) % (32 / 8)) * 1;

// Why * 1 in the above line?

      half* A_shared_ptr = A_shared
                           + ((int)threadIdx.y) * row_stride_warp * (32 + 8)
                           + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                           + (((int)threadIdx.x) % (32 / 8) ) * 8;

      half* B_shared_ptr = B_shared
                           + ((int)threadIdx.y) * (row_stride / 4) * (32 + 8)
                           + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                           + (((int)threadIdx.x) % (32 / 8)) * 8;


      const int* zeros_ptr = zeros
                       + ((int)threadIdx.y) * zeros_w * 8
                       + (((int)threadIdx.x) / (32 / 8)) * zeros_w
                       + (((int)blockIdx_y) % j_factors1) * 64 * zeros_w
                       // this term is zero
                       + (((int)threadIdx.x) % (32 / 8)) / G ;

      const half* scaling_factors_ptr = scaling_factors
                                  + ((int)threadIdx.y) * sf_w * 8
                                  + (((int)threadIdx.x) / (32 / 8)) * sf_w
                                  + (((int)blockIdx_y) % j_factors1) * (64) * sf_w
                                  // this term is zero
                                  + (((int)threadIdx.x) % (32 / 8)) * 8 / G;


      // Haotian: TBD, check, May 29 11:46 AM PST
      half* C_ptr = C
                    + static_cast<long long>(blockIdx_z) * M * OC        // blockIdx_z -> split_k dim
                    + (((int)blockIdx_y) % j_factors1) * 64
                    + (((int)threadIdx.y) / 2) * 32
                    + (((int)threadIdx.x) % 4) * 2;

      // preload s.f. and zeros
      int k_bound = make_divisible(IC / 32, split_k_iters); // (IC / 32 + split_k_iters - 1) / split_k_iters;
      if ((k_bound - 1) * 32 + blockIdx_z >= IC) k_bound -= 1;

      // TODO (Haotian): load scales and zero points to smem

      for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
        int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
        __syncthreads();
        // TODO: Haotian: Here we assume M % cta_M = 0.
        for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
        {
          if (ld_A_row + ax0_ax1_fused_0 * row_stride_A < M)
          {
            *(uint4*)(A_shared_ptr + ax0_ax1_fused_0 * row_stride_A * 40) = *(uint4*)(A_ptr + (ax0_ax1_fused_0 * row_stride_A * IC) + (k_0_0 * 32));
          }
          else
          {
            *(uint4*)(A_shared_ptr + ax0_ax1_fused_0 * row_stride_A * 40) = make_uint4(0, 0, 0, 0);
          }
        }


        const int* zeros_ptr_local = zeros_ptr + k_0_0 * 32 / G / 8;
        const half* scaling_factors_ptr_local = scaling_factors_ptr + k_0_0 * 32 / G;

        // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
        const int* B_ptr_local = B_ptr + k_0_0 * (32 / 8);

        for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {

          // B: 32 x 136 (128+8) float16
          // each warp: 32 x 4
          // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
          // row stride in shared memory: (NWARPS * 32 * 8 / cta_N)
          int B_loaded_current = *(B_ptr_local + ax0_ax1_fused_0 * row_stride * (IC / 8));
          int zeros_loaded = *(zeros_ptr_local + ax0_ax1_fused_0 * row_stride * zeros_w);
          zeros_loaded >>= ((k_0_0 * 32 / G) % 8) * 4;
          float current_zeros = (float)(zeros_loaded & 0xF);
          half scaling_factors_loaded = *(scaling_factors_ptr_local + ax0_ax1_fused_0 * row_stride * sf_w);
          half B_loaded_fp16[8];
#pragma unroll
          for (int ic_1 = 0; ic_1 < 8; ic_1++){
            float current_single_weight_fp = (float)(B_loaded_current & 0xF);
            half dequantized_weight = __float2half(__half2float(scaling_factors_loaded) * (current_single_weight_fp - current_zeros));
            B_loaded_current = B_loaded_current >> 4;
            B_loaded_fp16[ic_1] = dequantized_weight;
          }
          // write back
          *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (32 + 8)) = *reinterpret_cast<uint4*>(B_loaded_fp16);
        }
        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
          for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
            {
              unsigned int addr;
              asm volatile(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)((&(A_shared[((((((int)threadIdx.y) & 1) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
                );
              asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[3])
                : "r"(addr)
                );
            }
          }

          for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
            {
              unsigned int addr;
              asm volatile(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void *)((&(B_shared[((((((int)threadIdx.y) >> 1) * 1280) + (ax0_0_1 * 640)) + (k_0_1 * 16))])) + ((((((int)threadIdx.x) >> 4) * 320) + ((((int)threadIdx.x) & 7) * 40)) + (((((int)threadIdx.x) & 15) >> 3) * 8))))
                );
              asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[3])
                : "r"(addr)
                );
            }
          }

          for (int i_0_3 = 0; i_0_3 < 4; ++i_0_3) {
            for (int j_0_4 = 0; j_0_4 < 2; ++j_0_4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
          {
            asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3]));
          }

          {
            asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8 + 4)))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[3]));
          }

          {
            asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3]));
          }

          {
            asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8 + 4)))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8) + 4)))[3]));
          }
#else
              {
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                  "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                  :  "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3])
                  : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3]));
              }

              {
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                  "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                  :  "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[3])
                  : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[3]));
              }
#endif
            }
          }
        }
      }

// Haotian: Here (May 29 11:46AM PST)
// TODO: Shang: Hoist loop invariance.
      for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {
        for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
          for (int local_id = 0; local_id < 8; ++local_id) {
            int row_offset = (((int)blockIdx_y) / j_factors1) * 128 + (threadIdx.y % 2) * 64 + ax0_0_2 * 16 + (local_id % 4) / 2 * 8 + ((int)threadIdx.x) / 4;
            if (row_offset < M)
            {
              *(C_ptr + ax1_0 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax0_0_2 * 16) + (ax1_0 * 8) + local_id]);
            }
          }
        }
      }
#endif
    }

    // Reduce sum within the warp using the tree reduction algorithm.
    __device__ __forceinline__ float warp_reduce_sum(float sum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
#pragma unroll
      for(int i = 4; i >= 0; i--){
        sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
      }
      /*
      // Equivalent to the following tree reduction implementation:
      sum += __shfl_down_sync(0xffffffff, sum, 16);
      sum += __shfl_down_sync(0xffffffff, sum, 8);
      sum += __shfl_down_sync(0xffffffff, sum, 4);
      sum += __shfl_down_sync(0xffffffff, sum, 2);
      sum += __shfl_down_sync(0xffffffff, sum, 1);
      */
#endif
      return sum;
    }

/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
    __global__ void gemv_kernel_g64(
      const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs,
      const int IC, const int OC){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
      const int group_size = 64;
      float psum = 0;
      const int batch_idx = blockIdx.z;
      const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
      const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
      half* outputs = _outputs + batch_idx * OC;
      // This is essentially zeros_w.
      const int num_groups_packed = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
      const int weight_w = IC / PACK_FACTOR;
      // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
      const int zeros_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
      // consistent with input shape
      const int sf_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;
      // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w, sf_w);
      // tile size: 4 OC x 1024 IC per iter
      for(int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2; packed_group_idx++){
        // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
        uint64_t packed_zeros = *reinterpret_cast<const uint64_t*>(zeros + oc_idx * zeros_w + packed_group_idx * 2);
        uint32_t packed_weights[4];
        // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
        *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
        // load scaling factors
        // g64: two threads -> 64 numbers -> 1 group; 1 warp = 16 groups.
        float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
        float current_zeros = (float)((packed_zeros >> (threadIdx.x / 2 * 4)) & 0xF);
        int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4;
        const float4* inputs_ptr = inputs + inputs_ptr_delta;
        // multiply 32 weights with 32 inputs
#pragma unroll
        for (int ic_0 = 0; ic_0 < 4; ic_0++){
          // iterate over different uint32_t packed_weights in this loop
          uint32_t current_packed_weight = packed_weights[ic_0];
          half packed_inputs[PACK_FACTOR];
          // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
          if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
            *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
#pragma unroll
            for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
              // iterate over 8 numbers packed within each uint32_t number
              float current_single_weight_fp = (float)(current_packed_weight & 0xF);
              float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
              //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
              psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
              current_packed_weight = current_packed_weight >> 4;
            }
          }
        }
      }
      psum = warp_reduce_sum(psum);
      if (threadIdx.x == 0) {
        outputs[oc_idx] = __float2half(psum);
      }
#endif
    }


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
    __global__ void gemv_kernel_g128(
      const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs,
      const int IC, const int OC){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
      const int group_size = 128;
      float psum = 0;
      const int batch_idx = blockIdx.z;
      const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
      const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
      half* outputs = _outputs + batch_idx * OC;
      const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
      const int weight_w = IC / PACK_FACTOR;
      // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
      const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
      // consistent with input shape
      const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;
      //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
      // tile size: 4 OC x 1024 IC per iter
      for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++){
        // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
        uint32_t packed_zeros = *(zeros + oc_idx * zeros_w + packed_group_idx);
        uint32_t packed_weights[4];
        // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
        *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
        // load scaling factors
        // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
        float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
        float current_zeros = (float)((packed_zeros >> (threadIdx.x / 4 * 4)) & 0xF);
        int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4;
        const float4* inputs_ptr = inputs + inputs_ptr_delta;
        // multiply 32 weights with 32 inputs
#pragma unroll
        for (int ic_0 = 0; ic_0 < 4; ic_0++){
          // iterate over different uint32_t packed_weights in this loop
          uint32_t current_packed_weight = packed_weights[ic_0];
          half packed_inputs[PACK_FACTOR];
          // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
          if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
            *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
#pragma unroll
            for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
              // iterate over 8 numbers packed within each uint32_t number
              float current_single_weight_fp = (float)(current_packed_weight & 0xF);
              float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
              //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
              psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
              current_packed_weight = current_packed_weight >> 4;
            }
          }
        }
      }
      psum = warp_reduce_sum(psum);
      if (threadIdx.x == 0) {
        outputs[oc_idx] = __float2half(psum);
      }
#endif
    }

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv(const StorageView& a,
                                const StorageView& b,
                                const StorageView& scale,
                                const StorageView& zero,
                                StorageView& c) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
      dim_t num_in_channels = a.dim(-1);
      dim_t num_in_feats = a.size() / num_in_channels;
      if (a.rank() == 2)
        c.resize({num_in_feats, b.dim(0)});
      else if (a.rank() == 3)
        c.resize({a.dim(0), a.dim(1), b.dim(0)});

      const auto a_data = reinterpret_cast<const float4*>(a.data<In>());
      const auto b_data = reinterpret_cast<const uint32_t*>(b.data<int>());
      auto output_data = reinterpret_cast<half*>(c.data<In>());
      const auto scale_data = reinterpret_cast<const half*>(scale.data<In>());
      const auto zero_data = reinterpret_cast<const uint32_t*>(zero.data<int>());
      dim_t group_size = num_in_channels / scale.dim(-1);

      dim_t num_out_feats = num_in_feats;
      dim_t num_out_channels = c.dim(-1);
      dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
      dim3 num_threads(32, 4);
      if (group_size == 64)
      {
        gemv_kernel_g64<<<num_blocks, num_threads, 0, cuda::get_cuda_stream()>>>(
          // pointers
          a_data, b_data, zero_data, scale_data, output_data,
          // constants
          num_in_channels, num_out_channels
        );
      }
      else if (group_size == 128)
      {
        gemv_kernel_g128<<<num_blocks, num_threads, 0, cuda::get_cuda_stream()>>>(
          // pointers
          a_data, b_data, zero_data, scale_data, output_data,
          // constants
          num_in_channels, num_out_channels
        );
      }
#endif
    }

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv2(const StorageView& a,
                          const StorageView& b,
                          const StorageView& scale,
                          const StorageView& zero,
                          StorageView& c) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
      dim_t num_in_channels = a.dim(-1);
      dim_t num_in_feats = a.size() / num_in_channels;
      dim_t split_k_iters = 8;

      if (a.rank() == 2)
        c.resize({split_k_iters, num_in_feats, b.dim(0)});
      else if (a.rank() == 3)
        c.resize({split_k_iters, a.dim(0), a.dim(1), b.dim(0)});

      dim_t num_out_feats = num_in_feats;
      dim_t num_out_channels = c.dim(-1);

      const auto a_data = reinterpret_cast<const half*>(a.data<In>());
      const auto b_data = reinterpret_cast<const int*>(b.data<int>());
      auto output_data = reinterpret_cast<half*>(c.data<In>());
      const auto scale_data = reinterpret_cast<const half*>(scale.data<In>());
      const auto zero_data = reinterpret_cast<const int*>(zero.data<int>());
      dim_t group_size = num_in_channels / scale.dim(-1);

      if (num_out_channels % 64 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 64");
      if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");

      int j_factors1 = num_out_channels / 64 / 1;
      dim3 num_blocks((num_out_feats + 128 - 1) / 128 * j_factors1 * split_k_iters);

      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 4);
      if (group_size == 128)
      {
        gemmv2_forward_4bit_cuda_m128n64k32<128><<<num_blocks, threads_per_block>>>(
          split_k_iters, a_data, b_data, scale_data, zero_data, num_in_feats, num_in_channels, num_out_channels, output_data);
      }
      else if (group_size == 64)
      {
        gemmv2_forward_4bit_cuda_m128n64k32<64><<<num_blocks, threads_per_block>>>(
          split_k_iters, a_data, b_data, scale_data, zero_data, num_in_feats, num_in_channels, num_out_channels, output_data);
      }
      else
      {
        throw std::invalid_argument("Group size temporarily not supported.");
      }
#endif
    }


#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemvAwq::compute_gemv2<Device::CUDA, T, int>(                       \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    GemvAwq::compute_gemv<Device::CUDA, T, int>(                        \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}