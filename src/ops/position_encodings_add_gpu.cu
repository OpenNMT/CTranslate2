#include "ctranslate2/ops/position_encodings_add.h"

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T, typename AddFunc>
    __global__ void position_encodings_add_kernel(const T* input,
                                                  const T* encodings,
                                                  T* output,
                                                  const int32_t* offsets,
                                                  cuda::index_t step,
                                                  cuda::index_t max_time,
                                                  cuda::index_t depth,
                                                  const AddFunc& add_func) {
      const cuda::index_t batch = blockIdx.x / max_time;
      const cuda::index_t time = blockIdx.x % max_time;

      const int32_t offset = offsets ? offsets[batch] : 0;
      const int32_t encoding_offset = time - offset + step;

      if (encoding_offset < 0)
        return;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;
      encodings += encoding_offset * depth;

      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x) {
        output[i] = add_func(input[i], encodings[i]);
      }
    }

    template <Device D, typename T>
    void PositionEncodingsAdd::compute(const dim_t step,
                                       const StorageView* offsets,
                                       const StorageView& input,
                                       const StorageView& encodings,
                                       StorageView& output) const {
      const dim_t batch_size = input.dim(0);
      const dim_t time = input.dim(1);
      const dim_t depth = input.dim(2);

      const dim_t blocks = std::min(batch_size * time, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      position_encodings_add_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<T>()),
        cuda::device_cast(encodings.data<T>()),
        cuda::device_cast(output.data<T>()),
        offsets ? offsets->data<int32_t>() : nullptr,
        step,
        time,
        depth,
        cuda::plus<cuda::device_type<T>>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    PositionEncodingsAdd::compute<Device::CUDA, T>(const dim_t,         \
                                                   const StorageView*,  \
                                                   const StorageView&,  \
                                                   const StorageView&,  \
                                                   StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
