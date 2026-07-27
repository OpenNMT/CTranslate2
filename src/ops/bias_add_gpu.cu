#include "ctranslate2/ops/bias_add.h"

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct plus3 {
        __device__
        T operator()(const T& a, const thrust::tuple<T, T>& bc) const {
            return a + thrust::get<0>(bc) + thrust::get<1>(bc);
        }
    };

    template <typename T, typename Op, typename Epilogue>
    struct op_epilogue {
      Op op;
      Epilogue epilogue;

      __device__ T operator()(const T& lhs, const T& rhs) const {
        return epilogue(op(lhs, rhs));
      }
    };

    template <typename T>
    void trinary_add(const T* a, const T* b, const T* c, T* d,
                      dim_t width, dim_t depth, dim_t size) {
      auto index_a = cuda::repeat_vec_block<cuda::index_t>(width, depth);
      auto index_it = thrust::make_transform_iterator(thrust::counting_iterator<cuda::index_t>(0), index_a);
      auto a_it = thrust::make_permutation_iterator(cuda::device_cast(a), index_it);
      auto bc_it = thrust::make_zip_iterator(thrust::make_tuple(cuda::device_cast(b), cuda::device_cast(c)));
      THRUST_CALL(thrust::transform, a_it, a_it + size, bc_it, cuda::device_cast(d), plus3<cuda::device_type<T>>());
    }

    template <typename DeviceT, typename T, typename Epilogue>
    void bias_add(const T* x, const T* b, T* y, dim_t numel, dim_t width, dim_t depth, Epilogue epilogue) {
      cuda::binary_transform(b, x, y, numel,
                             op_epilogue<DeviceT, cuda::plus<DeviceT>, Epilogue>(),
                             cuda::repeat_vec_block<cuda::index_t>(width, depth));
    }

    // _activation_type(value + bias + residual)
    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output,
                          const StorageView* residual) const {
      const dim_t numel = value.size();
      const dim_t depth = bias.size();
      const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
      dim_t width = 1;
      for (dim_t i = axis + 1; i < value.rank(); ++i)
        width *= value.dim(i);

      using DeviceT = cuda::device_type<T>;
      const T* x = value.data<T>();
      const T* b = bias.data<T>();
      T* y = output.data<T>();

      if (residual) {
        trinary_add(b, x, residual->data<T>(), y, width, depth, value.size());
        if (_activation_type) // fuse if ever used
          get_activation_op(*_activation_type)(output, output);

      } else if (!_activation_type) {
        primitives<D>::add_block_broadcast(b, x, y, width, depth, value.size());

      } else {
        switch (*_activation_type) {

        case ActivationType::ReLU:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::relu_func<DeviceT>());
          break;

        case ActivationType::GELU:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::gelu_func<DeviceT>());
          break;

        case ActivationType::GELUTanh:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::gelu_tanh_func<DeviceT>());
          break;

        case ActivationType::GELUSigmoid:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::gelu_sigmoid_func<DeviceT>());
          break;

        case ActivationType::Sigmoid:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::sigmoid_func<DeviceT>());
          break;

        case ActivationType::Swish:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::swish_func<DeviceT>());
          break;

        case ActivationType::Tanh:
          bias_add<DeviceT>(x, b, y, numel, width, depth, cuda::tanh_func<DeviceT>());
          break;
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    BiasAdd::compute<Device::CUDA, T>(const StorageView& value,         \
                                      const StorageView& bias,          \
                                      StorageView& output,              \
                                      const StorageView* residual) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
