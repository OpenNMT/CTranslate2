#include "ctranslate2/ops/layer_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    LayerNorm::LayerNorm(const dim_t axis, const float epsilon)
      : _axis(axis)
      , _epsilon(epsilon)
    {
    }

    template<>
    bool LayerNorm::supports_fused_bias_residual<Device::CPU>(DataType) const {
      return false;
    }

    template<>
    bool LayerNorm::supports_fused_bias_residual<Device::CUDA>(DataType) const {
      return false;
    }

    template<>
    bool LayerNorm::try_fused_bias_residual<Device::CPU>(const StorageView&,
                                                          const StorageView&,
                                                          const StorageView&,
                                                          const StorageView&,
                                                          const StorageView&,
                                                          StorageView&) const {
      return false;
    }

    template<>
    bool LayerNorm::try_fused_bias_residual<Device::CUDA>(const StorageView&,
                                                           const StorageView&,
                                                           const StorageView&,
                                                           const StorageView&,
                                                           const StorageView&,
                                                           StorageView&) const {
      return false;
    }

    bool LayerNorm::supports_fused_bias_residual(Device device, DataType dtype) const {
      bool supported = false;
      DEVICE_DISPATCH(device, supported = supports_fused_bias_residual<D>(dtype));
      return supported;
    }

    bool LayerNorm::try_fused_bias_residual(const StorageView& beta,
                                             const StorageView& gamma,
                                             const StorageView& input,
                                             const StorageView& bias,
                                             const StorageView& residual,
                                             StorageView& output) const {
      bool supported = false;
      DEVICE_DISPATCH(input.device(),
                      supported = try_fused_bias_residual<D>(beta,
                                                             gamma,
                                                             input,
                                                             bias,
                                                             residual,
                                                             output));
      return supported;
    }

    void LayerNorm::operator()(const StorageView& beta,
                               const StorageView& gamma,
                               const StorageView& input,
                               StorageView& output) const {
      operator()(&beta, &gamma, input, output);
    }

    void LayerNorm::operator()(StorageView& input) const {
      operator()(input, input);
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      operator()(nullptr, nullptr, input, output);
    }

    void LayerNorm::operator()(const StorageView* beta,
                               const StorageView* gamma,
                               const StorageView& input,
                               StorageView& output) const {
      PROFILE("LayerNorm");
      output.resize_as(input);

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t axis_size = input.dim(axis);

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis + 1; i < input.rank(); ++i)
        inner_size *= input.dim(i);

      DEVICE_AND_FLOAT_DISPATCH("LayerNorm", input.device(), input.dtype(),
                                (compute<D, T>(beta,
                                               gamma,
                                               input,
                                               axis,
                                               outer_size,
                                               axis_size,
                                               inner_size,
                                               output)));
    }

  }
}
