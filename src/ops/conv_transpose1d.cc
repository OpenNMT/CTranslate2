#include "ctranslate2/ops/conv_transpose1d.h"

#include <limits>
#include <stdexcept>

#include "device_dispatch.h"
#include "storage_view_overlap.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      dim_t checked_product(dim_t a, dim_t b) {
        if (a && b > std::numeric_limits<dim_t>::max() / a)
          throw std::invalid_argument("ConvTranspose1D dimensions overflow");
        return a * b;
      }

      dim_t checked_sum(dim_t a, dim_t b) {
        if (b > std::numeric_limits<dim_t>::max() - a)
          throw std::invalid_argument("ConvTranspose1D dimensions overflow");
        return a + b;
      }
    }

    ConvTranspose1D::ConvTranspose1D(dim_t stride, dim_t padding, dim_t output_padding,
                                     dim_t dilation, dim_t groups)
      : _stride(stride)
      , _padding(padding)
      , _output_padding(output_padding)
      , _dilation(dilation)
      , _groups(groups) {
      if (stride <= 0 || dilation <= 0 || groups <= 0 || padding < 0 || output_padding < 0
          || (output_padding >= stride && output_padding >= dilation))
        throw std::invalid_argument("ConvTranspose1D invalid convolution parameters");
    }

    void ConvTranspose1D::operator()(const StorageView& input, const StorageView& weight,
                                     StorageView& output) const {
      forward(input, weight, nullptr, output);
    }

    void ConvTranspose1D::operator()(const StorageView& input, const StorageView& weight,
                                     const StorageView& bias, StorageView& output) const {
      forward(input, weight, &bias, output);
    }

    Shape ConvTranspose1D::validate_input(const StorageView& input, const StorageView& weight,
                                          const StorageView* bias,
                                          const StorageView& output) const {
      for (const auto* tensor : {&input, &weight, static_cast<const StorageView*>(&output), bias}) {
        if (tensor && (tensor->device() != input.device()
                       || tensor->device_index() != input.device_index()
                       || tensor->dtype() != DataType::FLOAT32))
          throw std::invalid_argument("ConvTranspose1D requires float32 tensors on the same device");
      }
      if (input.rank() != 3 || weight.rank() != 3)
        throw std::invalid_argument("ConvTranspose1D expects rank-3 input and weight");
      for (const auto* tensor : {&input, &weight}) {
        for (const auto dim : tensor->shape()) {
          if (dim <= 0)
            throw std::invalid_argument("ConvTranspose1D dimensions must be positive");
        }
      }
      const dim_t batch = input.dim(0), cin = input.dim(1), time = input.dim(2);
      const dim_t cout_group = weight.dim(1), kernel = weight.dim(2);
      if (weight.dim(0) != cin || cin % _groups != 0)
        throw std::invalid_argument("ConvTranspose1D incompatible input channels/groups");
      const dim_t cout = checked_product(cout_group, _groups);
      dim_t length =
          checked_sum(checked_product(time - 1, _stride), checked_product(kernel - 1, _dilation));
      length = checked_sum(length, checked_sum(_output_padding, 1));
      const auto crop = checked_product(_padding, 2);
      if (length <= crop)
        throw std::invalid_argument("ConvTranspose1D output length must be positive");
      length -= crop;
      checked_product(checked_product(checked_product(batch, cout), length), sizeof(float));
      if (bias && bias->shape() != Shape{cout})
        throw std::invalid_argument("ConvTranspose1D bias must have shape [Cout]");
      for (const auto* tensor : {&input, &weight, bias}) {
        if (tensor && (tensor == &output || storage_views_overlap(*tensor, output)))
          throw std::invalid_argument("ConvTranspose1D output must not alias inputs");
      }

      return {batch, cout, length};
    }

    void ConvTranspose1D::forward(const StorageView& input, const StorageView& weight,
                                  const StorageView* bias, StorageView& output) const {
      PROFILE("ConvTranspose1D");
      const auto shape = validate_input(input, weight, bias, output);
      const ScopedDeviceSetter setter(input.device(), input.device_index());
      output.resize(shape);
      DEVICE_DISPATCH(input.device(), (compute<D, float>(input, weight, bias, output)));
    }

  }
}
