#include "ctranslate2/ops/gemm.h"

#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "mps/utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    bool Gemm::try_compute_with_epilogue<Device::MPS>(
      const StorageView& a,
      const StorageView& b,
      StorageView& c,
      const StorageView* a_shift_compensation,
      const StorageView* bias,
      const StorageView* residual) const {
      if (mps::profile_enabled()) {
        const dim_t profile_k = a.dim(_trans_a ? -2 : -1);
        const dim_t profile_n = b.dim(_trans_b ? -2 : -1);
        const dim_t profile_m = a.size() / profile_k;
        const std::string detail =
          "gemm_epilogue m=" + std::to_string(profile_m)
          + " n=" + std::to_string(profile_n)
          + " bias=" + std::to_string(bias != nullptr)
          + " residual=" + std::to_string(residual != nullptr)
          + " activation=" + std::to_string(mps::activation_code(_activation_type));
        mps::record_profile_detail(detail.c_str());
      }

      if (a.dtype() != DataType::FLOAT16
          || (bias && bias->dtype() != DataType::FLOAT16)
          || (residual && residual->dtype() != DataType::FLOAT16)
          || a_shift_compensation)
        return false;

      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;
      if ((bias && bias->size() != n)
          || (residual && residual->size() != m * n))
        return false;

      Shape output_shape(a.shape());
      output_shape[output_shape.size() - 2] = a.dim(_trans_a ? -1 : -2);
      output_shape[output_shape.size() - 1] = n;
      c.resize(std::move(output_shape));

      if (mps::gemm_with_epilogue(DataType::FLOAT16,
                                  _trans_a,
                                  _trans_b,
                                  m,
                                  n,
                                  k,
                                  _alpha,
                                  a.data<float16_t>(),
                                  _trans_a ? m : k,
                                  b.data<float16_t>(),
                                  _trans_b ? k : n,
                                  _beta,
                                  c.data<float16_t>(),
                                  n,
                                  bias ? bias->data<float16_t>() : nullptr,
                                  residual ? residual->data<float16_t>() : nullptr,
                                  mps::activation_code(_activation_type)))
        return true;

      if (m == 1 && _trans_b && bias) {
        mps::gemv_row(DataType::FLOAT16,
                      _trans_a,
                      _trans_b,
                      n,
                      k,
                      _alpha,
                      a.data<float16_t>(),
                      _trans_a ? m : k,
                      0,
                      b.data<float16_t>(),
                      _trans_b ? k : n,
                      0,
                      _beta,
                      c.data<float16_t>(),
                      n,
                      0,
                      1,
                      bias->data<float16_t>(),
                      residual ? residual->data<float16_t>() : nullptr,
                      mps::activation_code(_activation_type));
        return true;
      }

      return false;
    }

    template<>
    bool Gemm::try_weight_only_int8<Device::MPS>(const StorageView& input,
                                                  const StorageView& weight,
                                                  const StorageView& scale,
                                                  StorageView& output,
                                                  const ActivationType* activation,
                                                  const StorageView* bias,
                                                  const StorageView* residual) const {
      if (input.dtype() != DataType::FLOAT16
          || weight.dtype() != DataType::INT8
          || (bias && bias->dtype() != DataType::FLOAT16)
          || (residual && residual->dtype() != DataType::FLOAT16))
        return false;

      const dim_t k = input.dim(-1);
      const dim_t n = weight.dim(0);
      const dim_t m = input.size() / k;
      if ((bias && bias->size() != n)
          || (residual && residual->size() != m * n))
        return false;

      Shape output_shape(input.shape());
      output_shape.back() = n;
      output.resize(std::move(output_shape));
      return mps::gemm_weight_only_int8_with_epilogue(
        input.data<float16_t>(),
        weight.data<int8_t>(),
        scale.data<float>(),
        scale.size(),
        bias ? bias->data<float16_t>() : nullptr,
        residual ? residual->data<float16_t>() : nullptr,
        output.data<float16_t>(),
        m,
        n,
        k,
        mps::activation_code(activation));
    }

  }
}
