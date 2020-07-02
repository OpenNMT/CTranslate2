#include "ctranslate2/layers/common.h"

#include <cmath>

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace layers {

    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _scale(model.get_flag_with_default(scope + "/multiply_by_sqrt_depth", true)
               ? new StorageView(static_cast<float>(sqrt(_embeddings.dim(-1))))
               : nullptr) {
    }

    void Embeddings::operator()(const StorageView& ids,
                                StorageView& output) const {
      PROFILE("Embeddings");
      if (_embeddings.dtype() == DataType::INT16 || _embeddings.dtype() == DataType::INT8) {
        const auto device = output.device();
        StorageView gathered(_embeddings.dtype(), device);
        _gather_op(_embeddings, ids, gathered);
        if (_qscale->is_scalar())
          ops::Dequantize()(gathered, *_qscale, output);
        else {
          StorageView scale(_qscale->dtype(), device);
          _gather_op(*_qscale, ids, scale);
          ops::Dequantize()(gathered, scale, output);
        }
      } else {
        _gather_op(_embeddings, ids, output);
      }

      if (_scale)
        ops::Mul()(output, *_scale, output);
    }


    static const StorageView& get_linear_weight(const models::Model& model,
                                                const std::string& scope,
                                                bool* is_packed) {
      const StorageView* weight = model.get_variable_if_exists(scope + "/weight_packed");
      if (weight) {
        *is_packed = true;
        return *weight;
      }
      *is_packed = false;
      return model.get_variable(scope + "/weight");
    }

    Dense::Dense(const models::Model& model, const std::string& scope)
      : _packed_weight(false)
      , _weight(get_linear_weight(model, scope, &_packed_weight))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _u8_shift_compensation(model.get_variable_if_exists(scope + "/weight_compensation"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), DataType::FLOAT)
      , _partial_qscale(_weight.device(), DataType::FLOAT)
      , _partial_u8_shift_compensation(_weight.device(), DataType::INT32)
      , _gemm_op(/*alpha=*/1,
                 /*beta=*/0,
                 /*trans_a=*/false,
                 /*trans_b=*/true,
                 /*a_is_packed=*/false,
                 _packed_weight)
      , _quantize_op(/*int16_scale_type=*/ops::Quantize::ScaleType::GLOBAL,
                     /*shit_to_uint8=*/bool(_u8_shift_compensation)) {
    }

    void Dense::mask_weights(const StorageView& index) {
      if (_packed_weight)
        throw std::runtime_error("Can't mask pre-packed weight");
      ops::Gather()(_weight, index, _partial_weight);
      if (_u8_shift_compensation)
        ops::Gather()(*_u8_shift_compensation, index, _partial_u8_shift_compensation);
      if (_bias)
        ops::Gather()(*_bias, index, _partial_bias);
      if (_qscale && !_qscale->is_scalar())
        ops::Gather()(*_qscale, index, _partial_qscale);
    }

    void Dense::reset_mask() {
      _partial_weight.clear();
      _partial_bias.clear();
      _partial_qscale.clear();
      _partial_u8_shift_compensation.clear();
    }

    void Dense::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Dense");
      const StorageView* qscale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      const StorageView* weight = _partial_weight.empty() ? &_weight : &_partial_weight;
      const StorageView* bias = _partial_bias.empty() ? _bias : &_partial_bias;
      const StorageView* compensation = (_partial_u8_shift_compensation.empty()
                                         ? _u8_shift_compensation
                                         : &_partial_u8_shift_compensation);

      if (_weight.dtype() == DataType::INT16 || _weight.dtype() == DataType::INT8) {
        const auto device = input.device();
        StorageView qinput(_weight.dtype(), device);
        StorageView qinput_scale(_qscale->dtype(), device);
        StorageView qoutput(DataType::INT32, device);
        _quantize_op(input, qinput, qinput_scale);
        _gemm_op(qinput, *weight, qoutput, compensation);
        _dequantize_op(qoutput,
                       qinput_scale,
                       *qscale,
                       /*trans_a=*/false,
                       /*trans_b=*/true,
                       output);
      } else {
        _gemm_op(input, *weight, output);
      }

      if (bias) {
        DEVICE_DISPATCH(output.device(),
                        primitives<D>::add_batch_broadcast(bias->data<float>(),
                                                           output.data<float>(),
                                                           bias->size(),
                                                           output.size()));
      }
    }


    LayerNorm::LayerNorm(const models::Model& model, const std::string& scope)
      : _beta(model.get_variable(scope + "/beta"))
      , _gamma(model.get_variable(scope + "/gamma")) {
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      _norm_op(_beta, _gamma, input, output);
    }

  }
}
