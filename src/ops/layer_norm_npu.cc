#include "ctranslate2/ops/layer_norm.h"
#include "../cann/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t,
                            const dim_t,
                            const dim_t,
                            StorageView& output) const {
      // This case is not implemented on CUDA, so we do not support it as well
      if (axis != input.rank() - 1 || !beta || !gamma)
        throw std::invalid_argument("Generalized LayerNorm is currently not implemented on CANN");

      aclFormat format = ACL_FORMAT_ND;
      const aclDataType aclType = cann::getACLType<T>();

      ctranslate2::cann::CannPreparation prepare;

      // LayerNorm in CANN also provides 'mean' and 'variance' as outputs, which we do not need.
      // But also we cannot instruct the operator to avoid calculating them. So we must declare them.
      const dim_t mean_variance_length = input.size()/input.shape()[axis];
      StorageView mean({mean_variance_length}, DataType::FLOAT32, D);
      StorageView variance({mean_variance_length}, DataType::FLOAT32, D);

      cann_prepare_inputdesc(prepare, aclType, input.shape().size(), input.shape().data(), format);
      cann_prepare_inputdesc(prepare, aclType, gamma->shape().size(), gamma->shape().data(), format);
      cann_prepare_inputdesc(prepare, aclType, beta->shape().size(), beta->shape().data(), format);
      cann_prepare_outputdesc(prepare, aclType, output.shape().size(), output.shape().data(), format);
      cann_prepare_outputdesc(prepare, ACL_FLOAT, mean.shape().size(), mean.shape().data(), format);
      cann_prepare_outputdesc(prepare, ACL_FLOAT, variance.shape().size(), variance.shape().data(), format);

      ACL_CALL(aclopSetAttrInt(prepare._opAttr, "begin_norm_axis", axis));
      ACL_CALL(aclopSetAttrInt(prepare._opAttr, "begin_params_axis", axis));
      ACL_CALL(aclopSetAttrFloat(prepare._opAttr, "epsilon", _epsilon));

      cann_prepare_inputbuffer(prepare, const_cast<T*>(input.data<T>()), input.size()*sizeof(T));
      cann_prepare_inputbuffer(prepare, const_cast<T*>(gamma->data<T>()), gamma->size()*sizeof(T));
      cann_prepare_inputbuffer(prepare, const_cast<T*>(beta->data<T>()), beta->size()*sizeof(T));
      cann_prepare_outputbuffer(prepare, output.data<T>(), output.size()*sizeof(T));
      cann_prepare_outputbuffer(prepare, mean.data<float>(), mean.size()*sizeof(float));
      cann_prepare_outputbuffer(prepare, variance.data<float>(), variance.size()*sizeof(float));

      ACL_CALL(aclopCompileAndExecute("LayerNorm",
                                      prepare._inputDesc.size(),
                                      prepare._inputDesc.data(),
                                      prepare._inputBuffers.data(),
                                      prepare._outputDesc.size(),
                                      prepare._outputDesc.data(),
                                      prepare._outputBuffers.data(),
                                      prepare._opAttr,
                                      ACL_ENGINE_SYS,
                                      ACL_COMPILE_SYS,
                                      NULL,
                                      cann::get_aclrt_stream()));
      ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CANN, T>(const StorageView* beta,        \
                                        const StorageView* gamma,       \
                                        const StorageView& input,       \
                                        const dim_t axis,               \
                                        const dim_t outer_size,         \
                                        const dim_t axis_size,          \
                                        const dim_t inner_size,         \
                                        StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
