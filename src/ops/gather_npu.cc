#include "ctranslate2/ops/gather.h"
#include "type_dispatch.h"
#include "../cann/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      // CANN expects int32_t indices according to documentation
      using indiceType = int32_t;
      const indiceType* indices = input.data<indiceType>();
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      if (axis == batch_dims) {
        const aclDataType aclType  = cann::getACLType<T>();

        ctranslate2::cann::CannPreparation prepare;

        cann_prepare_inputdesc(prepare, aclType, data.shape().size(), data.shape().data(), ACL_FORMAT_ND);
        cann_prepare_inputdesc(prepare, ACL_INT32, input.shape().size(), input.shape().data(), ACL_FORMAT_ND);
        cann_prepare_outputdesc(prepare, aclType, output.shape().size(), output.shape().data(), ACL_FORMAT_ND);

        cann_prepare_inputbuffer(prepare, const_cast<T*>(src), data.size()*sizeof(T));
        cann_prepare_inputbuffer(prepare, const_cast<indiceType*>(indices), input.size()*sizeof(indiceType));
        cann_prepare_outputbuffer(prepare, dst, output.size()*sizeof(T));

        ACL_CALL(aclopSetAttrBool(prepare._opAttr, "validate_indices", true));
        ACL_CALL(aclopSetAttrInt(prepare._opAttr, "batch_dims", static_cast<int>(batch_dims)));

        ACL_CALL(aclopCompileAndExecute("Gather",
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
      } else {
          throw std::invalid_argument("Gather only supports indexing the first non batch dimension");
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CANN, T>(const StorageView& data,           \
                                     const StorageView& input,          \
                                     const dim_t axis,                  \
                                     const dim_t batch_dims,            \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
