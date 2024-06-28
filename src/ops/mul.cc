#include "ctranslate2/ops/mul.h"
#ifdef CT2_WITH_CANN
#include "../cann/utils.h"
#endif
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Mul::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Mul");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

    template <typename T>
    void Mul::handleCann(const StorageView& a, const StorageView& b, StorageView& c) const {
#ifdef CT2_WITH_CANN
        if (a.shape() != b.shape() || a.dtype() != b.dtype())
            throw std::invalid_argument("Mul: a and b have incompatible shapes or types");

        const auto aclType = cann::getACLType<T>();
        aclFormat format = ACL_FORMAT_ND;

        cann::CannPreparation prepare;

        cann_prepare_inputdesc(prepare, aclType, a.shape().size(), a.shape().data(), format);
        cann_prepare_inputdesc(prepare, aclType, b.shape().size(), b.shape().data(), format);
        cann_prepare_outputdesc(prepare, aclType, c.shape().size(), c.shape().data(), format);

        cann_prepare_inputbuffer(prepare, const_cast<T*>(a.data<T>()), a.size_in_bytes());
        cann_prepare_inputbuffer(prepare, const_cast<T*>(b.data<T>()), b.size_in_bytes());
        cann_prepare_outputbuffer(prepare, c.data<T>(), c.size_in_bytes());

        ACL_CALL(aclopCompileAndExecute("Mul",
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
#endif
    }

    template <typename T>
    void Mul::handleCannScalar(const T scalar, const StorageView& a, StorageView& c) const {
#ifdef CT2_WITH_CANN
        if (a.shape() != c.shape() || a.dtype() != c.dtype())
          throw std::invalid_argument("Muls: a and c have incompatible shapes or types");

        const auto aclType = cann::getACLType<T>();
        aclFormat format = ACL_FORMAT_ND;

        cann::CannPreparation prepare;

        // Note: CANN documentation on "scalar" value is ambiguous
        ACL_CALL(aclopSetAttrFloat(prepare._opAttr, "value", static_cast<float>(scalar)));
        cann_prepare_inputdesc(prepare, aclType, a.shape().size(), a.shape().data(), format);
        cann_prepare_outputdesc(prepare, aclType, c.shape().size(), c.shape().data(), format);

        cann_prepare_inputbuffer(prepare, const_cast<T*>(a.data<T>()), a.size_in_bytes());
        cann_prepare_outputbuffer(prepare, c.data<T>(), c.size_in_bytes());

        ACL_CALL(aclopCompileAndExecute("Muls",
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
#endif
    }
  }
}
