#include "ctranslate2/ops/transpose.h"
#ifdef CT2_WITH_CANN
#include "../cann/utils.h"
#endif
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {
    // CANN can handle transpose using StorageView directly without the need of a primitive definition
    template <typename T>
    void Transpose::handleCann(const StorageView& x, const std::vector<dim_t>& perm, StorageView& y) const {
#ifdef CT2_WITH_CANN
      const auto a_b_type = cann::getACLType<T>();
      const auto perm_type = cann::getACLType<dim_t>();
      aclFormat format = ACL_FORMAT_ND;

      cann::CannPreparation prepare;

      Shape::value_type perm_size = perm.size();
      Shape perm_shape = {perm_size};
      cann_prepare_inputdesc(prepare, a_b_type, x.shape().size(), x.shape().data(), format);
      cann_prepare_inputdesc(prepare, perm_type, perm_shape.size(), perm_shape.data(), format);
      cann_prepare_outputdesc(prepare, a_b_type, y.shape().size(), y.shape().data(), format);

      cann_prepare_inputbuffer(prepare, const_cast<T*>(x.data<T>()), x.size_in_bytes());
      cann_prepare_inputbuffer(prepare, const_cast<dim_t*>(perm.data()), perm.size()*sizeof(dim_t));
      cann_prepare_outputbuffer(prepare, y.data<T>(), y.size_in_bytes());

      ACL_CALL(aclopCompileAndExecute("Transpose",
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
 
    Transpose::Transpose(const std::vector<dim_t>& perm)
      : _perm(perm) {
    }

    void Transpose::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Transpose");
      if (x.rank() <= 1) {
        y = x;
        return;
      }

      std::vector<dim_t> perm;
      bool identity = true;
      if (_perm.empty()) {
        perm.resize(x.rank());
        for (dim_t i = 0; i < x.rank(); ++i)
          perm[i] = x.rank() - i - 1;
        identity = false;
      } else {
        assert(_perm.size() == x.rank());
        perm = _perm;
        for (dim_t i = 0; i < x.rank(); ++i) {
          if (perm[i] != i) {
            identity = false;
            break;
          }
        }
      }

      if (identity) {
        y = x;
        return;
      }

      DEVICE_AND_TYPE_DISPATCH(x.device(), x.dtype(), (compute<D, T>(x, perm, y)));
    }

  }
}
