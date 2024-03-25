#include "ctranslate2/ops/topk.h"
#include "../cann/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      static_assert(std::is_same_v<IndexType, int32_t>); // indices have to be int32_t
      // derive types
      const auto in_out_acl_type = cann::getACLType<DataType>();
      using k_type = int32_t;
      const auto k_acl_type = cann::getACLType<k_type>();
      const auto index_acl_type = k_acl_type;
      aclFormat format = ACL_FORMAT_ND;

      cann::CannPreparation prepare;

      // x
      cann_prepare_inputdesc(prepare, in_out_acl_type, x.shape().size(), x.shape().data(), format);
      // k
      cann_prepare_inputdesc(prepare, k_acl_type, 0, nullptr, format); // handle k as scalar
      auto tmp_k = static_cast<k_type>(_k);
      cann_tensor_placement(prepare, 1, ACL_MEMTYPE_HOST);
      //values
      cann_prepare_outputdesc(prepare, in_out_acl_type, values.shape().size(), values.shape().data(), format);
      // indices
      cann_prepare_outputdesc(prepare, index_acl_type, indices.shape().size(), indices.shape().data(), format);

      // x
      cann_prepare_inputbuffer(prepare, const_cast<DataType*>(x.data<DataType>()), x.size_in_bytes());
      // k
      cann_prepare_inputbuffer(prepare, const_cast<k_type*>(&tmp_k), sizeof(k_type));
      //values
      cann_prepare_outputbuffer(prepare, const_cast<DataType*>(values.data<DataType>()), values.size_in_bytes());
      // indices
      cann_prepare_outputbuffer(prepare, const_cast<IndexType*>(indices.data<IndexType>()), indices.size_in_bytes());

      auto op_type= "TopKV2";
      // // TopK implementation
      // const int16_t kMaxTopkSize = std::numeric_limits<int16_t>::max(), kMaxK = 8, kMinK = 0;
      // if(x.size() > kMaxTopkSize && tmp_k > kMinK && tmp_k < kMaxK) {
      //   op_type = "TopK";
      //   // "dim" usage is ambiguous. According to paddle:
      //   // axis is always equal to -1
      //   // if (axis < 0)
      //   //     axis += x.dims().size();
      //    ACL_CALL(aclopSetAttrInt(prepare.opAttr_, "dim", x.rank()-1)); // axis == -1 always in TopK ctor!
      // }

      ACL_CALL(aclopCompileAndExecute(op_type,
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
    TopK::compute<Device::CANN, T, int32_t>(const StorageView& x,       \
                                            StorageView& values,        \
                                            StorageView& indices) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
