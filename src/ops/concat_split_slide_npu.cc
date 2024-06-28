#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"
#include "type_dispatch.h"
#include "../cann/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                     StorageView& output) const {
      // Tensors' descriptors have to be set in the order mentioned in the documentation.
      // For operators whose input is a list (such as concat) it is needed to set the name of each input tensor in the
      // same order of the tensor set.

      // prepare types
      using axis_type = decltype(_axis);
      static_assert(std::is_same_v<axis_type, int32_t> || std::is_same_v<axis_type, int64_t>);
      const auto axis_acl_type =  cann::getACLType<axis_type>();
      const auto in_out_acl_type = cann::getACLType<T>();

      aclFormat format = ACL_FORMAT_ND;

      cann::CannPreparation prepare;

      // input axis
      constexpr char const* axis_label = "concat_dim";
      cann_prepare_inputdesc(prepare, axis_acl_type, 0, nullptr, format); // handle axis as scalar
      cann_const_inputdesc(prepare, 0, const_cast<axis_type*>(&_axis), sizeof(axis_type)); // axis has to be set to const
      // axis_label is the first in the list of the descriptor names
      constexpr short axis_label_index = 0;
      cann_prepare_inputdescname(prepare, axis_label_index, axis_label);
      cann_prepare_inputbuffer(prepare, const_cast<axis_type*>(&_axis), sizeof(axis_type));

      // input tensors
      static const std::string desc_prefix = "x";
      for(size_t i=0; i<inputs.size(); ++i) {
        cann_prepare_inputdesc(prepare, in_out_acl_type, inputs[i]->shape().size(), inputs[i]->shape().data(), format);
        const auto descriptor_label = desc_prefix + std::to_string(i);
        cann_prepare_inputdescname(prepare, i + 1, descriptor_label.c_str()); // first element is already populated by axis_label
        cann_prepare_inputbuffer(prepare, const_cast<T*>(inputs[i]->data<T>()), inputs[i]->size_in_bytes());
      }

      // output
      cann_prepare_outputdesc(prepare, in_out_acl_type, output.shape().size(), output.shape().data(), format);
      cann_prepare_outputbuffer(prepare, output.data<T>(), output.size_in_bytes());

      // attribute is optional in Concat
      // ACL_CALL(aclopSetAttrInt(prepare.opAttr_, "N", inputs.size()));

      ACL_CALL(aclopCompileAndExecute("Concat",
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

    /**
     * Creates the input descriptors and buffers that the CANN "Split" operator needs to run correctly.
     */
    template <typename T>
    void prepare_split_inputs(const StorageView& input,
                              const int32_t axis,
                              ctranslate2::cann::CannPreparation& prepare) {
      const aclFormat format = ACL_FORMAT_ND;

      // input: split_dim. The CANN documentation for the "Split" operator specifies that 'split_dim' should be passed
      // after 'x', but in reality it should be first. 'split_dim' is a scalar, so according to the documentation we
      // need to specify its number of dimensions as 0.
      cann_prepare_inputdesc(prepare, ACL_INT32, 0, nullptr, format);

      // input: x
      cann_prepare_inputdesc(prepare, cann::getACLType<T>(), input.shape().size(), input.shape().data(), format);

      auto split_dim_sv = StorageView(axis, Device::CANN);
      cann_prepare_inputbuffer(prepare, split_dim_sv.data<int32_t>(), sizeof(int32_t));
      cann_prepare_inputbuffer(prepare, const_cast<T*>(input.data<T>()), input.size_in_bytes());
    }

    /**
     * Creates the input descriptors and buffers that the CANN "SplitV" operator needs to run correctly.
     */
    template <typename T>
    void prepare_splitv_inputs(const StorageView& input,
                               const int32_t axis,
                               const std::vector<dim_t>& size_splits,
                               ctranslate2::cann::CannPreparation& prepare) {
      static_assert(std::is_same_v<dim_t, int64_t>);
      const aclFormat format = ACL_FORMAT_ND;

      // input: x
      cann_prepare_inputdesc(prepare, cann::getACLType<T>(), input.shape().size(), input.shape().data(), format);

      // input: size_splits
      const Shape size_splits_shape = {static_cast<dim_t>(size_splits.size())};
      cann_prepare_inputdesc(prepare, ACL_INT64, size_splits_shape.size(), size_splits_shape.data(), format);

      // input: split_dim. This is a scalar, so according to the documentation we need to specify its number of
      // dimensions as 0.
      cann_prepare_inputdesc(prepare, ACL_INT32, 0, nullptr, format);

      cann_prepare_inputbuffer(prepare, const_cast<T*>(input.data<T>()), input.size_in_bytes());
      cann_prepare_inputbuffer(prepare, const_cast<dim_t*>(size_splits.data()), size_splits.size()*sizeof(dim_t));
      auto split_dim_sv = StorageView(axis, Device::CANN);
      cann_prepare_inputbuffer(prepare, split_dim_sv.data<int32_t>(), sizeof(int32_t));
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      ctranslate2::cann::CannPreparation prepare;
      const int32_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      std::string op_name;

      if (_split.empty()) {
        op_name = "Split";
        prepare_split_inputs<T>(input, axis, prepare);
      } else {
        op_name = "SplitV";
        prepare_splitv_inputs<T>(input, axis, _split, prepare);
      }

      ACL_CALL(aclopSetAttrInt(prepare._opAttr, "num_split", outputs.size()));

      // output: y
      const std::string desc_prefix = "y";
      std::string descriptor_label;
      const aclFormat format = ACL_FORMAT_ND;
      const aclDataType aclType = cann::getACLType<T>();
      for(size_t i=0; i<outputs.size(); ++i) {
        cann_prepare_outputdesc(prepare, aclType, outputs[i]->shape().size(), outputs[i]->shape().data(), format);
        descriptor_label = desc_prefix + std::to_string(i);
        cann_prepare_outputdescname(prepare, i, descriptor_label.c_str());
        cann_prepare_outputbuffer(prepare, outputs[i]->data<T>(), outputs[i]->size_in_bytes());
      }

      ACL_CALL(aclopCompileAndExecute(op_name.c_str(),
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

    template <Device D, typename T>
    void Slide::compute(const StorageView& input, StorageView& output, const dim_t& index) const {
      THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CANN, T>(const std::vector<const StorageView*>& inputs, \
                                     StorageView& output) const;        \
    template void                                                       \
    Split::compute<Device::CANN, T>(const StorageView& input,           \
                                    std::vector<StorageView*>& outputs) const;  \
    template void                                                       \
    Slide::compute<Device::CANN, T>(const StorageView& input,            \
                                   StorageView& output,                 \
                                   const dim_t& index) const;
    
    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
