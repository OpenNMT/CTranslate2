#include "ctranslate2/ops/softmax.h"
#include "../cann/utils.h"

namespace ctranslate2 {
  namespace ops {

    template<typename T>
    void run_softmax(const StorageView& input,
                     StorageView& output,
                     bool log){
      const aclDataType aclType  = cann::getACLType<T>();
      if(aclType == ACL_BF16)
        THROW_RUNTIME_ERROR("Unsupported ACL type: " + std::to_string(aclType));

      ctranslate2::cann::CannPreparation prepare;
      const aclFormat format = ACL_FORMAT_ND;
      cann_prepare_inputdesc(prepare, aclType, input.shape().size(), input.shape().data(), format);
      cann_prepare_outputdesc(prepare, aclType, output.shape().size(), output.shape().data(), format);

      cann_prepare_inputbuffer(prepare, const_cast<T*>(input.data<T>()), input.size()*sizeof(T));
      cann_prepare_outputbuffer(prepare, output.data<T>(), output.size()*sizeof(T));

      std::string op_type = log ? "LogSoftmaxV2" : "SoftmaxV2";
      ACL_CALL(aclopCompileAndExecute(op_type.c_str(),
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
    }

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      if (!lengths) {
        run_softmax<T>(input, output, _log);
      } else {
        // todo reduce number of operator calls for this case in the future
        dim_t batch_size = input.size() / input.dim(-1);

        std::vector<int32_t> lengths_vector = lengths->to_vector<int32_t>();
        int32_t current_length;

        // View 'input' and 'output' as 2D vectors with 'batch_size' number of rows.
        auto input_2D  = StorageView({batch_size, input.dim(-1)}, const_cast<T*>(input.data<T>()), D);
        auto output_2D = StorageView({batch_size, input.dim(-1)}, const_cast<T*>(output.data<T>()), D);

        // keeps track of the indices of the current slice of the input/output StorageView
        // {0, 0} corresponds to the 1st slice, {1, 0} corresponds to the 2nd slice etc...
        std::vector<dim_t> indices(2, 0);

        for (size_t i=0; i<batch_size; ++i) {
          indices[0] = i;
          current_length = lengths_vector[i];
          StorageView input_slice  = StorageView({current_length}, const_cast<T*>(input_2D.index<T>(indices)), D),
                      output_slice = StorageView({current_length}, const_cast<T*>(output_2D.index<T>(indices)), D);

          run_softmax<T>(input_slice, output_slice, _log);

          if (input_2D.dim(-1) == current_length) {
            continue;
          }

          // point at the end of the current slice
          indices[1] = current_length;

          // fill the end of the current slice with zeros
          StorageView({input_2D.dim(-1) - current_length}, const_cast<T*>(output_2D.index<T>(indices)), D).zero(false);

          // point back at the start of the current slice for the next iteration
          indices[1] = 0;
        }
      }

      // Synchronize stream only once in the end, not after operator call
      ACL_CALL(aclrtSynchronizeStream(cann::get_aclrt_stream()));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CANN, T>(const StorageView& input,         \
                                      const StorageView* lengths,       \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

