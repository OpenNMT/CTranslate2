#include "ctranslate2/models/wavlm.h"

#include <algorithm>

#include "ctranslate2/decoding.h"

#include "dispatch.h"
#include "dtw.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif


namespace ctranslate2 {
  namespace models {

    const Vocabulary& WavLMModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t WavLMModel::current_spec_revision() const {
      return 3;
    }

    void WavLMModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "[UNK]";
      vocab_info.bos_token = "<s>";
      vocab_info.eos_token = "</s>";

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");
    }

    bool WavLMModel::is_quantizable(const std::string& variable_name) const {
      return Model::is_quantizable(variable_name);
    }

    bool WavLMModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> WavLMModel::clone() const {
      return std::make_unique<WavLMModel>(*this);
    }


    std::unique_ptr<WavLMReplica> WavLMReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const WavLMModel*>(&model))
        throw std::invalid_argument("The model is not a WavLM model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const WavLMModel>(model_ptr);
      return std::make_unique<WavLMReplica>(concrete_model);
    }

    WavLMReplica::WavLMReplica(const std::shared_ptr<const WavLMModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::WavLMEncoder>(*model, "encoder"))
    {
    }

    StorageView WavLMReplica::encode(StorageView features, const bool to_cpu) {
      PROFILE("WavLMReplica::encode");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();
      features.move_to(device, dtype);

      StorageView encoder_output(dtype, device);
      if (_encoder->_upgraded_model) {
        encoder_output = maybe_encode(std::move(features));
      }
      else {
        (*_encoder)(features, encoder_output);
      }

      if (to_cpu) {
        if (device != Device::CPU)
          encoder_output = encoder_output.to(Device::CPU);

        return encoder_output;
      }

      // Ensure all operations are finished before returning the output.
      synchronize_stream(device);

      return encoder_output;
    }

    StorageView WavLMReplica::maybe_encode(StorageView features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      features.move_to(device, dtype);

      if (_encoder->is_encoded(features))
        return features;

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);
      return encoder_output;
    }

    std::future<StorageView> WavLM::encode(const StorageView& features, const bool to_cpu) {
      return post<StorageView>(
        [features = features.sync_copy(), to_cpu](WavLMReplica& replica) mutable {
          return replica.encode(std::move(features), to_cpu);
        });
    }

  }
}
