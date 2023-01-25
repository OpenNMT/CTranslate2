#include "ctranslate2/models/model.h"

#include <spdlog/spdlog.h>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/models/model_factory.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/utils.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

    static const std::string binary_file = "model.bin";
    static const std::string config_file = "config.json";

    static inline void report_stream_error(const std::streampos position,
                                           const size_t read_size,
                                           const std::string& read_type) {
      throw std::runtime_error("File " + binary_file + " is incomplete: "
                               + "failed to read a " + read_type + " of size "
                               + std::to_string(read_size)
                               + " at position "
                               + std::to_string(position));
    }

    template <typename T>
    T consume(std::istream& in) {
      const std::streampos position = in.tellg();
      const size_t read_size = sizeof (T);
      T val;
      in.read(reinterpret_cast<char*>(&val), read_size);
      if (!in)
        report_stream_error(position, read_size, "value");
      return val;
    }

    template <typename T>
    T* consume(std::istream& in, size_t n, T* data = nullptr) {
      if (n == 0)
        return nullptr;
      const std::streampos position = in.tellg();
      const size_t read_size = n * sizeof (T);
      T* dst = data ? data : new T[n];
      in.read(reinterpret_cast<char*>(dst), read_size);
      if (!in) {
        if (dst != data)
          delete [] dst;
        report_stream_error(position, read_size, "buffer");
      }
      return dst;
    }

    template<>
    std::string consume(std::istream& in) {
      const auto str_length = consume<uint16_t>(in);
      const auto c_str = consume<char>(in, str_length);
      std::string str(c_str);
      delete [] c_str;
      return str;
    }

    static void move_variables_to_device(VariablesCollection& variables, const Device device) {
      for (auto& pair : variables) {
        StorageView& variable = *pair.second;
        if (variable.is_scalar() || variable.device() == device)
          continue;
        variable = variable.to(device);
      }
    }

    static void move_variables(VariablesCollection& variables,
                               const Device src_device, const int src_device_index,
                               const Device dst_device, const int dst_device_index) {
      if (variables.empty())
        return;
      if (src_device == dst_device && src_device_index == dst_device_index)
        return;

      // Move variables back to the CPU device.
      if (src_device != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(src_device, src_device_index);
        move_variables_to_device(variables, Device::CPU);
      }

      // Move variables to the destination device.
      if (dst_device != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(dst_device, dst_device_index);
        move_variables_to_device(variables, dst_device);
      }

      synchronize_device(src_device, src_device_index);  // Wait for asynchronous deallocations.
    }

    static StorageView copy_variable(const StorageView& variable,
                                     const Device device, const int device_index) {
      if (variable.is_scalar() || (variable.device() == Device::CPU && device == Device::CPU))
        return variable;

      StorageView copy;

      if (variable.device() != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(variable.device(), variable.device_index());
        copy = variable.to(Device::CPU);
      }

      if (device != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(device, device_index);
        if (copy)
          copy = copy.to(device);
        else
          copy = variable.to(device);
      }

      return copy;
    }


    std::unique_ptr<SequenceToSequenceReplica> Model::as_sequence_to_sequence() const {
      throw std::runtime_error("This model cannot be used as a sequence-to-sequence model");
    }

    std::unique_ptr<SequenceGeneratorReplica> Model::as_sequence_generator() const {
      throw std::runtime_error("This model cannot be used as a sequence generator");
    }

    Model::~Model() {
      if (!_variable_index.empty()) {
        _variable_index.clear();
        synchronize_device(_device, _device_index);  // Wait for asynchronous deallocations.
      }
    }

    size_t Model::current_spec_revision() const {
      return 1;
    }

    void Model::set_device(const Device device, const int index) {
      move_variables(_variable_index, _device, _device_index, device, index);
      _device = device;
      _device_index = index;
    }

    const StorageView* Model::get_variable_if_exists(const std::string& name) const {
      auto it = _variable_index.find(name);
      if (it == _variable_index.end())
        return nullptr;
      return it->second.get();
    }

    const StorageView& Model::get_variable(const std::string& name) const {
      const auto* var = get_variable_if_exists(name);
      if (var == nullptr)
        throw std::out_of_range("variable " + name + " not found");
      return *var;
    }

    std::unordered_map<std::string, StorageView> Model::get_variables() const {
      std::unordered_map<std::string, StorageView> variables;
      variables.reserve(_variable_index.size());
      for (const auto& pair : _variable_index)
        variables.emplace(pair.first, *pair.second);
      return variables;
    }

    bool Model::layer_exists(std::string prefix) const {
      if (!prefix.empty() && prefix.back() != '/')
        prefix += '/';
      for (const auto& pair : _variable_index) {
        const auto& name = pair.first;
        if (starts_with(name, prefix))
          return true;
      }
      return false;
    }

    bool Model::get_flag_with_default(const std::string& name, bool default_value) const {
      return get_attribute_with_default(name, static_cast<int8_t>(default_value));
    }

    void Model::register_variable(std::string name, std::shared_ptr<StorageView> variable) {
      _variable_index.emplace(std::move(name), std::move(variable));
    }

    void Model::register_variable(std::string name, StorageView variable) {
      register_variable(std::move(name), std::make_shared<StorageView>(std::move(variable)));
    }

    void Model::register_variable_alias(std::string alias, const std::string& variable_name) {
      auto it = _variable_index.find(variable_name);
      if (it == _variable_index.end())
        return;
      _variable_index.emplace(std::move(alias), it->second);
    }

    void Model::remove_variable(const std::string& name) {
      _variable_index.erase(name);
    }

    bool Model::is_quantizable(const std::string& variable_name) const {
      return ends_with(variable_name, "weight");
    }

    bool Model::is_linear_weight(const std::string&) const {
      return false;
    }

    bool Model::is_packable(const std::string& variable_name) const {
      return is_linear_weight(variable_name);
    }

    bool Model::is_convertible(const std::string& name, DataType dtype) const {
      return is_float_type(dtype) && name.find("_scale") == std::string::npos;
    }

    static void
    convert_weight(const std::string& name,
                   StorageView& variable,
                   StorageView& scale,
                   const DataType target_dtype,
                   const bool round_before_cast) {
      const bool is_int8 = variable.dtype() == DataType::INT8;
      const bool is_int16 = variable.dtype() == DataType::INT16;
      const bool is_float = variable.dtype() == DataType::FLOAT;
      const bool is_float16 = variable.dtype() == DataType::FLOAT16;

      if (!scale) {
        if (is_int16) {
          // Backward compatibility with int16 models without a saved scale.
          scale = StorageView(ops::Quantize::global_int16_scale);
        } else if (is_int8) {
          throw std::runtime_error("Missing quantization scale for int8 variable " + name);
        }
      }

      if (variable.dtype() == target_dtype)
        return;

      // Use the same quantization logic as in model_spec.py.
      const ops::Quantize quantize_op(/*int16_scale_type=*/ops::Quantize::ScaleType::PER_LAYER,
                                      /*shift_to_uint8=*/false,
                                      /*round_before_cast=*/round_before_cast);
      const ops::Dequantize dequantize_op{};
      StorageView target_variable(target_dtype);

      if (target_dtype == DataType::FLOAT || target_dtype == DataType::FLOAT16) {
        if (is_float16) {
          target_variable = variable.to_float();
        } else if (is_float) {
          target_variable = variable.to_float16();
        } else {
          // Dequantize int8 or int16 back to float32.
          StorageView dequantized;
          dequantize_op(variable, scale, dequantized);
          scale.clear();  // The scale is no longer needed.
          if (target_dtype == DataType::FLOAT16) {
            target_variable = dequantized.to_float16();
          } else {
            target_variable = std::move(dequantized);
          }
        }

      } else if (is_float || is_float16) {
        // Quantize float32 to int8 or int16.
        if (is_float16) {
          quantize_op(variable.to_float(), target_variable, scale);
        } else {
          quantize_op(variable, target_variable, scale);
        }

      } else {
        // Convert int8 -> float32 -> int16 or int16 -> float32 -> int8.
        StorageView tmp_variable;
        dequantize_op(variable, scale, tmp_variable);
        quantize_op(tmp_variable, target_variable, scale);
      }

      variable = std::move(target_variable);
    }

    static DataType get_dtype_from_item_size(uint8_t item_size) {
      // This is the old (and flawed) logic of resolving the dtype of saved variables.
      switch (item_size) {
      case 4:
        return DataType::FLOAT;
      case 2:
        return DataType::INT16;
      case 1:
        return DataType::INT8;
      default:
        throw std::runtime_error("unknown data type of width " + std::to_string(item_size));
      }
    }

    static void check_version(const size_t saved_version,
                              const size_t current_version,
                              const std::string& version_type) {
      if (saved_version > current_version)
        throw std::runtime_error("Unsupported model " + version_type
                                 + ". This executable supports models with " + version_type + " v"
                                 + std::to_string(current_version)
                                 + " or below, but the model has " + version_type + " v"
                                 + std::to_string(saved_version)
                                 + ". This usually means that the model was generated by a later "
                                 + "version of CTranslate2. "
                                 + "(Forward compatibility is not guaranteed.)");
    }

    // See the model serialization in python/ctranslate2/specs/model_spec.py.

    struct SerializedVariable {
      std::string name;
      Shape shape;
      DataType dtype;
      size_t offset;
      size_t num_bytes;

      SerializedVariable(std::istream& in, size_t binary_version) {
        name = consume<std::string>(in);

        const size_t rank = consume<uint8_t>(in);
        const auto* dimensions = consume<uint32_t>(in, rank);
        shape.assign(dimensions, dimensions + rank);
        delete [] dimensions;

        if (binary_version >= 4) {
          const auto type_id = consume<uint8_t>(in);
          dtype = static_cast<DataType>(type_id);
          num_bytes = consume<uint32_t>(in);
        } else {
          const auto item_size = consume<uint8_t>(in);
          dtype = get_dtype_from_item_size(item_size);
          num_bytes = consume<uint32_t>(in) * item_size;
        }

        offset = in.tellg();
        in.seekg(offset + num_bytes);
      }

      // Actually load the variable in memory.
      StorageView load(std::istream& in) const {
        StorageView variable(shape, dtype);

        const auto previous_offset = in.tellg();
        in.seekg(offset);
        consume<char>(in, num_bytes, static_cast<char*>(variable.buffer()));
        in.seekg(previous_offset);

        return variable;
      }
    };

    struct SerializedModel {
      size_t spec_revision;
      std::string spec_name;
      std::vector<SerializedVariable> variables;

      SerializedModel(std::istream& in, size_t binary_version) {
        if (binary_version >= 2) {
          spec_name = consume<std::string>(in);
          spec_revision = consume<uint32_t>(in);
        } else {
          spec_revision = 1;
        }

        const auto num_variables = consume<uint32_t>(in);
        variables.reserve(num_variables);

        for (uint32_t i = 0; i < num_variables; ++i) {
          variables.emplace_back(in, binary_version);
        }

        if (binary_version >= 3) {
          const auto num_aliases = consume<uint32_t>(in);
          variables.reserve(num_variables + num_aliases);

          for (uint32_t i = 0; i < num_aliases; ++i) {
            auto alias_name = consume<std::string>(in);
            auto variable_name = consume<std::string>(in);

            auto* variable = get_variable(variable_name);
            if (variable) {
              variables.emplace_back(*variable);
              variables.back().name = std::move(alias_name);
            }
          }
        }
      }

      SerializedVariable* get_variable(const std::string& name) {
        for (SerializedVariable& variable : variables) {
          if (variable.name == name)
            return &variable;
        }

        return nullptr;
      }
    };

    std::shared_ptr<const Model> Model::load(const std::string& path,
                                             Device device,
                                             int device_index,
                                             ComputeType compute_type) {
      ModelFileReader model_reader(path);
      return load(model_reader, device, device_index, compute_type);
    }

    std::shared_ptr<const Model> Model::load(ModelReader& model_reader,
                                             Device device,
                                             int device_index,
                                             ComputeType compute_type) {
      {
        // Log the system configuration the first time a model is loaded.
        static std::once_flag log_once;
        std::call_once(log_once, log_system_config);
      }

      const ScopedDeviceSetter scoped_device_setter(device, device_index);

      std::unique_ptr<std::istream> model_file_ptr = model_reader.get_required_file(binary_file,
                                                                                    /*binary=*/true);
      std::istream& model_file = *model_file_ptr;

      const size_t binary_version = consume<uint32_t>(model_file);
      check_version(binary_version, current_binary_version, "binary version");

      SerializedModel serialized_model(model_file, binary_version);

      auto model = create_model(serialized_model.spec_name);
      model->_binary_version = binary_version;
      model->_spec_revision = serialized_model.spec_revision;
      model->_device = device;
      model->_device_index = device_index;

      check_version(serialized_model.spec_revision, model->current_spec_revision(), "revision");

      {
        std::unique_ptr<std::istream> config_file_ptr = model_reader.get_file(config_file);
        if (config_file_ptr)
          model->config = nlohmann::json::parse(*config_file_ptr);
      }

      // Multiple variables can point to the same buffer in the model.
      // We will make sure that these variables remain shared.
      std::unordered_map<size_t, std::vector<const SerializedVariable*>> variables_at_offset;
      variables_at_offset.reserve(serialized_model.variables.size());

      DataType weight_type = DataType::FLOAT;
      DataType float_type = DataType::FLOAT;

      for (auto& variable : serialized_model.variables) {
        // Models may rename some variables for backward compatibility.
        model->update_variable_name(variable.name);

        // Quantization scales will be processed alongside their corresponding weight.
        if (ends_with(variable.name, "_scale"))
          continue;

        // Scalars can be registered immediately.
        if (variable.shape.empty()) {
          model->register_variable(variable.name, variable.load(model_file));
          continue;
        }

        // Gather some information about the variables type to resolve the default compute type.
        if (model->is_quantizable(variable.name))
          weight_type = variable.dtype;
        else if (model->is_convertible(variable.name, variable.dtype))
          float_type = variable.dtype;

        variables_at_offset[variable.offset].push_back(&variable);
      }

      ComputeType default_compute_type = data_type_to_compute_type(weight_type, float_type);
      ComputeType effective_compute_type = resolve_compute_type(compute_type,
                                                                default_compute_type,
                                                                device,
                                                                device_index);

      // Update the target dtypes based on the effective compute type.
      weight_type = compute_type_to_data_type(effective_compute_type);
      float_type = get_default_float_type(effective_compute_type);

      model->_compute_type = compute_type;
      model->_effective_compute_type = effective_compute_type;
      model->_preferred_size_multiple = get_preferred_size_multiple(effective_compute_type,
                                                                    device,
                                                                    device_index);

      const bool quantization_round_mode = model->round_before_cast_in_quantization();

      for (const auto& pair : variables_at_offset) {
        VariablesCollection variables;

        for (const auto* serialized_variable : pair.second) {
          const auto& name = serialized_variable->name;

          auto& weight = variables[""];
          if (!weight)
            weight = std::make_shared<StorageView>(serialized_variable->load(model_file));

          if (model->is_quantizable(name)) {
            auto& scale = variables["_scale"];

            if (!scale) {
              const auto* serialized_scale = serialized_model.get_variable(name + "_scale");
              scale = (serialized_scale
                       ? std::make_shared<StorageView>(serialized_scale->load(model_file))
                       : std::make_shared<StorageView>());
            }

            convert_weight(name, *weight, *scale, weight_type, quantization_round_mode);

            if (!scale->empty()) {
              if (scale->device() != device && !scale->is_scalar())
                *scale = scale->to(device);
              model->register_variable(name + "_scale", scale);
            }

            if (model->is_linear_weight(name)) {
              layers::Dense::register_weight(name,
                                             weight,
                                             *model,
                                             variables,
                                             device,
                                             effective_compute_type,
                                             model->is_packable(name));
            } else {
              if (weight->device() != device)
                *weight = weight->to(device);
              model->register_variable(name, weight);
            }

          } else {
            if (model->is_convertible(name, weight->dtype()) && weight->dtype() != float_type)
              *weight = weight->to(float_type);
            if (weight->device() != device)
              *weight = weight->to(device);
            model->register_variable(name, weight);
          }
        }
      }

      // Run additional model initialization.
      model->initialize(model_reader);
      return model;
    }

    std::shared_ptr<const Model> Model::copy_to(Device device, int device_index) const {
      auto model = clone();

      // We should consider and keep aliased variables in the new model.
      std::unordered_map<const StorageView*, std::shared_ptr<StorageView>> seen_variables;
      seen_variables.reserve(_variable_index.size());

      for (const auto& pair : _variable_index) {
        const auto& name = pair.first;
        const auto& value = pair.second;

        auto it = seen_variables.find(value.get());

        if (it != seen_variables.end()) {
          model->_variable_index[name] = it->second;
        } else {
          auto copy = std::make_shared<StorageView>(copy_variable(*value, device, device_index));
          model->_variable_index[name] = copy;
          seen_variables.emplace(value.get(), copy);
        }
      }

      model->_device = device;
      model->_device_index = device_index;
      return model;
    }

    bool contains_model(const std::string& path) {
      return bool(ModelFileReader(path).get_file(binary_file));
    }

    ModelLoader::ModelLoader(const std::string& model_path)
      : model_reader(std::make_shared<ModelFileReader>(model_path))
    {
    }

    ModelLoader::ModelLoader(const std::shared_ptr<ModelReader>& model_reader_)
      : model_reader(model_reader_)
    {
    }

    std::vector<std::shared_ptr<const Model>>
    ModelLoader::load() const {
      if (device_indices.empty())
        throw std::invalid_argument("At least one device index should be set");
#ifdef CT2_WITH_CUDA
      if (device == Device::CUDA && !cuda::have_same_compute_capability(device_indices))
        throw std::invalid_argument("Cannot use multiple GPUs with different Compute Capabilities "
                                    "for the same model");
#endif

      std::vector<std::shared_ptr<const Model>> models;
      models.reserve(device_indices.size() * num_replicas_per_device);

      for (const size_t device_index : device_indices) {
        std::shared_ptr<const Model> model;

        if (models.empty())
          model = Model::load(*model_reader, device, device_index, compute_type);
        else
          model = models.back()->copy_to(device, device_index);

        spdlog::info("Loaded model {} on device {}:{}",
                     model_reader->get_model_id(),
                     device_to_str(device),
                     device_index);
        spdlog::info(" - Binary version: {}", model->binary_version());
        spdlog::info(" - Model specification revision: {}", model->spec_revision());
        spdlog::info(" - Selected compute type: {}",
                     compute_type_to_str(model->effective_compute_type()));

        for (size_t i = 0; i < num_replicas_per_device; ++i)
          models.emplace_back(model);
      }

      return models;
    }

  }
}
