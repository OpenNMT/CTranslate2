#pragma once

#include <ctranslate2/replica_pool.h>
#include <ctranslate2/models/model_factory.h>
#include <ctranslate2/models/model.h>

#include <unordered_map>
#include <optional>
#include "utils.h"

namespace ctranslate2 {
  namespace python {

    inline std::shared_ptr<models::ModelReader>
    create_model_reader(const std::string& model, py::object files) {
      if (files.is_none())
        return std::make_shared<models::ModelFileReader>(model);

      if (!py::isinstance<py::dict>(files))
        throw pybind11::type_error("files argument must be a dictionary mapping file names "
                                   "to the file contents");

      auto reader = std::make_shared<models::ModelMemoryReader>(model);

      for (const auto& pair : files.cast<py::dict>()) {
        auto filename = pair.first;
        auto content = pair.second;

        auto read = py::getattr(content, "read", py::none());
        if (!read.is_none())
          content = read();
        else if (!py::isinstance<py::bytes>(content))
          throw pybind11::type_error("File content must be a file-like or bytes object");

        reader->register_file(filename.cast<std::string>(), content.cast<std::string>());
      }

      return reader;
    }

    template <typename T>
    class ReplicaPoolHelper {
    public:
      ReplicaPoolHelper(const std::string& model_path,
                        const std::string& device,
                        const std::variant<int, std::vector<int>>& device_index,
                        const StringOrMap& compute_type,
                        size_t inter_threads,
                        size_t intra_threads,
                        long max_queued_batches,
                        py::object files)
        : _model_loader(create_model_reader(model_path, files))
      {
        pybind11::gil_scoped_release nogil;

        _model_loader->device = str_to_device(device);
        _model_loader->device_indices = std::visit(DeviceIndexResolver(), device_index);
        _model_loader->compute_type = std::visit(ComputeTypeResolver(device), compute_type);
        _model_loader->num_replicas_per_device = inter_threads;

        _pool_config.num_threads_per_replica = intra_threads;
        _pool_config.max_queued_batches = max_queued_batches;

        _pool = std::make_unique<T>(_model_loader.value(), _pool_config);
      }

      ReplicaPoolHelper(const std::string& spec,
                        const size_t& spec_version,
                        const size_t& binary_version,
                        std::unordered_map<std::string, std::string>& aliases,
                        std::unordered_map<std::string, std::vector<std::string>>& vocabularies,
                        std::unordered_map<std::string, StorageView>& variables,
                        const std::string& config,
                        const std::string& device,
                        const std::variant<int, std::vector<int>>& device_index,
                        const StringOrMap& compute_type,
                        size_t ,//inter_threads
                        size_t intra_threads,
                        long max_queued_batches)
      {
        pybind11::gil_scoped_release nogil;

        // Load the variables.
        auto model_device = str_to_device(device);
        auto model_device_indices = std::visit(DeviceIndexResolver(), device_index)[0];
        auto model_compute_type = std::visit(ComputeTypeResolver(device), compute_type);

        auto model = models::Model::load(spec,
                                         spec_version,
                                         binary_version,
                                         aliases,
                                         vocabularies,
                                         variables,
                                         config,
                                         model_device,
                                         model_device_indices,
                                         model_compute_type);

        _pool_config.num_threads_per_replica = intra_threads;
        _pool_config.max_queued_batches = max_queued_batches;

        _pool = std::make_unique<T>(model, _pool_config);
      }

      ~ReplicaPoolHelper() {
        pybind11::gil_scoped_release nogil;
        _pool.reset();
      }

      std::string device() const {
        if (_model_loader.has_value())
          return device_to_str(_model_loader->device);
        if (_device)
          return _device.value();
        return "";
      }

      const std::vector<int>& device_index() const {
        if (_model_loader.has_value())
          return _model_loader->device_indices;
        if (!_device_index.has_value() || _device_index->empty())
          throw pybind11::type_error("No device index found");
        return _device_index.value();
      }

      std::string compute_type() const {
        return compute_type_to_str(model()->effective_compute_type());
      }

      size_t num_replicas() const {
        return _pool->num_replicas();
      }

      size_t num_queued_batches() const {
        return _pool->num_queued_batches();
      }

      size_t num_active_batches() const {
        return _pool->num_active_batches();
      }

    protected:
      std::unique_ptr<T> _pool;
      std::optional<models::ModelLoader> _model_loader;
      std::optional<std::string> _device;
      std::optional<std::vector<int>> _device_index;
      ReplicaPoolConfig _pool_config;

      const std::shared_ptr<const models::Model>& model() const {
        return _pool->get_first_replica().model();
      }
    };

  }
}
