#include "test_utils.h"
#include "layers/drama/drama_projection.h"

#include <atomic>
#include <future>
#include <iostream>
#include <unordered_set>

namespace {
  class SharedWeightsModel : public models::Model {
  public:
    explicit SharedWeightsModel(dim_t width = 1024) {
      for (int i = 0; i < 4; ++i)
        register_variable("proj" + std::to_string(i) + "/weight",
                          StorageView({width, width}, 1.f / 1024));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }
    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<SharedWeightsModel>(*this);
    }
    bool is_packable(const std::string&) const override { return false; }
  };
}

TEST(DramaDeviceWeightsTest, ConcurrentPreparationAndLastOwnerEviction) {
  layers::DramaDeviceWeightRegistry registry;
  std::atomic<int> preparations{0};
  auto prepare = [&] {
    ++preparations;
    auto weights = std::make_shared<layers::DramaDeviceWeights>();
    weights->weight = StorageView({16}, 1.f);
    return weights;
  };
  std::vector<std::future<std::shared_ptr<const layers::DramaDeviceWeights>>> jobs;
  for (int i = 0; i < 4; ++i)
    jobs.push_back(std::async(std::launch::async, [&] { return registry.get("weight", prepare); }));
  auto first = jobs.front().get();
  for (size_t i = 1; i < jobs.size(); ++i)
    EXPECT_EQ(jobs[i].get().get(), first.get());
  EXPECT_EQ(preparations, 1);
  auto second = registry.get("weight", prepare);
  first.reset();
  EXPECT_EQ(second->weight.data<float>()[0], 1.f);
  std::weak_ptr<const layers::DramaDeviceWeights> weak = second;
  second.reset();
  EXPECT_TRUE(weak.expired());
  first = registry.get("weight", prepare);
  EXPECT_EQ(preparations, 2);
}

TEST(DramaDeviceWeightsTest, GpuReplicaMemoryAndConcurrentExecution) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  const ScopedDeviceSetter setter(Device::CUDA, 0);
  SharedWeightsModel model;
  for (const auto type : {ComputeType::FLOAT32, ComputeType::FLOAT16, ComputeType::INT8_FLOAT32}) {
    try {
      resolve_compute_type(type, ComputeType::FLOAT32, Device::CUDA, 0);
    } catch (const std::invalid_argument&) {
      continue;
    }
    for (const size_t replicas : {1, 2, 4}) {
      size_t baseline = 0;
      for (const bool shared : {false, true}) {
        layers::DramaExecutionOptions options;
        options.device = Device::CUDA;
        options.compute_type = type;
        if (shared)
          options.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
        std::vector<std::vector<std::unique_ptr<layers::DramaProjection>>> projections(replicas);
        for (auto& replica : projections)
          for (int i = 0; i < 4; ++i)
            replica.emplace_back(std::make_unique<layers::DramaProjection>(
                model, "proj" + std::to_string(i), options));
        std::vector<std::future<void>> jobs;
        for (size_t r = 0; r < replicas; ++r) {
          jobs.push_back(std::async(std::launch::async, [&, r] {
            const ScopedDeviceSetter worker_setter(Device::CUDA, 0);
            auto input = StorageView({1, 1024}, .125f).to(Device::CUDA);
            for (auto& projection : projections[r]) {
              StorageView output(DataType::FLOAT32, Device::CUDA);
              (*projection)(input, output);
              auto host = output.to(Device::CPU);
              synchronize_stream(Device::CUDA);
              for (dim_t j = 0; j < host.size(); ++j)
                EXPECT_NEAR(host.data<float>()[j], .125f, .001f);
            }
          }));
        }
        for (auto& job : jobs)
          job.get();
        // Count actual allocated tensor capacities once per distinct GPU buffer.
        // Allocator pools, contexts, and temporary activations are excluded.
        size_t bytes = 0;
        std::unordered_set<const void*> buffers;
        for (auto& replica : projections)
          for (auto& projection : replica)
            if (buffers.insert(projection->device_weights().weight.buffer()).second)
              bytes += projection->device_weight_bytes();
        if (!shared) {
          baseline = bytes;
        } else {
          EXPECT_EQ(bytes * replicas, baseline);
          std::cout << "GPU_WEIGHT_MEMORY type=" << compute_type_to_str(type)
                    << " replicas=" << replicas << " baseline_bytes=" << baseline
                    << " shared_bytes=" << bytes << " saved_bytes=" << baseline - bytes << '\n';
          if (replicas > 1) {
            const auto original = projections[1][0]->device_weights().weight.buffer();
            projections[0][0]->release_device_weights();
            EXPECT_EQ(projections[1][0]->device_weights().weight.buffer(), original);
            EXPECT_EQ(projections[0][0]->device_weights().weight.buffer(), original);
          }
        }
      }
    }
  }
}

TEST(DramaDeviceWeightsTest, SelectedGpuWeightsShareOnlyMatchingSelections) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  const ScopedDeviceSetter setter(Device::CUDA, 0);
  SharedWeightsModel model(16);
  layers::DramaExecutionOptions options;
  options.device = Device::CUDA;
  options.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
  layers::DramaProjection a(model, "proj0", options), b(model, "proj0", options);
  StorageView indices({2}, std::vector<int32_t>{1, 3});
  a.select_weights(&indices);
  b.select_weights(&indices);
  const auto pointer = a.device_weights().weight.buffer();
  EXPECT_EQ(b.device_weights().weight.buffer(), pointer);
  StorageView other({2}, std::vector<int32_t>{2, 4});
  b.select_weights(&other);
  EXPECT_NE(b.device_weights().weight.buffer(), pointer);
  a.release_device_weights();
  b.release_device_weights();
  // Equal rows with different extra biases must not share prepared parameters.
  StorageView bias({2}, .1f);
  a.select_weights(&indices, &bias);
  b.select_weights(&indices, &bias);
  EXPECT_NE(a.device_weights().weight.buffer(), b.device_weights().weight.buffer());
}

TEST(DramaDeviceWeightsTest, ImmutableParametersShareUploadsAcrossReplicas) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  const ScopedDeviceSetter setter(Device::CUDA, 0);
  layers::DramaExecutionOptions options;
  options.device = Device::CUDA;
  options.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
  StorageView parameter({1024}, .25f);
  std::vector<std::shared_ptr<const layers::DramaDeviceWeights>> first, second;
  auto a = layers::upload_drama_weight(parameter, options, first);
  auto b = layers::upload_drama_weight(parameter, options, second);
  EXPECT_EQ(a.buffer(), b.buffer());
  EXPECT_FALSE(a.owns_data());
  first.clear();
  auto host = b.to(Device::CPU);
  synchronize_stream(Device::CUDA);
  expect_storage_eq(host, parameter);
}
