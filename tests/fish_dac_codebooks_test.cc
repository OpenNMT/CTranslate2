#include <limits>

#include "test_utils.h"
#include "layers/drama/fish_dac_codebooks.h"

namespace {

  class TinyCodebooks : public models::Model {
  public:
    TinyCodebooks() {
      add("semantic_quantizer/quantizers/0", {3, 4, -2, 5, 6, -7}, {1, 2, -3, 4}, {5, -6});
      add("quantizer/quantizers/0", {8, -9, 10, 11, -12, 13}, {2, -1, 3, 2}, {-7, 8});
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void replace(const std::string& name, StorageView value) {
      remove_variable("codec/quantizer/" + name);
      register_variable("codec/quantizer/" + name, std::move(value));
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyCodebooks>(*this);
    }

  private:
    void add(const std::string& name, std::vector<float> e, std::vector<float> p,
             std::vector<float> b) {
      register_variable("codec/quantizer/" + name + "/codebook/weight", StorageView({3, 2}, e));
      register_variable("codec/quantizer/" + name + "/out_proj/weight", StorageView({2, 2, 1}, p));
      register_variable("codec/quantizer/" + name + "/out_proj/bias", StorageView({2}, b));
    }
  };

}

TEST(FishDACCodebooksTest, RawVectorsBiasedProjectionsAndSumMatchIndependentScalarEquation) {
  TinyCodebooks model;
  layers::FishDACCodebooks books(model, "codec/quantizer", 2);
  StorageView ids({1, 2, 3}, std::vector<int32_t>{0, 1, 2, 2, 0, 1}), output;
  books(ids, output);
  // Raw code vectors deliberately have norms far from one. Including each
  // projection's bias gives these values; encoder-side L2 normalization does not.
  expect_storage_eq(output, StorageView({1, 2, 3}, std::vector<float>{-28, 31, -1, -1, 34, 8}));
}

TEST(FishDACCodebooksTest, OutOfRangeCodesPreserveOutput) {
  TinyCodebooks model;
  layers::FishDACCodebooks books(model, "codec/quantizer", 2);
  StorageView ids({1, 2, 2}, std::vector<int32_t>{0, 1, 1, 2});
  for (const int32_t bad : {-1, 3}) {
    ids.data<int32_t>()[3] = bad;
    StorageView output({2}, std::vector<float>{123, 456});
    EXPECT_THROW(books(ids, output), std::invalid_argument);
    expect_storage_eq(output, StorageView({2}, std::vector<float>{123, 456}));
  }
}

TEST(FishDACCodebooksTest, RejectsShapesTypesAndAliasing) {
  TinyCodebooks model;
  layers::FishDACCodebooks books(model, "codec/quantizer", 2);
  StorageView output;
  for (const auto& shape : {Shape{2, 2}, Shape{2, 2, 1}, Shape{1, 3, 1}, Shape{1, 2, 0}}) {
    StorageView ids(shape, DataType::INT32);
    EXPECT_THROW(books(ids, output), std::invalid_argument);
  }
  StorageView wrong_type({1, 2, 1}, 0.f);
  EXPECT_THROW(books(wrong_type, output), std::invalid_argument);
  StorageView ids({1, 2, 1}, std::vector<int32_t>{0, 1});
  StorageView alias({1}, reinterpret_cast<float*>(ids.data<int32_t>()));
  EXPECT_THROW(books(ids, alias), std::invalid_argument);
  StorageView retained({8}, 0.f);
  StorageView shared_ids({1, 2, 1}, reinterpret_cast<int32_t*>(retained.data<float>() + 4));
  retained.resize({0});
  EXPECT_THROW(books(shared_ids, retained), std::invalid_argument);
  const auto& weight =
      model.get_variable("codec/quantizer/semantic_quantizer/quantizers/0/codebook/weight");
  StorageView weight_alias({1}, const_cast<float*>(weight.data<float>()) + 1);
  EXPECT_THROW(books(ids, weight_alias), std::invalid_argument);
}

TEST(FishDACCodebooksTest, RejectsMalformedOrNonfiniteWeightsAndNonfiniteResult) {
  TinyCodebooks model;
  EXPECT_THROW(layers::FishDACCodebooks(model, "codec/quantizer", 0), std::invalid_argument);
  model.replace("semantic_quantizer/quantizers/0/out_proj/weight", StorageView({2, 3, 1}, 1.f));
  EXPECT_THROW(layers::FishDACCodebooks(model, "codec/quantizer", 2), std::invalid_argument);
  TinyCodebooks nonfinite;
  nonfinite.replace("semantic_quantizer/quantizers/0/codebook/weight",
                    StorageView({3, 2}, std::numeric_limits<float>::quiet_NaN()));
  EXPECT_THROW(layers::FishDACCodebooks(nonfinite, "codec/quantizer", 2), std::invalid_argument);
  TinyCodebooks overflow;
  overflow.replace("semantic_quantizer/quantizers/0/out_proj/weight",
                   StorageView({2, 2, 1}, std::numeric_limits<float>::max()));
  layers::FishDACCodebooks books(overflow, "codec/quantizer", 2);
  StorageView ids({1, 2, 1}, std::vector<int32_t>{0, 0}), output({1}, 12.f);
  EXPECT_THROW(books(ids, output), std::runtime_error);
  expect_storage_eq(output, StorageView({1}, 12.f));
}
