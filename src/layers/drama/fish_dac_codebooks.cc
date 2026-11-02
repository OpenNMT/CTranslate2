#include "layers/drama/fish_dac_codebooks.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {

    FishDACCodebooks::FishDACCodebooks(const models::Model& model, const std::string& scope,
                                       dim_t num_books)
      : _channels(0) {
      if (model.device() != Device::CPU || num_books <= 0)
        throw std::invalid_argument("FishDAC codebooks require CPU and positive book count");
      auto parameter = [&](const std::string& name) {
        const auto& value = model.get_variable(name);
        if (value.device() != Device::CPU || value.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("FishDAC codebook weights require CPU float32");
        for (dim_t i = 0; i < value.size(); ++i)
          if (!std::isfinite(value.data<float>()[i]))
            throw std::invalid_argument("FishDAC codebook weights must be finite");
        return &value;
      };
      for (dim_t i = 0; i < num_books; ++i) {
        const auto prefix = scope
                            + (i == 0 ? "/semantic_quantizer/quantizers/0"
                                      : "/quantizer/quantizers/" + std::to_string(i - 1));
        Book book{parameter(prefix + "/codebook/weight"), parameter(prefix + "/out_proj/weight"),
                  parameter(prefix + "/out_proj/bias")};
        const auto& e = *book.embedding;
        const auto& p = *book.projection;
        if (e.rank() != 2 || e.dim(0) <= 0 || e.dim(1) <= 0 || p.rank() != 3 || p.dim(0) <= 0
            || p.dim(1) != e.dim(1) || p.dim(2) != 1 || book.bias->shape() != Shape{p.dim(0)}
            || (i && p.dim(0) != _channels))
          throw std::invalid_argument("FishDAC codebook/projection shape mismatch");
        _channels = p.dim(0);
        _books.push_back(book);
      }
    }

    void FishDACCodebooks::validate_codes(const StorageView& codes,
                                          const StorageView& output) const {
      if (codes.device() != Device::CPU || codes.dtype() != DataType::INT32
          || output.device() != Device::CPU || output.dtype() != DataType::FLOAT32
          || codes.rank() != 3 || codes.dim(0) != 1
          || codes.dim(1) != static_cast<dim_t>(_books.size()) || codes.dim(2) <= 0)
        throw std::invalid_argument("FishDAC codes require CPU int32 [1,books,time]");
      // Compute into fresh storage, so output cannot overwrite codes or weights
      // during computation. Reject overlapping caller storage rather than losing
      // ownership or silently changing a model parameter on assignment.
      const auto out = reinterpret_cast<uintptr_t>(output.buffer());
      const auto size = static_cast<uintptr_t>(
          std::max(output.reserved_memory(), output.size() * output.item_size()));
      auto overlaps = [&](const StorageView& value) {
        const auto p = reinterpret_cast<uintptr_t>(value.buffer());
        const auto n = static_cast<uintptr_t>(value.size() * value.item_size());
        return out && p && size && n && (out <= p ? p - out < size : out - p < n);
      };
      if (&codes == &output || overlaps(codes))
        throw std::invalid_argument("FishDAC code output aliases input");
      const dim_t frames = codes.dim(2);
      for (size_t b = 0; b < _books.size(); ++b) {
        const auto& book = _books[b];
        if (overlaps(*book.embedding) || overlaps(*book.projection) || overlaps(*book.bias))
          throw std::invalid_argument("FishDAC code output aliases weights");
        for (dim_t t = 0; t < frames; ++t) {
          const int32_t code = codes.data<int32_t>()[b * frames + t];
          if (code < 0 || code >= book.embedding->dim(0))
            throw std::invalid_argument("FishDAC code outside codebook vocabulary");
        }
      }
    }

    void FishDACCodebooks::operator()(const StorageView& codes, StorageView& output) const {
      validate_codes(codes, output);
      const dim_t frames = codes.dim(2);
      StorageView result({1, _channels, frames}, 0.f);
      for (size_t b = 0; b < _books.size(); ++b) {
        const auto& book = _books[b];
        const dim_t width = book.embedding->dim(1);
        for (dim_t t = 0; t < frames; ++t) {
          const auto* embedding =
              book.embedding->data<float>() + codes.data<int32_t>()[b * frames + t] * width;
          for (dim_t c = 0; c < _channels; ++c) {
            float value = book.bias->data<float>()[c];
            const auto* weight = book.projection->data<float>() + c * width;
            for (dim_t k = 0; k < width; ++k)
              value += weight[k] * embedding[k];
            result.data<float>()[c * frames + t] += value;
          }
        }
      }
      for (dim_t i = 0; i < result.size(); ++i)
        if (!std::isfinite(result.data<float>()[i]))
          throw std::runtime_error("FishDAC reconstruction produced nonfinite values");
      output = std::move(result);
    }

  }
}
