#include "ctranslate2/decoding_utils.h"

#include <set>

#include "ctranslate2/ops/ops.h"
#include "dispatch.h"

namespace ctranslate2 {

  DisableTokens::DisableTokens(StorageView& logits, const float disable_value)
    : _logits(logits)
    , _logits_data(logits.device() == Device::CPU ? logits.data<float>() : nullptr)
    , _disable_value(disable_value)
    , _batch_size(logits.dim(0))
    , _vocabulary_size(logits.dim(1))
  {
  }

  void DisableTokens::apply() {
    const dim_t num_indices = _flat_indices.size();
    if (num_indices == 0)
      return;

    const Device device = _logits.device();
    const DataType dtype = _logits.dtype();
    const StorageView flat_indices({num_indices}, _flat_indices, device);

    DEVICE_AND_TYPE_DISPATCH(device, dtype,
                             primitives<D>::indexed_fill(_logits.data<T>(),
                                                         static_cast<T>(_disable_value),
                                                         flat_indices.data<int32_t>(),
                                                         num_indices));

    _flat_indices.clear();
  }

  BiasTokens::BiasTokens(StorageView& logits)
    : _logits(logits)
    , _logits_data(logits.device() == Device::CPU ? logits.data<float>() : nullptr)
    , _batch_size(logits.dim(0))
    , _vocabulary_size(logits.dim(1))
  {
  }

  void BiasTokens::apply() {
    const dim_t num_indices = _flat_indices.size();
    if (num_indices == 0)
      return;

    const Device device = _logits.device();
    const DataType dtype = _logits.dtype();

    std::vector<int32_t> indices(num_indices);
    std::vector<float> values(num_indices);

    std::transform(_flat_indices.begin(), _flat_indices.end(), indices.begin(),
                   [](const auto& pair) { return pair.first; });
    std::transform(_flat_indices.begin(), _flat_indices.end(), values.begin(),
                   [](const auto& pair) { return pair.second; });

    const StorageView flat_indices({num_indices}, indices, device);
    const StorageView flat_values({num_indices}, values, device);

    // Apply the disable values on GPU
    DEVICE_AND_TYPE_DISPATCH(device, dtype,
                             primitives<Device::CUDA>::indexed_pointwise_multiply(_logits.data<T>(),
                                                                                  flat_values.data<T>(),
                                                                                  flat_indices.data<int32_t>(),
                                                                                  num_indices));
    _flat_indices.clear();
    }

  RepetitionPenalty::RepetitionPenalty(const float penalty)
    : _penalty(penalty)
  {
  }

  void RepetitionPenalty::apply(dim_t,
                                StorageView& logits,
                                DisableTokens&,
                                BiasTokens&,
                                const StorageView& sequences,
                                const std::vector<dim_t>&,
                                const std::vector<std::vector<size_t>>*) {
    if (!sequences)
      return;

    const Device device = logits.device();
    const DataType dtype = logits.dtype();

    StorageView previous_ids = sequences.to(device);
    StorageView previous_scores(device, dtype);
    ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(logits, previous_ids, previous_scores);

    DEVICE_AND_TYPE_DISPATCH(device, dtype,
                             primitives<D>::penalize_previous_tokens(logits.data<T>(),
                                                                     previous_scores.data<T>(),
                                                                     previous_ids.data<int32_t>(),
                                                                     static_cast<T>(_penalty),
                                                                     logits.dim(0),
                                                                     previous_ids.dim(-1),
                                                                     logits.dim(-1)));
  }


  NoRepeatNgram::NoRepeatNgram(const size_t ngram_size)
    : _ngram_size(ngram_size)
  {
  }

  void NoRepeatNgram::apply(dim_t,
                            StorageView&,
                            DisableTokens& disable_tokens,
                            BiasTokens&,
                            const StorageView& sequences,
                            const std::vector<dim_t>&,
                            const std::vector<std::vector<size_t>>*) {
    if (!sequences || sequences.dim(-1) < _ngram_size)
      return;

    const dim_t batch_size = sequences.dim(0);
    const dim_t length = sequences.dim(1);

    for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      const auto* begin = sequences.index<int32_t>({batch_id, 0});
      const auto* end = begin + length;
      const auto* current_ngram_begin = end - _ngram_size + 1;

      std::set<size_t> ngram_final_tokens;

      while (true) {
        begin = std::search(begin, end, current_ngram_begin, end);
        if (begin + _ngram_size > end)
          break;
        ngram_final_tokens.emplace(begin[_ngram_size - 1]);
        begin += 1;
      }

      for (const auto token_id : ngram_final_tokens)
        disable_tokens.add(batch_id, token_id);
    }
  }


  SuppressSequences::SuppressSequences(std::vector<std::vector<size_t>> sequences) {
    for (auto& sequence : sequences) {
      if (sequence.empty())
        continue;
      if (sequence.size() == 1)  // Single tokens are always suppressed.
        _ids.emplace_back(sequence[0]);
      else
        _sequences.emplace_back(std::move(sequence));
    }
  }

  void SuppressSequences::apply(dim_t,
                                StorageView&,
                                DisableTokens& disable_tokens,
                                BiasTokens&,
                                const StorageView& sequences,
                                const std::vector<dim_t>&,
                                const std::vector<std::vector<size_t>>*) {
    for (const auto token_id : _ids)
      disable_tokens.add(token_id);

    if (!sequences)
      return;

    const dim_t batch_size = sequences.dim(0);
    const dim_t length = sequences.dim(1);

    for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      const auto* begin = sequences.index<int32_t>({batch_id, 0});
      const auto* end = begin + length;

      for (const auto& banned_sequence : _sequences) {
        const dim_t compare_length = banned_sequence.size() - 1;

        if (length < compare_length)
          continue;

        const bool disable_last = std::equal(end - compare_length,
                                             end,
                                             banned_sequence.begin(),
                                             banned_sequence.begin() + compare_length);

        if (disable_last)
          disable_tokens.add(batch_id, banned_sequence.back());
      }
    }
  }

  BiasSequences::BiasSequences(std::vector<std::pair<std::vector<size_t>, float>> sequences) {
    for (auto& sequence : sequences) {
      if (sequence.first.empty())
        continue;
      if (sequence.first.size() == 1)  // Single tokens are always suppressed.
        _ids.emplace_back(std::make_pair(sequence.first[0], sequence.second));
      else
        _sequences.emplace_back(std::move(sequence));
    }
  }

  void BiasSequences::apply(dim_t,
                                StorageView& logits,
                                DisableTokens&,
                                BiasTokens& bias_tokens,
                                const StorageView& sequences,
                                const std::vector<dim_t>&,
                                const std::vector<std::vector<size_t>>*) {
    for (const auto token_id : _ids)
      bias_tokens.add(token_id.first, token_id.second);

    if (!sequences)
      return;

    const dim_t batch_size = sequences.dim(0);
    const dim_t length = sequences.dim(1);

    for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      const auto* begin = sequences.index<int32_t>({batch_id, 0});
      const auto* end = begin + length;

      for (const auto& biased_sequence : _sequences) {
        const dim_t compare_length = biased_sequence.first.size() - 1;

        if (length < compare_length)
          continue;

        const bool bias_last = std::equal(end - compare_length,
                                             end,
                                             biased_sequence.first.begin(),
                                             biased_sequence.first.begin() + compare_length);

        if (bias_last)
          bias_tokens.add(batch_id, biased_sequence.first.back(), biased_sequence.second);
        }
      }
    }


  SuppressTokens::SuppressTokens(std::vector<size_t> ids)
    : _ids(std::move(ids))
  {
  }

  void SuppressTokens::apply(dim_t,
                             StorageView&,
                             DisableTokens& disable_tokens,
                             BiasTokens&,
                             const StorageView&,
                             const std::vector<dim_t>&,
                             const std::vector<std::vector<size_t>>*) {
    for (const auto token_id : _ids)
      disable_tokens.add(token_id);
  }


  SuppressTokensBegin::SuppressTokensBegin(std::vector<size_t> ids)
    : _ids(std::move(ids))
  {
  }

  void SuppressTokensBegin::apply(dim_t step,
                                  StorageView& logits,
                                  DisableTokens& disable_tokens,
                                  BiasTokens&,
                                  const StorageView&,
                                  const std::vector<dim_t>& batch_offset,
                                  const std::vector<std::vector<size_t>>* prefix) {
    const dim_t batch_size = logits.dim(0);

    for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      const dim_t sample_begin = get_sample_begin(batch_size, batch_id, batch_offset, prefix);

      if (step != sample_begin)
        continue;

      for (const auto token_id : _ids)
        disable_tokens.add(batch_id, token_id);
    }
  }

}
