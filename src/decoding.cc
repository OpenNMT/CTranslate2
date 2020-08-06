#include "ctranslate2/decoding.h"

#include <cmath>
#include <map>

#include "ctranslate2/ops/ops.h"
#include "device_dispatch.h"
#include "type_dispatch.h"

namespace ctranslate2 {

  static const ops::Gather gather;

  static void split_batch_beam(StorageView& input, dim_t beam_size) {
    Shape shape = input.shape();
    shape.insert(shape.begin() + 1, beam_size);
    shape[0] /= beam_size;
    input.reshape(shape);
  }

  static void merge_batch_beam(StorageView& input) {
    Shape shape = input.shape();
    shape[0] *= shape[1];
    shape.erase(shape.begin() + 1);
    input.reshape(shape);
  }

  static void gather_batch(StorageView& data, const StorageView& indices, dim_t beam_size) {
    split_batch_beam(data, beam_size);
    gather(data, indices);
    merge_batch_beam(data);
  }

  static void tile(StorageView& input, const StorageView& repeats) {
    static const ops::Tile tile_op{};
    tile_op(input, repeats);
  }

  static void expand_to_beam_size(StorageView& input, dim_t beam_size) {
    Shape original_shape(input.shape());
    Shape tile_shape(input.shape());
    tile_shape.insert(std::next(tile_shape.begin()), 1);
    input.reshape(tile_shape);
    StorageView repeats({input.rank()}, static_cast<int32_t>(1));
    repeats.at<int32_t>(1) = beam_size;
    tile(input, repeats);
    original_shape[0] *= beam_size;
    input.reshape(original_shape);
  }

  static void expand_to_beam_size(layers::DecoderState& state, dim_t beam_size) {
    for (auto& pair : state) {
      if (!pair.second.empty())
        expand_to_beam_size(pair.second, beam_size);
    }
  }

  static void penalize_token(StorageView& log_probs, const size_t id) {
    DEVICE_DISPATCH(log_probs.device(),
                    TYPE_DISPATCH(log_probs.dtype(),
                                  primitives<D>::strided_fill(log_probs.data<T>() + id,
                                                              static_cast<T>(-1e10),
                                                              log_probs.dim(-1),
                                                              log_probs.dim(0))));
  }

  static void update_sample_with_prefix(const dim_t step,
                                        StorageView& sampled_ids,
                                        StorageView& sampled_scores,
                                        const std::vector<std::vector<size_t>>& prefix_ids,
                                        const std::vector<dim_t>& batch_offset) {
    const dim_t batch_size = sampled_scores.dim(0);
    const dim_t beam_size = sampled_scores.dim(1);
    for (dim_t i = 0; i < batch_size; ++i) {
      const dim_t batch_id = batch_offset[i];
      const auto& prefix = prefix_ids[batch_id];
      const dim_t prefix_length = prefix.size();
      if (step >= prefix_length)
        continue;
      for (dim_t k = 0; k < beam_size; ++k) {
        sampled_ids.at<int32_t>({i, k}) = prefix[step];
        // Set the highest log score for the first beam and penalize the others.
        TYPE_DISPATCH(sampled_scores.dtype(),
                      sampled_scores.at<T>({i, k}) = (k == 0 ? 0 : T(-1e10)));
      }
    }
  }

  template <typename T>
  static void initialize_cum_log_probs(StorageView& cum_log_probs,
                                       const dim_t batch_size,
                                       const dim_t beam_size) {
    const dim_t size = batch_size * beam_size;
    cum_log_probs.resize({size});
    auto* data = cum_log_probs.data<T>();
    for (dim_t i = 0; i < size; ++i) {
      data[i] = (i % beam_size == 0 ? T(0) : std::numeric_limits<T>::lowest());
    }
  }


  BeamSearch::BeamSearch(const dim_t beam_size, const float length_penalty, const float coverage_penalty)
    : _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty) {
  }

  void
  BeamSearch::search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     const Sampler& sampler,
                     const std::vector<size_t>& start_ids,
                     const size_t end_id,
                     const dim_t start_step,
                     const dim_t max_length,
                     const dim_t min_length,
                     const std::vector<size_t>* output_ids_map,
                     std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                     std::vector<std::vector<float>>* scores,
                     std::vector<std::vector<std::vector<std::vector<float>>>>* attention,
                     const size_t num_hypotheses,
                     const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("beam_search");
    const dim_t min_step = start_step + min_length;
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const bool expand_after_first_step = (device == Device::CPU);
    const dim_t batch_size = start_ids.size();
    dim_t cur_batch_size = batch_size;

    StorageView gather_indices(DataType::INT32);
    StorageView topk_ids({batch_size, 1},
                         std::vector<int32_t>(start_ids.begin(), start_ids.end()));
    StorageView topk_scores(dtype);
    StorageView topk_log_probs(dtype);

    if (!expand_after_first_step) {
      expand_to_beam_size(state, _beam_size);
      expand_to_beam_size(topk_ids, _beam_size);
      TYPE_DISPATCH(dtype, initialize_cum_log_probs<T>(topk_log_probs, batch_size, _beam_size));
    }

    using Result = std::pair<std::vector<size_t>, std::vector<std::vector<float>>>;
    std::vector<std::map<float, Result>> hypotheses;
    hypotheses.resize(batch_size);
    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    if (scores) {
      scores->clear();
      scores->resize(batch_size);
    }
    if (attention) {
      attention->clear();
      attention->resize(batch_size);
    }

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].reserve(num_hypotheses);
      if (scores)
        (*scores)[i].reserve(num_hypotheses);
      if (attention)
        (*attention)[i].reserve(num_hypotheses);
    }

    StorageView logits(dtype, device);
    StorageView log_probs(dtype, device);
    StorageView alive_seq(topk_ids.dtype());
    StorageView alive_attention;
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    StorageView coverage;

    for (dim_t step = start_step; step < max_step; ++step) {
      // Compute log probs for the current step.
      decoder(step,
              topk_ids.to(device),
              state,
              &logits,
              (attention || _coverage_penalty != 0) ? &attention_step_device : nullptr);
      ops::LogSoftMax()(logits, log_probs);
      const dim_t vocabulary_size = log_probs.dim(-1);
      const bool is_expanded = (!expand_after_first_step || step > start_step);

      // Multiply by the current beam log probs.
      if (is_expanded) {
        DEVICE_DISPATCH(
          log_probs.device(),
          TYPE_DISPATCH(log_probs.dtype(),
                        primitives<D>::add_depth_broadcast(topk_log_probs.to(device).data<T>(),
                                                           log_probs.data<T>(),
                                                           topk_log_probs.size(),
                                                           log_probs.size())));
      }

      // Penalize by the length, if enabled.
      float length_penalty_weight = 1.0;
      if (_length_penalty != 0) {
        length_penalty_weight = std::pow((5.0 + static_cast<float>(step + 1)) / 6.0, _length_penalty);
        ops::Mul()(log_probs,
                   StorageView(1.f / length_penalty_weight).to(log_probs.dtype()),
                   log_probs);
      }

      // Penalize end_id, if configured.
      if (step < min_step)
        penalize_token(log_probs, end_id);

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, -1});

      // TopK candidates.
      sampler(log_probs, topk_ids, topk_scores, _beam_size);
      if (prefix_ids)
        update_sample_with_prefix(step, topk_ids, topk_scores, *prefix_ids, batch_offset);

      topk_log_probs = topk_scores;
      // Recover the true log probs if length penalty was applied.
      if (_length_penalty != 0)
        ops::Mul()(topk_log_probs,
                   StorageView(length_penalty_weight).to(topk_log_probs.dtype()),
                   topk_log_probs);

      // Unflatten the ids.
      gather_indices.resize({cur_batch_size * _beam_size});
      for (dim_t i = 0; i < topk_ids.size(); ++i) {
        auto flat_id = topk_ids.at<int32_t>(i);
        auto beam_id = flat_id / vocabulary_size;
        auto word_id = flat_id % vocabulary_size;
        auto batch_id = i / _beam_size;
        if (output_ids_map)
          word_id = output_ids_map->at(word_id);
        topk_ids.at<int32_t>(i) = word_id;
        // On the first step, batches are not yet replicated beam_size times.
        gather_indices.at<int32_t>(i) = (is_expanded
                                         ? beam_id + batch_id * _beam_size
                                         : batch_id);
      }

      // Append last prediction.
      topk_ids.reshape({cur_batch_size, _beam_size, 1});
      if (alive_seq) {
        gather(alive_seq, gather_indices);
        alive_seq.reshape({cur_batch_size, _beam_size, alive_seq.dim(-1)});
        StorageView cur_alive_seq(std::move(alive_seq));
        ops::Concat(-1)({&cur_alive_seq, &topk_ids}, alive_seq);
      } else {
        alive_seq = topk_ids;
      }

      topk_log_probs.reshape({cur_batch_size, _beam_size});
      topk_scores.reshape({cur_batch_size, _beam_size});
      topk_ids.reshape({cur_batch_size, _beam_size});

      if (attention_step_device) {
        attention_step.copy_from(attention_step_device.to_float());
        if (!is_expanded) {
          expand_to_beam_size(attention_step, _beam_size);
        }
      }

      if (_coverage_penalty != 0) {
        if (!coverage) {
          coverage = attention_step;
        } else {
          gather(coverage, gather_indices);
          ops::Add()(attention_step, coverage, coverage);
        }
        StorageView tmp(dtype, device);
        ops::Min()(coverage, 1.0f, tmp);
        ops::Log()(tmp, tmp);
        tmp.reshape({-1, tmp.dim(-1)});
        StorageView penalty(dtype, device);
        ops::MatMul()(tmp, StorageView({tmp.dim(-1), 1}, 1.0f), penalty);
        ops::Mul()(penalty, StorageView(_coverage_penalty), penalty);
        ops::Add()(penalty.to(topk_scores.dtype()), topk_scores, topk_scores);
      }

      if (attention) {
        if (!alive_attention) {
          alive_attention = attention_step;
        } else {
          gather(alive_attention, gather_indices);
          StorageView cur_alive_attention(std::move(alive_attention));
          ops::Concat(1)({&cur_alive_attention, &attention_step}, alive_attention);
        }
        alive_attention.reshape({cur_batch_size,
                                 _beam_size,
                                 alive_attention.dim(1),
                                 alive_attention.dim(2)});
      }

      // Check if some hypotheses are finished.
      std::vector<bool> finished(cur_batch_size, false);
      dim_t finished_count = 0;
      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        for (dim_t k = 0; k < _beam_size; ++k) {
          if (topk_ids.at<int32_t>({i, k}) == static_cast<int32_t>(end_id)
              || step + 1 == max_step) {
            if (k == 0)
              top_beam_finished[i] = true;
            float score = topk_scores.scalar_at<float>({i, k});
            // Prevent this beam from advancing in the next step.
            TYPE_DISPATCH(dtype, topk_log_probs.at<T>({i, k}) = T(-1e10));
            // Save the finished hypothesis only if it is still a candidate.
            if (hypotheses[batch_id].size() < num_hypotheses
                || -score < hypotheses[batch_id].rbegin()->first) {
              std::vector<size_t> hypothesis;
              std::vector<std::vector<float>> attn;
              const dim_t max_time = alive_seq.dim(-1);
              hypothesis.reserve(max_time);
              if (attention)
                attn.reserve(max_time);
              for (dim_t t = 0; t < max_time; ++t) {
                const int32_t id = alive_seq.at<int32_t>({i, k, t});
                if (id == static_cast<int32_t>(end_id))
                  break;
                hypothesis.push_back(id);
                if (attention) {
                  const auto* attn_vec = alive_attention.index<float>({i, k, t});
                  attn.emplace_back(attn_vec, attn_vec + alive_attention.dim(-1));
                }
              }

              // Use -score as the key to iterate the map from best to worst.
              hypotheses[batch_id].emplace(std::piecewise_construct,
                                           std::forward_as_tuple(-score),
                                           std::forward_as_tuple(std::move(hypothesis),
                                                                 std::move(attn)));
            }
          }
        }

        if (top_beam_finished[i] && hypotheses[batch_id].size() >= num_hypotheses) {
          ++finished_count;
          finished[i] = true;

          // Return the "num_hypotheses" best hypotheses.
          for (auto& pair : hypotheses[batch_id]) {
            if (sampled_ids[batch_id].size() >= num_hypotheses)
              break;
            sampled_ids[batch_id].emplace_back(std::move(pair.second.first));
            if (scores) {
              (*scores)[batch_id].push_back(-pair.first);
            }
            if (attention) {
              (*attention)[batch_id].emplace_back(std::move(pair.second.second));
            }
          }
          hypotheses[batch_id].clear();
        }
      }

      // If all remaining sentences are finished, no need to go further.
      if (finished_count == cur_batch_size) {
        if (!is_expanded) {
          // We should ensure that states are replicated before exiting this function.
          expand_to_beam_size(state, _beam_size);
        }
        break;
      }

      // If some sentences finished on this step, ignore them for the next step.
      if (finished_count > 0) {
        auto old_batch_size = cur_batch_size;
        cur_batch_size -= finished_count;
        StorageView keep_batches({cur_batch_size}, DataType::INT32);
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished.size(); ++read_index) {
          if (!finished[read_index]) {
            keep_batches.at<int32_t>(write_index) = read_index;
            top_beam_finished[write_index] = top_beam_finished[read_index];
            batch_offset[write_index] = batch_offset[read_index];
            ++write_index;
          }
        }
        batch_offset.resize(write_index);
        gather(topk_ids, keep_batches);
        gather(topk_log_probs, keep_batches);
        gather(alive_seq, keep_batches);
        if (attention)
          gather(alive_attention, keep_batches);

        // On CPU, we reorder first and then remove finished batches. Otherwise, we remove
        // finished batches from the reorder indices and then reorder. The motivation for this
        // difference is to enable the fast in place gather on CPU for state elements that should
        // not be reordered (see Decoder::gather_state and Gather::operator()).

        if (device == Device::CPU) {
          decoder.gather_state(state, gather_indices);
          for (auto& pair : state)
            gather_batch(pair.second, keep_batches, _beam_size);
        } else {
          gather_batch(gather_indices, keep_batches, _beam_size);
          decoder.gather_state(state, gather_indices.to(device));
        }

        if(_coverage_penalty != 0){
          coverage.reshape({old_batch_size, _beam_size, coverage.dim(1), coverage.dim(2)});
          gather(coverage, keep_batches);
          coverage.reshape({cur_batch_size * _beam_size, coverage.dim(2), coverage.dim(3)});
        }
      } else {
        decoder.gather_state(state, gather_indices.to(device));
      }

      topk_ids.reshape({cur_batch_size * _beam_size, 1});
      topk_log_probs.reshape({cur_batch_size * _beam_size});
      alive_seq.reshape({cur_batch_size * _beam_size, alive_seq.dim(-1)});
      if (attention)
        alive_attention.reshape({cur_batch_size * _beam_size,
                                 alive_attention.dim(2),
                                 alive_attention.dim(3)});
    }
  }

  void
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const size_t end_id,
                       const dim_t start_step,
                       const dim_t max_length,
                       const dim_t min_length,
                       const std::vector<size_t>* output_ids_map,
                       std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                       std::vector<std::vector<float>>* scores,
                       std::vector<std::vector<std::vector<std::vector<float>>>>* attention,
                       const size_t,
                       const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("greedy_search");
    const dim_t min_step = start_step + min_length;
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const dim_t batch_size = start_ids.size();
    StorageView sample_from({batch_size, 1},
                            std::vector<int32_t>(start_ids.begin(), start_ids.end()));

    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    if (scores) {
      scores->clear();
      scores->resize(batch_size);
    }
    if (attention) {
      attention->clear();
      attention->resize(batch_size);
    }

    StorageView logits(dtype, device);
    StorageView log_probs(dtype, device);
    StorageView alive({batch_size}, DataType::INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].resize(1);
      if (scores)
        (*scores)[i].resize(1);
      if (attention)
        (*attention)[i].resize(1);
    }

    StorageView best_ids( DataType::INT32);
    StorageView best_probs(dtype);
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    for (dim_t step = start_step; step < max_step; ++step) {
      decoder(step,
              sample_from.to(device),
              state,
              &logits,
              attention ? &attention_step_device : nullptr);

      // Compute log probs only if scores should be returned.
      if (scores) {
        ops::LogSoftMax()(logits, log_probs);
      } else {
        log_probs.shallow_copy(logits);
      }

      // Penalize end_id, if configured.
      if (step < min_step)
        penalize_token(log_probs, end_id);

      sampler(log_probs, best_ids, best_probs);
      if (prefix_ids)
        update_sample_with_prefix(step, best_ids, best_probs, *prefix_ids, batch_offset);
      if (attention)
        attention_step.copy_from(attention_step_device.to_float());

      std::vector<bool> finished_batch(log_probs.dim(0), false);
      bool one_finished = false;
      dim_t count_alive = 0;
      for (dim_t i = 0; i < log_probs.dim(0); ++i) {
        int32_t true_id = best_ids.scalar_at<int32_t>({i});
        if (output_ids_map)
          true_id = output_ids_map->at(true_id);
        dim_t batch_id = batch_offset[i];
        if (true_id == static_cast<int32_t>(end_id)) {
          finished[batch_id] = true;
          finished_batch[i] = true;
          one_finished = true;
        } else {
          sample_from.at<int32_t>(i) = true_id;
          sampled_ids[batch_id][0].push_back(true_id);
          ++count_alive;
          if (scores) {
            (*scores)[batch_id][0] += best_probs.scalar_at<float>({i});
          }
          if (attention) {
            const auto* attn = attention_step.index<float>({i});
            (*attention)[batch_id][0].emplace_back(attn, attn + attention_step.dim(-1));
          }
        }
      }

      // No more sentences are alive, stop here.
      if (count_alive == 0)
        break;

      // Remove finished sentences from the execution.
      if (one_finished) {
        alive.resize({count_alive});
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished_batch.size(); ++read_index) {
          if (!finished_batch[read_index]) {
            batch_offset[write_index] = batch_offset[read_index];
            alive.at<int32_t>(write_index) = read_index;
            ++write_index;
          }
        }
        batch_offset.resize(write_index);
        gather(sample_from, alive);
        auto alive_device = alive.to(device);
        decoder.gather_state(state, alive_device);
      }
    }
  }

  void initialize_decoder_with_prefix(layers::Decoder& decoder,
                                      layers::DecoderState& state,
                                      const std::vector<size_t>& start_ids,
                                      const std::vector<size_t>& prefix_ids,
                                      std::vector<std::vector<float>>* prefix_attention) {
    const Device device = decoder.device();
    const size_t prefix_size = prefix_ids.size();

    StorageView input({1, 1}, std::vector<int32_t>(start_ids.begin(), start_ids.end()));
    StorageView attention(device);
    if (prefix_attention)
      prefix_attention->reserve(prefix_size);

    for (size_t i = 0; i < prefix_size; ++i) {
      decoder(i,
              input.to(device),
              state,
              /*logits=*/nullptr,
              prefix_attention ? &attention : nullptr);
      if (prefix_attention)
        prefix_attention->emplace_back(attention.to_vector<float>());
      input.at<int32_t>(0) = prefix_ids[i];
    }
  }

  template <typename T>
  static std::vector<std::vector<T>> unflatten_hypotheses(std::vector<std::vector<T>> array,
                                                          const size_t batch_size,
                                                          const size_t num_hypotheses) {
    // Reshape array from batch_size*num_hypotheses x 1 to batch_size x num_hypotheses.
    if (array.empty())
      return array;
    std::vector<std::vector<T>> new_array;
    new_array.reserve(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      std::vector<T> hypotheses;
      hypotheses.reserve(num_hypotheses);
      for (size_t i = 0; i < num_hypotheses; ++i) {
        hypotheses.emplace_back(std::move(array[b * num_hypotheses + i][0]));
      }
      new_array.emplace_back(std::move(hypotheses));
    }
    return new_array;
  }

  std::vector<GenerationResult<size_t>>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         const SearchStrategy& search_strategy,
         const Sampler& sampler,
         std::vector<size_t> start_ids,
         const std::vector<std::vector<size_t>>* prefix_ids,
         const std::vector<size_t>* output_ids_map,
         const size_t end_id,
         dim_t max_length,
         dim_t min_length,
         const size_t num_hypotheses,
         const bool return_alternatives,
         const bool return_scores,
         const bool return_attention) {
    const size_t batch_size = start_ids.size();
    dim_t start_step = 0;

    std::vector<std::vector<std::vector<float>>> prefix_attention;
    std::vector<std::vector<std::vector<size_t>>> expanded_ids;
    std::vector<std::vector<float>> expanded_scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> expanded_attention;
    if (return_alternatives) {
      if (prefix_ids) {
        if (prefix_ids->size() > 1)
          throw std::invalid_argument("Returning alternatives from a prefix is not supported "
                                      "in batch mode");
        if (return_attention)
          prefix_attention.resize(1);
        initialize_decoder_with_prefix(decoder,
                                       state,
                                       start_ids,
                                       prefix_ids->front(),
                                       return_attention ? &prefix_attention[0] : nullptr);
        start_ids[0] = prefix_ids->front().back();
        const dim_t prefix_length = prefix_ids->front().size();
        start_step += prefix_length;
        max_length = std::max(max_length - prefix_length, dim_t(0));
        min_length = std::max(min_length - prefix_length, dim_t(0));
      }

      // In this translation mode, we first expand the next "num_hypotheses" candidate words
      // before running the full decoding on each prefix. This is to ensure that we get unique
      // alternatives at this decoding position.
      BeamSearch(num_hypotheses).search(decoder,
                                        state,
                                        BestSampler(),
                                        start_ids,
                                        end_id,
                                        start_step,
                                        /*max_length=*/1,
                                        /*min_length=*/1,
                                        output_ids_map,
                                        expanded_ids,
                                        return_scores ? &expanded_scores : nullptr,
                                        return_attention ? &expanded_attention : nullptr,
                                        num_hypotheses);

      // The next input is the words we just expanded.
      start_ids.resize(batch_size * num_hypotheses);
      for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_hypotheses; ++i) {
          start_ids[b * num_hypotheses + i] = expanded_ids[b][i].back();
        }
      }
      start_step += 1;
      max_length = std::max(max_length - 1, dim_t(0));
      min_length = std::max(min_length - 1, dim_t(0));
    }

    std::vector<std::vector<std::vector<size_t>>> sampled_ids;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> attention;
    search_strategy.search(decoder,
                           state,
                           sampler,
                           start_ids,
                           end_id,
                           start_step,
                           max_length,
                           min_length,
                           output_ids_map,
                           sampled_ids,
                           return_scores ? &scores : nullptr,
                           return_attention ? &attention : nullptr,
                           return_alternatives ? 1 : num_hypotheses,
                           return_alternatives ? nullptr : prefix_ids);

    if (return_alternatives) {
      // Convert outputs from shape batch_size*num_hypotheses x 1 to batch_size x num_hypotheses.
      sampled_ids = unflatten_hypotheses(std::move(sampled_ids), batch_size, num_hypotheses);
      scores = unflatten_hypotheses(std::move(scores), batch_size, num_hypotheses);
      attention = unflatten_hypotheses(std::move(attention), batch_size, num_hypotheses);
    }

    // Build results.
    std::vector<GenerationResult<size_t>> results;
    results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {

      // Aggregate result from the optional prefix and expansion step.
      if (return_alternatives) {

        for (size_t h = 0; h < num_hypotheses; ++h) {
          // Finalize the generated ids.
          std::vector<size_t>& ids = sampled_ids[i][h];
          if (!expanded_ids.empty())
            ids.insert(ids.begin(), expanded_ids[i][h][0]);
          if (prefix_ids)
            ids.insert(ids.begin(), prefix_ids->at(i).begin(), prefix_ids->at(i).end());

          // Finalize the score.
          if (return_scores && !expanded_scores.empty())
            scores[i][h] += expanded_scores[i][h];

          // Finalize the attention.
          if (return_attention) {
            std::vector<std::vector<float>>& attn = attention[i][h];
            if (!expanded_attention.empty())
              attn.insert(attn.begin(), expanded_attention[i][h][0]);
            if (!prefix_attention.empty())
              attn.insert(attn.begin(),
                          prefix_attention[i].begin(),
                          prefix_attention[i].end());
          }
        }
      }

      GenerationResult<size_t> result(std::move(sampled_ids[i]));
      if (return_scores)
        result.set_scores(std::move(scores[i]));
      if (return_attention)
        result.set_attention(std::move(attention[i]));
      results.emplace_back(std::move(result));
    }

    return results;
  }

}
