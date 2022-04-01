#include "ctranslate2/layers/decoder.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace layers {

    void zero_first_timestep(StorageView& x, dim_t step) {
      if (step == 0) {
        x.zero();
      } else if (step < 0) {
        // TODO: a more direct way to set the first timestep to 0.
        const auto dtype = x.dtype();
        const auto device = x.device();
        StorageView first_step(dtype, device);
        StorageView other_steps(dtype, device);
        ops::Split(1, {1, x.dim(1) - 1})(x, first_step, other_steps);
        first_step.zero();
        ops::Concat(1)({&first_step, &other_steps}, x);
      }
    }


    Decoder::Decoder(Device device)
      : _device(device) {
    }

    void Decoder::gather_state(DecoderState& state, const StorageView& indices) const {
      static const ops::Gather gather_op;

      // When the batch size is unchanged, assume that we are reordering beams.
      bool beam_reordering = indices.size() == batch_size(state);

      for (auto& pair : state) {
        const auto& name = pair.first;
        auto& value = pair.second;
        if (beam_reordering && !should_reorder_state(name))
          continue;
        gather_op(value, indices);
      }
    }

    dim_t Decoder::batch_size(const DecoderState& state) const {
      return state.begin()->second.dim(0);
    }

    bool Decoder::should_reorder_state(const std::string&) const {
      return true;
    }

    Device Decoder::device() const {
      return _device;
    }

    const std::vector<size_t>*
    Decoder::update_output_layer(const dim_t size_multiple,
                                 const std::vector<size_t>& include_ids,
                                 const std::vector<size_t>& exclude_ids) {
      const dim_t current_output_size = output_size();

      if (_vocabulary_size == 0)
        _vocabulary_size = current_output_size;

      std::vector<size_t> ids;

      if (!include_ids.empty()) {
        ids = include_ids;

      } else {
        dim_t target_output_size = _vocabulary_size - exclude_ids.size();
        if (target_output_size % size_multiple != 0)
          target_output_size += size_multiple - (target_output_size % size_multiple);

        // Do not update the layer if the output size is unchanged.
        if (target_output_size == current_output_size && exclude_ids == _previous_exclude_ids)
          return _output_layer_index.empty() ? nullptr : &_output_layer_index;

        // Reset the output layer if the output size is the vocabulary size.
        if (target_output_size == _vocabulary_size && exclude_ids.empty()) {
          output_layer().select_weights(nullptr);
          _output_layer_index.clear();
          _previous_exclude_ids.clear();
          return nullptr;
        }

        ids.reserve(target_output_size);
        ids.resize(_vocabulary_size);
        std::iota(ids.begin(), ids.end(), size_t(0));
      }

      for (const size_t exclude_id : exclude_ids) {
        const auto it = std::lower_bound(ids.begin(), ids.end(), exclude_id);
        if (it != ids.end() && *it == exclude_id)
          ids.erase(it);
      }

      // Pad size to the next multiple.
      while (ids.size() % size_multiple != 0)
        ids.push_back(0);

      const dim_t output_size = ids.size();

      // Select weights.
      StorageView index({output_size}, DataType::INT32);
      for (dim_t i = 0; i < output_size; ++i)
        index.at<int32_t>(i) = ids[i];
      if (index.device() != _device)
        index = index.to(_device);
      output_layer().select_weights(&index);

      _output_layer_index = std::move(ids);
      _previous_exclude_ids = exclude_ids;
      return &_output_layer_index;
    }

  }
}
