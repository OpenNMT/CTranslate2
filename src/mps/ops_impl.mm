// MPS direct implementations for ops that only use primitives<D>::copy / add.
// Apple Silicon uses MTLResourceStorageModeShared, so all MPS buffers are
// CPU-accessible via unified memory.  No device-to-host copies are needed.

#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include <algorithm>
#include <numeric>
#include <vector>

#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/tile.h"
#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/activation.h"
#include "ctranslate2/ops/add.h"

#include "ctranslate2/primitives.h"
#include "mps/kernels.h"
#include "mps/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

namespace {
int mps_activation_code(const ActivationType* activation) {
  if (!activation)
    return -1;
  switch (*activation) {
  case ActivationType::ReLU:
    return static_cast<int>(mps::UnaryOp::RELU);
  case ActivationType::GELUTanh:
    return static_cast<int>(mps::UnaryOp::GELU_TANH);
  case ActivationType::Swish:
    return static_cast<int>(mps::UnaryOp::SWISH);
  case ActivationType::GELU:
    return static_cast<int>(mps::UnaryOp::GELU);
  case ActivationType::GELUSigmoid:
    return static_cast<int>(mps::UnaryOp::GELU_SIGMOID);
  case ActivationType::Tanh:
    return static_cast<int>(mps::UnaryOp::TANH);
  case ActivationType::Sigmoid:
    return static_cast<int>(mps::UnaryOp::SIGMOID);
  }
  return -1;
}
}

// ============================================================
// Gather
// ============================================================

template <Device D, typename T>
void Gather::compute(const StorageView& data,
                     const StorageView& input,
                     const dim_t axis,
                     const dim_t batch_dims,
                     StorageView& output) const {
  if (axis == batch_dims) {
    const dim_t copy_size = data.stride(axis);
    const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
    const dim_t batch_size = data.size() / batch_stride;
    const dim_t num_indices = input.size();
    const dim_t num_indices_per_batch = num_indices / batch_size;
    mps::gather(DataTypeToEnum<T>::value,
                data.data<T>(),
                input.data<int32_t>(),
                output.data<T>(),
                copy_size,
                batch_stride,
                num_indices,
                num_indices_per_batch);
  } else {
    throw std::invalid_argument("Gather only supports indexing the first non batch dimension");
  }
}

#define DECLARE_GATHER(T) \
  template void Gather::compute<Device::MPS, T>( \
      const StorageView&, const StorageView&, dim_t, dim_t, StorageView&) const;
DECLARE_ALL_TYPES(DECLARE_GATHER)
#undef DECLARE_GATHER

// ============================================================
// Concat / Split / Slide  (shared helpers)
// ============================================================

namespace {
  static dim_t _css_copy_size(const StorageView& x, dim_t axis) {
    dim_t s = 1;
    for (dim_t i = axis; i < x.rank(); ++i) s *= x.dim(i);
    return s;
  }
  static dim_t _css_iter_size(const StorageView& x, dim_t axis) {
    dim_t s = 1;
    for (dim_t i = 0; i < axis; ++i) s *= x.dim(i);
    return s;
  }
}

template <Device D, typename T>
void Concat::compute(const std::vector<const StorageView*>& inputs,
                     StorageView& output) const {
  const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
  const dim_t step_size = output.dim(axis) * output.stride(axis);
  T* out = output.data<T>();

  for (const StorageView* inp : inputs) {
    const dim_t copy_size = _css_copy_size(*inp, axis);
    if (copy_size == 0) continue;
    const dim_t iter_size = _css_iter_size(*inp, axis);
    const T* x = inp->data<T>();
    for (dim_t i = 0; i < iter_size; ++i)
      primitives<Device::MPS>::copy(x + i * copy_size, out + i * step_size, copy_size);
    out += copy_size;
  }
}

template <Device D, typename T>
void Split::compute(const StorageView& input,
                    std::vector<StorageView*>& outputs) const {
  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
  const dim_t step_size = input.dim(axis) * input.stride(axis);
  const T* in = input.data<T>();

  for (StorageView* out : outputs) {
    const dim_t copy_size = _css_copy_size(*out, axis);
    if (copy_size == 0) continue;
    const dim_t iter_size = _css_iter_size(*out, axis);
    T* x = out->data<T>();
    for (dim_t i = 0; i < iter_size; ++i)
      primitives<Device::MPS>::copy(in + i * step_size, x + i * copy_size, copy_size);
    in += copy_size;
  }
}

template <Device D, typename T>
void Slide::compute(const StorageView& input,
                    StorageView& output,
                    const dim_t& index) const {
  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
  const dim_t stride_axis = input.stride(axis) == 0 ? 1 : input.stride(axis);
  const dim_t step_size = input.dim(axis) * stride_axis;
  const T* in = input.data<T>() + index * stride_axis;
  T* out = output.data<T>();

  const dim_t copy_size = _css_copy_size(output, axis);
  if (copy_size == 0) return;
  const dim_t iter_size = _css_iter_size(output, axis);
  for (dim_t i = 0; i < iter_size; ++i)
    primitives<Device::MPS>::copy(in + i * step_size, out + i * copy_size, copy_size);
}

#define DECLARE_CSS(T) \
  template void Concat::compute<Device::MPS, T>( \
      const std::vector<const StorageView*>&, StorageView&) const; \
  template void Split::compute<Device::MPS, T>( \
      const StorageView&, std::vector<StorageView*>&) const; \
  template void Slide::compute<Device::MPS, T>( \
      const StorageView&, StorageView&, const dim_t&) const;
DECLARE_ALL_TYPES(DECLARE_CSS)
#undef DECLARE_CSS

// ============================================================
// Tile
// ============================================================

template <Device D, typename T>
void Tile::compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t inner_size,
                   StorageView& output) const {
  mps::tile(DataTypeToEnum<T>::value,
            input.data<T>(),
            output.data<T>(),
            outer_size,
            inner_size,
            _num_tiles);
}

#define DECLARE_TILE(T) \
  template void Tile::compute<Device::MPS, T>( \
      const StorageView&, dim_t, dim_t, StorageView&) const;
DECLARE_ALL_TYPES(DECLARE_TILE)
#undef DECLARE_TILE

// ============================================================
// BiasAdd  (primitives<MPS>::add_batch_broadcast/add_block_broadcast are
//           implemented in primitives.mm; Add::compute is header-inline)
// ============================================================

template <Device D, typename T>
void BiasAdd::compute(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output,
                      const StorageView* residual) const {
  dim_t block = 0;
  if (_axis == -1 || _axis == value.rank() - 1) {
    block = 0;
  } else {
    const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
    for (dim_t i = axis + 1; i < value.rank(); ++i)
      block = (block == 0 ? 1 : block) * value.dim(i);
  }
  mps::bias_add(DataTypeToEnum<T>::value,
                bias.data<T>(),
                value.data<T>(),
                residual ? residual->data<T>() : nullptr,
                output.data<T>(),
                bias.size(),
                value.size(),
                block,
                mps_activation_code(_activation_type));
}

template void BiasAdd::compute<Device::MPS, float>(
    const StorageView&, const StorageView&, StorageView&, const StorageView*) const;
template void BiasAdd::compute<Device::MPS, float16_t>(
    const StorageView&, const StorageView&, StorageView&, const StorageView*) const;
template void BiasAdd::compute<Device::MPS, bfloat16_t>(
    const StorageView&, const StorageView&, StorageView&, const StorageView*) const;

// ============================================================
// TopK  (direct – MPS buffers are CPU-accessible on Apple Silicon)
// ============================================================

template <Device D, typename DataType, typename IndexType>
void TopK::compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const {
  const dim_t depth = x.dim(-1);
  const dim_t batch_size = x.size() / depth;

  if (mps::supports_topk(x.dtype(), _k)) {
    mps::topk(x.dtype(),
              x.data<DataType>(),
              values.data<DataType>(),
              indices.data<int32_t>(),
              batch_size,
              depth,
              _k);
    return;
  }

  mps::record_profile_event(mps::ProfileEvent::TopKCpu);
  mps::record_profile_event(mps::ProfileEvent::CpuFallback);
  mps::synchronize();

  const DataType* x_data   = x.data<DataType>();
  DataType*       v_data   = values.data<DataType>();
  IndexType*      i_data   = indices.data<IndexType>();

  if (_k == 1) {
    for (dim_t i = 0; i < batch_size; ++i) {
      const DataType* row = x_data + i * depth;
      const DataType* best = std::max_element(row, row + depth);
      v_data[i] = *best;
      i_data[i] = static_cast<IndexType>(std::distance(row, best));
    }
  } else {
    std::vector<IndexType> ids(depth);
    for (dim_t i = 0; i < batch_size; ++i) {
      const DataType* row = x_data + i * depth;
      DataType*       val = v_data  + i * _k;
      IndexType*      ind = i_data  + i * _k;

      std::iota(ids.begin(), ids.end(), IndexType(0));
      std::partial_sort(ids.begin(), ids.begin() + _k, ids.end(),
                        [&row](IndexType a, IndexType b) {
                          return row[a] > row[b];
                        });
      for (dim_t j = 0; j < _k; ++j) {
        ind[j] = ids[j];
        val[j] = row[ind[j]];
      }
    }
  }
}

template void TopK::compute<Device::MPS, float, int32_t>(
    const StorageView&, StorageView&, StorageView&) const;
template void TopK::compute<Device::MPS, float16_t, int32_t>(
    const StorageView&, StorageView&, StorageView&) const;
template void TopK::compute<Device::MPS, bfloat16_t, int32_t>(
    const StorageView&, StorageView&, StorageView&) const;

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_MPS
#endif  // __APPLE__
