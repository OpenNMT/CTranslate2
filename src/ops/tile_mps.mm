#include "ctranslate2/ops/tile.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

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

  }
}
