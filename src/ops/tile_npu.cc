#include "ctranslate2/ops/tile.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t,
                       const dim_t inner_size,
                       StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                          \
    template void                                                \
    Tile::compute<Device::CANN, T>(const StorageView& input,     \
                                   const dim_t outer_size,       \
                                   const dim_t inner_size,       \
                                   StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
