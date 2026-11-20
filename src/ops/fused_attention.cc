#include "ctranslate2/ops/fused_attention.h"

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    bool FusedAttention::compute<Device::CPU>(const StorageView&,
                                               const StorageView&,
                                               const StorageView&,
                                               const StorageView*,
                                               float,
                                               bool,
                                               StorageView&) const {
      return false;
    }

    template<>
    bool FusedAttention::compute<Device::CUDA>(const StorageView&,
                                                const StorageView&,
                                                const StorageView&,
                                                const StorageView*,
                                                float,
                                                bool,
                                                StorageView&) const {
      return false;
    }

    template<>
    bool FusedAttention::supports_interleaved_queries<Device::CPU>(DataType) const {
      return false;
    }

    template<>
    bool FusedAttention::supports_interleaved_queries<Device::CUDA>(DataType) const {
      return false;
    }

    bool FusedAttention::operator()(const StorageView& queries,
                                    const StorageView& keys,
                                    const StorageView& values,
                                    const StorageView* values_lengths,
                                    float queries_scale,
                                    bool interleaved_queries,
                                    StorageView& output) const {
      bool supported = false;
      DEVICE_DISPATCH(queries.device(),
                      supported = compute<D>(queries,
                                             keys,
                                             values,
                                             values_lengths,
                                             queries_scale,
                                             interleaved_queries,
                                             output));
      return supported;
    }

    bool FusedAttention::supports_interleaved_queries(Device device, DataType dtype) const {
      bool supported = false;
      DEVICE_DISPATCH(device, supported = supports_interleaved_queries<D>(dtype));
      return supported;
    }

  }
}
