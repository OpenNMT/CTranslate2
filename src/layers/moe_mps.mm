#include "mps/kernels.h"
#include "mps/moe.h"
#include <climits>
#include <cstdlib>

namespace ctranslate2 {
  namespace mps {
    void
    moe_forward(const StorageView& input, const StorageView& ids,
                const StorageView& weights,
                const std::vector<std::array<const StorageView*, 3>>& experts,
                StorageView& output, bool force_gemv, bool force_small_tiles,
                const std::vector<std::array<const StorageView*, 3>>* scales,
                bool force_generic_plan) {
      const dim_t rows = input.dim(0), hidden = input.dim(1), topk = ids.dim(1);
      if (experts.empty())
        throw std::invalid_argument("MPS MoE requires at least one expert");
      const dim_t count = experts.size(),
                  intermediate = experts.front()[0]->dim(0);
      // Check all GPU index spans before allocating the grouped workspaces.
      if (rows <= 0 || hidden <= 0 || intermediate <= 0 || topk <= 0 ||
          topk > count || count > INT32_MAX ||
          uint64_t(rows) > INT32_MAX / uint64_t(topk) ||
          uint64_t(rows) * topk >
              UINT32_MAX / uint64_t(std::max(hidden, intermediate)))
        throw std::invalid_argument("Unsupported MPS MoE workspace dimensions");
      const auto dtype = input.dtype();
      const bool decode = force_gemv || moe_use_gemv(rows);
      static const bool disable_short_gemm = [] {
        const char* value = std::getenv("CT2_MPS_DISABLE_SHORT_MOE_GEMM");
        return value && value[0] && std::string(value) != "0";
      }();
      const bool wide = !decode && !force_small_tiles &&
                        dtype == DataType::FLOAT16 &&
                        (rows >= 64 || !disable_short_gemm);
      StorageView offsets({count + 1}, DataType::INT32, Device::MPS),
          permutation({rows, topk}, DataType::INT32, Device::MPS),
          inverse(permutation.shape(), DataType::INT32, Device::MPS),
          indirect({count, 6}, DataType::INT32, Device::MPS),
          activated({rows * topk, intermediate}, dtype, Device::MPS),
          projected({rows * topk, hidden}, dtype, Device::MPS);
      moe_dispatch_plan(ids.data<int32_t>(), offsets.data<int32_t>(),
                        permutation.data<int32_t>(), inverse.data<int32_t>(),
                        indirect.data<int32_t>(), rows, count, topk, hidden,
                        intermediate, decode, wide, force_generic_plan);
      for (dim_t e = 0; e < count; ++e) {
        moe_expert_project(
            dtype, input.buffer(), experts[e][0]->buffer(),
            experts[e][1]->buffer(), activated.buffer(),
            offsets.data<int32_t>(), permutation.data<int32_t>(),
            indirect.data<int32_t>(), rows, count, e, topk, intermediate,
            hidden, true, decode, wide,
            scales && (*scales)[e][0] ? (*scales)[e][0]->data<float>()
                                      : nullptr,
            scales && (*scales)[e][1] ? (*scales)[e][1]->data<float>()
                                      : nullptr);
        moe_expert_project(
            dtype, activated.buffer(), experts[e][2]->buffer(), nullptr,
            projected.buffer(), offsets.data<int32_t>(),
            permutation.data<int32_t>(), indirect.data<int32_t>(), rows, count,
            e, topk, hidden, intermediate, false, decode, wide,
            scales && (*scales)[e][2] ? (*scales)[e][2]->data<float>()
                                      : nullptr);
      }
      output.resize({rows, hidden});
      moe_aggregate(dtype, projected.buffer(), inverse.data<int32_t>(),
                    ids.data<int32_t>(), weights.data<float>(), output.buffer(),
                    rows, count, topk, hidden);
    }
  } // namespace mps
} // namespace ctranslate2
