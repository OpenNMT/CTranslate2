// Appended to the shared Metal library. Router logits/weights stay in FP32.
R"METAL(
struct MoeTopKArgs { uint rows, experts, k, threads, normalize_selected; };
inline bool moe_before(float a, uint ai, float b, uint bi) {
  if (ai == 0x7fffffffu) return false;
  if (bi == 0x7fffffffu) return true;
  if (isnan(a)) return isnan(b) && ai < bi;
  return isnan(b) || a > b || (a == b && ai < bi);
}
inline float moe_probability(float value, float maximum) {
  return isnan(value) || maximum == -INFINITY ? 0.f
    : maximum == INFINITY ? float(value == maximum) : precise::exp(value - maximum);
}

kernel void moe_topk_f32(device const float* logits [[buffer(0)]],
                          device int* indices [[buffer(1)]],
                          device float* weights [[buffer(2)]],
                          constant MoeTopKArgs& a [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
  threadgroup float maxima[4];
  threadgroup uint winners[4];
  const ulong inbase = ulong(row)*a.experts, outbase = ulong(row)*a.k;
  for (uint choice = 0; choice < a.k; ++choice) {
    const uint previous = choice ? uint(indices[outbase+choice-1]) : 0x7fffffffu;
    const float threshold = choice ? logits[inbase+previous] : NAN;
    uint best_index = 0x7fffffffu;
    float best = NAN;
    for (uint e = tid; e < a.experts; e += a.threads) {
      const float value = logits[inbase+e];
      // Previously selected candidates form a prefix of this total ordering.
      // A threshold avoids scanning all earlier selected IDs for every expert.
      if (choice && !moe_before(threshold, previous, value, e)) continue;
      if (moe_before(value, e, best, best_index)) { best = value; best_index = e; }
    }
    const uint numeric = best_index != 0x7fffffffu && !isnan(best);
    const uint any_numeric = simd_max(numeric);
    const float maximum = simd_max(numeric ? best : -INFINITY);
    const uint candidate = best_index != 0x7fffffffu &&
      (any_numeric ? numeric && best == maximum : isnan(best)) ? best_index : 0x7fffffffu;
    const uint winner = simd_min(candidate);
    if (lane == 0) { maxima[simd] = any_numeric ? maximum : NAN; winners[simd] = winner; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint selected = winners[0]; float value = maxima[0];
      for (uint group = 1; group < a.threads/32; ++group)
        if (moe_before(maxima[group], winners[group], value, selected)) {
          value = maxima[group]; selected = winners[group];
        }
      indices[outbase+choice] = int(selected);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  if (tid != 0) return;
  const float maximum = logits[inbase+uint(indices[outbase])];
  // Compensated FP32 summation keeps normalization close to the CPU's FP64
  // reference even when k includes every expert. All terms are in [0,1].
  float sum = 0, correction = 0;
  for (uint i = 0; i < a.k; ++i) {
    const float probability = moe_probability(logits[inbase+uint(indices[outbase+i])], maximum);
    const float y = probability-correction;
    const float next = sum+y;
    correction = (next-sum)-y; sum = next;
    weights[outbase+i] = probability;
  }
  if (!a.normalize_selected) {
    sum = 0; correction = 0;
    for (uint e = 0; e < a.experts; ++e) {
      const float p = moe_probability(logits[inbase+e], maximum);
      const float y = p-correction, next = sum+y;
      correction = (next-sum)-y; sum = next;
    }
  }
  for (uint i = 0; i < a.k; ++i) {
    const float p = weights[outbase+i];
    float normalized = sum > 0 ? precise::divide(p, sum) : 0;
    if (sum > 0) {
      // The compensated denominator is approximately sum-correction. Retain
      // its low part through division as well, avoiding a second rounding loss.
      const float residual = fma(-normalized, sum, p) + normalized*correction;
      normalized += precise::divide(residual, sum);
    }
    weights[outbase+i] = normalized;
  }
}
)METAL"
