# Drama3 execution layers

These headers are private implementation details and are not installed with the
public CTranslate2 headers. Drama3 and FishDAC layers live here; model-specific
GPU helpers live in `src/ops/drama`. Reusable operators live in `src/ops`, with
public headers in `include/ctranslate2/ops` and tests in `tests`.
Core attention shares its dot-product implementation
through the private `src/layers/attention_utils.h` interface.

Each `Drama3Replica` owns a complete set of layers. A replica pool worker runs
one request at a time on that set. Layers can update lazy GPU projections, MoE
residency, and rotary tables, including from const methods; they must not be
shared between workers. Immutable CPU model weights are shared across replicas.
Sequence caches and sampling generators are created separately for each request.

The public entry point is `models::Drama3`, which schedules synchronous, async,
and batch submissions through `ReplicaPool`. Batch submission schedules individual
sequences; it does not fuse them into a tensor batch.
