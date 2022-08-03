# Performance tips

Below are some general recommendations to further improve performance. Many of these recommendations were used in the [WNGT 2020 efficiency task submission](https://github.com/OpenNMT/CTranslate2/tree/master/examples/wngt2020).

* Set the compute type to "auto" to automatically select the fastest execution path on the current system
* Reduce the beam size to the minimum value that meets your quality requirement
* When using a beam size of 1, keep `return_scores` disabled if you are not using prediction scores: the final softmax layer can be skipped
* Set `max_batch_size` and pass a larger batch to `*_batch` methods: the input sentences will be sorted by length and split by chunk of `max_batch_size` elements for improved efficiency
* Prefer the "tokens" `batch_type` to make the total number of elements in a batch more constant
* Consider using {ref}`translation:dynamic vocabulary reduction` for translation

**On CPU**

* Use an Intel CPU supporting AVX512
* If you are processing a large volume of data, prefer increasing `inter_threads` over `intra_threads` to improve throughput
* Avoid the total number of threads `inter_threads * intra_threads` to be larger than the number of physical cores
* For single core execution on Intel CPUs, consider enabling packed GEMM (set the environment variable `CT2_USE_EXPERIMENTAL_PACKED_GEMM=1`)

**On GPU**

* Use a larger batch size
* Use a NVIDIA GPU with Tensor Cores (Compute Capability >= 7.0)
* Pass multiple GPU IDs to `device_index` to execute on multiple GPUs

## Example: maximize CPU throughput with `translate_batch`

This example demonstrates how to efficiently translate a stream of data on CPU using `translate_batch`.

```{note}
If you are translating files on disk, consider using `translate_file` directly which applies most of these optimizations by default.
```

```python
import collections
import ctranslate2
import multiprocessing
import sys

translator = ctranslate2.Translator(
    model_path,

    # Enable automatic quantization.
    compute_type="auto",

    # Set the number of workers to the number of physical cores.
    inter_threads=multiprocessing.cpu_count() // 2,

    # Each worker will use 1 thread.
    intra_threads=1,
)


def batch_iterator(lines, batch_size):
    batch = []
    for line in lines:
        tokens = line.strip().split()
        batch.append(tokens)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def translate(lines, batch_size=32):
    # We run translation requests asynchronously to ensure there is enough work
    # for all parallel workers.
    queue = collections.deque()

    for batch in batch_generator(lines, batch_size * 16):
        # The input batch will be sorted by length and split by chunks of size
        # batch_size so that the number of padding positions is reduced.
        async_results = translator.translate_batch(
            batch,
            beam_size=1,
            max_batch_size=batch_size,
            asynchronous=True,
        )

        queue.extend(async_results)

        # Try to return earlier results if they are ready.
        while queue and queue[0].done():
            yield queue.popleft().result()

    # Wait for all remaining results.
    while queue:
        yield queue.popleft().result()


for result in translate(sys.stdin):
    print(" ".join(result.hypotheses[0]))
```
