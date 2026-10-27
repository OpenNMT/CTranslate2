# Speech synthesis

CTranslate2 integrates experimental Drama 3 text-to-speech inference.
It generates mono speech from text using a converted local checkpoint.

```{tip}
See the {ref}`guides/drama3:drama 3` guide for conversion and synthesis examples.
```

The implementation supports offline CPU synthesis with optional CUDA/HIP
acceleration for backbone and tower projections. The remaining graph and codec
execute on CPU. Voice cloning from reference audio, streaming, automatic
long-text splitting, and batching are not implemented. English and Spanish have
been exercised, but broad language coverage and speech quality have not been established. Evaluation
has found repeated codes, prolonged silence, and generation reaching the frame
limit; release acceptance remains incomplete.
