# CANN example query
This example demonstrates a translation query employing `CANN` using the English-German Transformer model trained with OpenNMT-py as in [CTranslate2 documentation](https://opennmt.net/CTranslate2/quickstart.html).

## Environment setup  
- Create  environment:`docker/cann/Dockerfile_cann`
- Run the container: `docker/cann/run_container_cann.sh`

## Download model
```bash
wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xf transformer-ende-wmt-pyOnmt.tar.gz
```

## Build executable
Run `examples/cann/build_run.sh`

### Expected output

```
current path: "<current path>"
input data path: "<input data path>"
[<timestamp>] [ctranslate2] [thread 49835] [info] CPU: ARM (NEON=true)
[<timestamp>] [ctranslate2] [thread 49835] [info]  - Selected ISA: NEON
[<timestamp>] [ctranslate2] [thread 49835] [info]  - Use Intel MKL: false
[<timestamp>] [ctranslate2] [thread 49835] [info]  - SGEMM backend: OpenBLAS (packed: false)
[<timestamp>] [ctranslate2] [thread 49835] [info]  - GEMM_S16 backend: none (packed: false)
[<timestamp>] [ctranslate2] [thread 49835] [info]  - GEMM_S8 backend: Ruy (packed: false, u8s8 preferred: false)
[<timestamp>] [ctranslate2] [thread 49835] [info] NPU:
[<timestamp>] [ctranslate2] [thread 49835] [info]  - Number of NPU cores: 8
[<timestamp>] [ctranslate2] [thread 49835] [info]  - aclrtRunMode: ACL_HOST
[<timestamp>] [ctranslate2] [thread 49835] [info] Loaded model <path> on device cann:0
[<timestamp>] [ctranslate2] [thread 49835] [info]  - Binary version: 6
[<timestamp>] [ctranslate2] [thread 49835] [info]  - Model specification revision: 7
[<timestamp>] [ctranslate2] [thread 49835] [info]  - Selected compute type: float32
input data:
▁H ello ▁world !
Start: Warmup examples
output:
▁Hallo ▁Welt !
input data:
▁H ello ▁world !
Start: Query examples
output:
▁Hallo ▁Welt !
```
