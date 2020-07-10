# WNGT 2020 Efficiency Shared Task

This directory contains the runtime resources used for the OpenNMT submission to the [WNGT 2020 Efficiency Shared Task](https://sites.google.com/view/wngt20/efficiency-task). The submission is described in details [here](https://www.aclweb.org/anthology/2020.ngt-1.25/).

Note: the dependencies, compilations flags, and hyperparameters are hardcoded for the purpose of this task.

## Submitted images

* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-4-3-256-2ffn-cpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-4-3-256-2ffn-gpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-4-3-256-cpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-4-3-256-gpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-6-3-256-cpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-6-3-256-gpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-base-cpu.tar
* https://opennmt-models.s3.amazonaws.com/wngt2020_opennmt_transformer-base-gpu.tar

## Rebuilding the image

The path to the model can be passed by argument when building the Docker image:

```text
docker build --build-arg MODEL_PATH=<model_path> [...]
```
