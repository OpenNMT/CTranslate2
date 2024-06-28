#!/bin/bash

# build image that will host CANN environment
cd ../../
docker build -t ctranslate2-aarch64 -f docker/cann/Dockerfile_cann --platform linux/arm64 .

# run the respective container
docker run  \
-d --cap-add sys_ptrace \
--pids-limit 409600 \
--privileged --shm-size=128G \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/dcmi:/usr/local/dcmi \
--name ctranslate2-aarch64 <container>
