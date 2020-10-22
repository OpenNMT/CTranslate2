#!/bin/bash

set -e

USE_GPU=${1:0}
NUM_THREADS=4

# Build all Docker images.
docker build -t opennmt/ctranslate2-benchmark -f ctranslate2/Dockerfile .
docker build -t opennmt/opennmt-py-benchmark -f opennmt_py/Dockerfile .
docker build -t opennmt/opennmt-tf-benchmark -f opennmt_tf/Dockerfile .

# Get test data and SentencePiece model.
sacrebleu -t wmt14 -l en-de --echo src > wmt14-en-de.src
sacrebleu -t wmt14 -l en-de --echo ref > wmt14-en-de.tgt
wget -q -N https://opennmt-trainingdata.s3.amazonaws.com/wmtende.model

# Set shared arguments.
PREFIX="python benchmark.py --num_samples 3 --num_threads $NUM_THREADS --src wmt14-en-de.src --tgt wmt14-en-de.tgt --sp_model wmtende.model"

# Run benchmark.
if [[ $USE_GPU -eq 0 ]]
then

    $PREFIX --image opennmt/opennmt-tf-benchmark --name gpu-opennmt-tf-float \
            --command "%s %s 32 4"

    $PREFIX --image opennmt/opennmt-py-benchmark --name cpu-opennmt-py-float \
            --command "-src %s -output %s -beam_size 4 -batch_size 32"

    $PREFIX --image opennmt/ctranslate2-benchmark --name cpu-ctranslate2-float \
            --command "--src %s --out %s --beam_size 4 --batch_size 32"
    $PREFIX --image opennmt/ctranslate2-benchmark --name cpu-ctranslate2-int16 \
            --command "--src %s --out %s --beam_size 4 --batch_size 32 --compute_type int16"
    $PREFIX --image opennmt/ctranslate2-benchmark --name cpu-ctranslate2-int8 \
            --command "--src %s --out %s --beam_size 4 --batch_size 32 --compute_type int8"
    $PREFIX --image opennmt/ctranslate2-benchmark --name cpu-ctranslate2-int8-vmap \
            --command "--src %s --out %s --beam_size 4 --batch_size 32 --compute_type int8 --use_vmap"

else

    $PREFIX --gpu --image opennmt/opennmt-tf-benchmark --name gpu-opennmt-tf-float \
            --command "%s %s 32 4"

    $PREFIX --gpu --image opennmt/opennmt-py-benchmark --name gpu-opennmt-py-float \
            --command "-src %s -output %s -beam_size 4 -batch_size 32 -gpu 0"

    $PREFIX --gpu --image opennmt/ctranslate2-benchmark --name gpu-ctranslate2-float \
            --command "--src %s --out %s --beam_size 4 --batch_size 32"
    $PREFIX --gpu --image opennmt/ctranslate2-benchmark --name gpu-ctranslate2-float16 \
           --command "--src %s --out %s --beam_size 4 --batch_size 32 --compute_type float16"
    $PREFIX --gpu --image opennmt/ctranslate2-benchmark --name gpu-ctranslate2-int8 \
            --command "--src %s --out %s --beam_size 4 --batch_size 32 --compute_type int8"

fi
