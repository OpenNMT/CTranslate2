#!/bin/bash

SOURCE_FILE=$2
OUTPUT_FILE=$3

EXTRA_ARGS=""
if [ ${USE_VMAP:-0} = "1" ]; then
    EXTRA_ARGS+=" --use_vmap"
fi

/opt/ctranslate2/bin/translate --model /model --src $SOURCE_FILE --out $OUTPUT_FILE --device auto --batch_size 32 --beam_size 4 --compute_type ${COMPUTE_TYPE:-default} $EXTRA_ARGS
