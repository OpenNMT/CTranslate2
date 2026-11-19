#! /bin/bash
set -eux

ONEDNN_VERSION=${1:?oneDNN version required (e.g. 3.1.1)}
ONEDNN_REPO=${2:-oneapi-src/oneDNN}
ONEDNN_INSTALL_PREFIX=${3:-/usr/local}

workdir=$(mktemp -d)
trap "rm -rf $workdir" EXIT

curl -L https://github.com/${ONEDNN_REPO}/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz | tar xz -C "$workdir"
cmake -S "$workdir/oneDNN-${ONEDNN_VERSION}" -B "$workdir/build" -DCMAKE_INSTALL_PREFIX="$ONEDNN_INSTALL_PREFIX" -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF
cmake --build "$workdir/build" -j "$(nproc)" --target install
