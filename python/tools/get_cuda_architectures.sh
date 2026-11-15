#!/bin/bash

# Default targets for the CUDA 12 release builds.
set -e

if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    printf '%s\n' "$CMAKE_CUDA_ARCHITECTURES"
    exit 0
fi

cuda_compiler=${1:-nvcc}
cuda_version=$("$cuda_compiler" --version)
if [[ "$cuda_version" =~ release\ ([0-9]+)\.([0-9]+) ]]; then
    cuda_major=${BASH_REMATCH[1]}
    cuda_minor=${BASH_REMATCH[2]}
else
    echo "Cannot determine CUDA version from $cuda_compiler" >&2
    exit 1
fi

architectures="53-real;60-real;61-real;70-real;75-real;80-real;86"
if (( cuda_major > 12 || (cuda_major == 12 && cuda_minor >= 8) )); then
    architectures+=";100;120"
fi
printf '%s\n' "$architectures"
