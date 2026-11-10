#! /bin/bash

set -e
set -x

pip install "cmake==3.24.*"

if [ "$CIBW_ARCHS" == "aarch64" ]; then

    OPENBLAS_VERSION=0.3.26
    curl -L -O https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz
    tar xf *.tar.gz && rm *.tar.gz
    cd OpenBLAS-*
    # NUM_THREADS: maximum value for intra_threads
    # NUM_PARALLEL: maximum value for inter_threads
    make -j$(nproc) TARGET=ARMV8 NO_SHARED=1 BUILD_SINGLE=1 NO_LAPACK=1 ONLY_CBLAS=1 USE_OPENMP=1
    make -j$(nproc) install NO_SHARED=1
    cd ..
    rm -r OpenBLAS-*

else
    dnf install -y dnf-plugins-core
    # Install CUDA 12.8:
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    # error mirrorlist.centos.org doesn't exists anymore.
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
    dnf install --setopt=obsoletes=0 -y \
        cuda-nvcc-12-8-12.8.93-1 \
        cuda-cudart-devel-12-8-12.8.90-1 \
        libcurand-devel-12-8-10.3.9.90-1 \
        libcudnn9-devel-cuda-12-9.10.2.21-1 \
        libcublas-devel-12-8-12.8.4.1-1 \
        libnccl-2.26.2-1+cuda12.8 \
        libnccl-devel-2.26.2-1+cuda12.8
    ln -s cuda-12.8 /usr/local/cuda
    # Native CUDA language support needs an explicit compiler when nvcc is not on PATH.
    export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
    "$CUDACXX" --version

    "$(dirname "$0")/install_mkl.sh" 2025.3
    "$(dirname "$0")/build_onednn.sh" 3.1.1
    "$(dirname "$0")/build_openmpi.sh" 4.1.6
    export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
fi

mkdir build-release && cd build-release

if [ "$CIBW_ARCHS" == "aarch64" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP -DCMAKE_PREFIX_PATH="/opt/OpenBLAS" -DWITH_OPENBLAS=ON -DWITH_RUY=ON ..
else
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-msse4.1" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_DYNAMIC_LOADING=ON -DCMAKE_CUDA_FLAGS="-Xfatbin=-compress-all" -DCMAKE_CUDA_ARCHITECTURES="53-real;60-real;61-real;70-real;75-real;80-real;86" -DWITH_TENSOR_PARALLEL=ON ..
fi

VERBOSE=1 make -j$(nproc) install
cd ..
rm -r build-release

cp README.md python/
