#! /bin/bash

set -e
set -x

build_x86 ()
{
    # Install CUDA 11.2, see:
    # * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/base/Dockerfile
    # * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/devel/Dockerfile
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    yum install --setopt=obsoletes=0 -y \
        cuda-nvcc-11-2-11.2.152-1 \
        cuda-cudart-devel-11-2-11.2.152-1 \
        libcurand-devel-11-2-10.2.3.152-1 \
        libcublas-devel-11-2-11.4.1.1043-1
    ln -s cuda-11.2 /usr/local/cuda

    ONEAPI_VERSION=2021.4.0
    MKL_BUILD=640
    DNNL_BUILD=467
    yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD intel-oneapi-dnnl-devel-$ONEAPI_VERSION-$DNNL_BUILD
    echo "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin" > /etc/ld.so.conf.d/libiomp5.conf
    echo "/opt/intel/oneapi/dnnl/latest/cpu_iomp/lib" > /etc/ld.so.conf.d/intel-dnnl.conf
    ldconfig

    pip install "cmake==3.18.4"

    mkdir build-release && cd build-release
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DWITH_DNNL=ON -DOPENMP_RUNTIME=INTEL -DWITH_CUDA=ON -DCUDA_DYNAMIC_LOADING=ON -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="Common" ..
    make -j$(nproc) install
} 

build_aarch64() 
{
    # install openblas
    yum install -y wget
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.13.tar.gz
    tar xzvf v0.3.13.tar.gz
    cd OpenBLAS-0.3.13
    make TARGET=ARMV8 CC=gcc FC=gfortran HOSTCC=gcc NO_LAPACK=1 -j $(nproc)
    make PREFIX=/usr install

    cd ..

    mkdir build-release && cd build-release

    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI=OFF \
    -DOPENMP_RUNTIME=COMP \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=ON \
    -DWITH_RUY=ON \
    ..

    make -j $(nproc) install
}


case $(uname -m) in
    'x86_64')
        build_x86;
        ;;
    'aarch64')
        build_aarch64;
        ;;
    *) echo >&2 "error: unsupported architecture $(uname -m)"; exit 1 ;; \
esac;


cd ..
rm -r build-release

cp README.md python/
