#! /bin/bash

set -e
set -x

rm -rf /host/usr/local/lib/{android,node_modules}
rm -rf /host/usr/local/.ghcup
rm -rf /host/usr/local/share/{powershell,chromium}
rm -rf /host/usr/local/julia*
rm -rf /host/usr/share/{dotnet,swift}
rm -rf /host/usr/share/az_*
rm -rf /host/usr/lib/{jvm,google-cloud-sdk}
rm -rf /host/opt/hostedtoolcache/{CodeQL,go,node,Ruby}
rm -rf /host/opt/{microsoft,az,google}
df -h

export LIBRARY_PATH="/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14:${LIBRARY_PATH:-}"

tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm 7.2.0 repository
baseurl=https://repo.radeon.com/rocm/el8/7.2/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key

[amdgraphics]
name=AMD Graphics 7.2.0 repository
baseurl=https://repo.radeon.com/graphics/7.2/el/8/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
dnf clean all
dnf install -y rocm-hip-sdk

export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/amdgcn/bitcode
export HIP_CLANG_ROOT=$ROCM_PATH/lib/llvm
export PYTORCH_ROCM_ARCH="gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

pip install "cmake==3.22.*"

ONEAPI_VERSION=2025.3.0
dnf config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
dnf install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION

ONEDNN_VERSION=3.10.2
curl -L -O https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
tar xf *.tar.gz && rm *.tar.gz
cd oneDNN-*
cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
make -j$(nproc) install
cd ..
rm -r oneDNN-*

OPENMPI_VERSION=4.1.6
curl -L -O https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.bz2
tar xf *.tar.bz2 && rm *.tar.bz2
cd openmpi-*
./configure
make -j$(nproc) install
cd ..
rm -r openmpi-*
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"

mkdir build-release && cd build-release

cmake -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-msse4.1 -Wno-deprecated-literal-operator" -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DOPENMP_RUNTIME=COMP -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$PYTORCH_ROCM_ARCH" ..

VERBOSE=1 make -j$(nproc) install
cd ..
rm -r build-release

cp README.md python/
