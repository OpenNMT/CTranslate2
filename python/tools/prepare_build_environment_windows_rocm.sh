#! /bin/bash

set -e
set -x

pip install --no-cache-dir \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
rocm-sdk init

export ROCM_PATH=$(python -c "from rocm_sdk._devel import get_devel_root;print(get_devel_root().as_posix())")
export PATH="$ROCM_PATH:$PATH"
export CC="$ROCM_PATH/lib/llvm/bin/clang.exe"
export CXX="$ROCM_PATH/lib/llvm/bin/clang++.exe"

export HIP_PLATFORM="amd"
export HIP_PATH="$ROCM_PATH"
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
export HIP_CLANG_ROOT="$ROCM_PATH/lib/llvm"
export PYTORCH_ROCM_ARCH="gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

# See https://github.com/oneapi-src/oneapi-ci for installer URLs
curl --netrc-optional -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1f18901e-877d-469d-a41a-a10f11b39336/intel-oneapi-base-toolkit-2025.3.0.372_offline.exe
./webimage.exe -s -x -f webimage_extracted --log extract.log
rm webimage.exe
./webimage_extracted/bootstrapper.exe -s --action install --components="intel.oneapi.win.mkl.devel" --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 --log-dir=.

NPROC=$(nproc)
ONEDNN_VERSION=3.10.2
curl --netrc-optional -L -O https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
tar xf *.tar.gz && rm *.tar.gz
cd oneDNN-*
cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
cmake --build . --config Release --target install --parallel $NPROC
cd ..
rm -r oneDNN-*

cmake -GNinja -DCMAKE_BUILD_TYPE=Release -S . -B build -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator" -DCMAKE_INSTALL_PREFIX=$CTRANSLATE2_ROOT -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib;C:/Program Files (x86)/oneDNN" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$PYTORCH_ROCM_ARCH"
cmake --build build --config Release --target install --parallel $NPROC --verbose
rm -r build

cp README.md python/
cp $CTRANSLATE2_ROOT/bin/ctranslate2.dll python/ctranslate2/
cp "C:/Program Files (x86)/Intel/oneAPI/2025.3/bin/libiomp5md.dll" python/ctranslate2/
