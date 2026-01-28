#! /bin/bash

set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
./cuda.exe -s nvcc_12.8 cudart_12.8 cublas_dev_12.8 curand_dev_12.8

rm cuda.exe

CUDNN_ROOT="C:/Program Files/NVIDIA/CUDNN/v9.10"
curl --netrc-optional -L -nv -o cudnn.exe https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn_9.10.2_windows.exe
./cudnn.exe -s
sleep 10
# Remove 11.8 folders
rm -rf "$CUDNN_ROOT/bin/11.8"
rm -rf "$CUDNN_ROOT/lib/11.8"
rm -rf "$CUDNN_ROOT/include/11.8"

# Move contents of 12.9 to parent directories
mv "$CUDNN_ROOT/bin/12.9/"* "$CUDNN_ROOT/bin/"
mv "$CUDNN_ROOT/lib/12.9/"* "$CUDNN_ROOT/lib/"
mv "$CUDNN_ROOT/include/12.9/"* "$CUDNN_ROOT/include/"

# Remove empty 12.9 folders
rmdir "$CUDNN_ROOT/bin/12.9"
rmdir "$CUDNN_ROOT/lib/12.9"
rmdir "$CUDNN_ROOT/include/12.9"
cp -r "$CUDNN_ROOT"/* "$CUDA_ROOT"
rm cudnn.exe

# See https://github.com/oneapi-src/oneapi-ci for installer URLs
curl --netrc-optional -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1f18901e-877d-469d-a41a-a10f11b39336/intel-oneapi-base-toolkit-2025.3.0.372_offline.exe
./webimage.exe -s -x -f webimage_extracted --log extract.log
rm webimage.exe
./webimage_extracted/bootstrapper.exe -s --action install --components="intel.oneapi.win.mkl.devel" --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 --log-dir=.

NPROC=$(nproc)
ONEDNN_VERSION=3.1.1
curl --netrc-optional -L -O https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
tar xf *.tar.gz && rm *.tar.gz
cd oneDNN-*
cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
cmake --build . --config Release --target install --parallel $NPROC
cd ..
rm -r oneDNN-*

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CTRANSLATE2_ROOT -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win;C:/Program Files (x86)/oneDNN" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" -DCUDA_DYNAMIC_LOADING=ON -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="Common" ..
cmake --build . --config Release --target install --parallel $NPROC --verbose
cd ..
rm -r build

cp README.md python/
cp $CTRANSLATE2_ROOT/bin/ctranslate2.dll python/ctranslate2/
cp "C:/Program Files (x86)/Intel/oneAPI/2025.3/bin/libiomp5md.dll" python/ctranslate2/
cp "$CUDA_ROOT/bin/cudnn64_9.dll" python/ctranslate2/
