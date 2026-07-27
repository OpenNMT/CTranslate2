#!/bin/bash
set -e
set -x

brew install libomp

# Get the actual libomp path
LIBOMP_PREFIX=$(brew --prefix libomp)

# Set environment variables
export LDFLAGS="-L${LIBOMP_PREFIX}/lib"
export CPPFLAGS="-I${LIBOMP_PREFIX}/include"
export CMAKE_PREFIX_PATH="${LIBOMP_PREFIX}"

# Critical: Set OpenMP flags explicitly for CMake
export OpenMP_C_FLAGS="-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include"
export OpenMP_C_LIB_NAMES="omp"
export OpenMP_CXX_FLAGS="-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include"
export OpenMP_CXX_LIB_NAMES="omp"
export OpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"

mkdir build-release && cd build-release

CMAKE_EXTRA_OPTIONS=''

if [ "$CIBW_ARCHS" == "arm64" ]; then

    CMAKE_EXTRA_OPTIONS='-DCMAKE_OSX_ARCHITECTURES=arm64 -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_RUY=ON'

else

    # Install OneAPI MKL
    # See https://github.com/oneapi-src/oneapi-ci for installer URLs
    ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cd013e6c-49c4-488b-8b86-25df6693a9b7/m_BaseKit_p_2023.2.0.49398.dmg
    wget -q $ONEAPI_INSTALLER_URL
    hdiutil attach -noverify -noautofsck $(basename $ONEAPI_INSTALLER_URL)
    sudo /Volumes/$(basename $ONEAPI_INSTALLER_URL .dmg)/bootstrapper.app/Contents/MacOS/bootstrapper --silent --eula accept --components intel.oneapi.mac.mkl.devel

    ONEDNN_VERSION=3.1.1
    wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
    tar xf *.tar.gz && rm *.tar.gz
    cd oneDNN-*
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DCMAKE_BUILD_TYPE=Release \
          -DONEDNN_LIBRARY_TYPE=STATIC \
          -DONEDNN_BUILD_EXAMPLES=OFF \
          -DONEDNN_BUILD_TESTS=OFF \
          -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
          -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" \
          -DONEDNN_BUILD_GRAPH=OFF \
          -DOpenMP_C_FLAGS="${OpenMP_C_FLAGS}" \
          -DOpenMP_CXX_FLAGS="${OpenMP_CXX_FLAGS}" \
          -DOpenMP_omp_LIBRARY="${OpenMP_omp_LIBRARY}" \
          -DCMAKE_C_FLAGS="${CPPFLAGS}" \
          -DCMAKE_CXX_FLAGS="${CPPFLAGS}" .
    sudo make -j$(sysctl -n hw.physicalcpu_max) install
    cd ..
    rm -r oneDNN-*

    CMAKE_EXTRA_OPTIONS='-DWITH_DNNL=ON'

fi

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CLI=OFF \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DOpenMP_C_FLAGS="${OpenMP_C_FLAGS}" \
      -DOpenMP_CXX_FLAGS="${OpenMP_CXX_FLAGS}" \
      -DOpenMP_omp_LIBRARY="${OpenMP_omp_LIBRARY}" \
      $CMAKE_EXTRA_OPTIONS ..

sudo VERBOSE=1 make -j$(sysctl -n hw.physicalcpu_max) install
cd ..
rm -r build-release

cp README.md python/
