#! /bin/bash

set -e
set -x
set -o pipefail

OPENBLAS_VERSION=0.3.28
OPENBLAS_ROOT="$PWD/openblas-install"

curl --netrc-optional -L -o openblas.tar.gz https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz
tar xzf openblas.tar.gz
cd OpenBLAS-${OPENBLAS_VERSION}
(
  cmake -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DTARGET=ARMV8 -DBINARY=64 -DNOFORTRAN=ON -DBUILD_WITHOUT_LAPACK=ON -DONLY_CBLAS=ON -DCMAKE_C_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$OPENBLAS_ROOT" -B build .
  cmake --build build --config Release --target install --parallel
) > /dev/null 2>&1
cd ..
rm -rf OpenBLAS-${OPENBLAS_VERSION} openblas.tar.gz

NPROC=$(nproc)

mkdir build
cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CTRANSLATE2_ROOT -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_RUY=OFF -DOPENMP_RUNTIME=COMP -DOPENBLAS_INCLUDE_DIR="$OPENBLAS_ROOT/include/openblas" -DOPENBLAS_LIBRARY="$OPENBLAS_ROOT/lib/openblas.lib" -DBUILD_CLI=OFF ..
cmake --build . --config Release --target install --parallel $NPROC --verbose
cd ..
rm -r build

cp README.md python/
cp $CTRANSLATE2_ROOT/bin/ctranslate2.dll python/ctranslate2/
cp "$OPENBLAS_ROOT/bin/openblas.dll" python/ctranslate2/
