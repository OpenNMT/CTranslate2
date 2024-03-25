#!/bin/bash

# execute from project root

# first build ct2lib
rm -rf build-release/
mkdir build-release && cd build-release || exit

cmake -DWITH_CANN=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP -DCMAKE_PREFIX_PATH="/opt/OpenBLAS" -DWITH_OPENBLAS=ON -DWITH_RUY=ON ..

make -j"$(nproc)"

rm CMakeCache.txt

# then build cann_run
cmake -DCMAKE_BUILD_TYPE=Release ../examples/cann/

make -j"$(nproc)"
# ./cann_run <ende_ctranslate2_path>
