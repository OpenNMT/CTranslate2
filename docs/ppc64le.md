# IBM Power10 -ppc64le

CTranslate2 fully supports IBM Power10 MMA and VSX extensions. Each Power10 core has 4 Matrix Math Accelerator units. For optimum performance use at least SMT4, in some cases SMT8 seems to perform better, but it is advicable to try out both. A simple way to test this is to use --intra_threads parameter to control the number of threads CTranslate2 is executing. At maximum this should be 8*number of physical cores (SMT-8).

Based on preliminary testing Power10 core offer 27-42% higher tokens/s compared to Intel Gold Core.

It should be possible to build for Power9, but missing MMA units will have significant impact on performance.

OneDNN is used for int8 matrix math that is fully utilizing MMA units, it should be possible to build with OpenBLAS for 16bit MMA usage.

## Build docker / podman container

This is the easy way:
```git clone --recursive https://github.com/OpenNMT/CTranslate2/
cd CTranslate2/docker
podman build  -t elinar.ai/ct2-ppc64le -f Dockerfile.ppc64le ..

```

Then run CTranslate2 container (substitue mount point, MODEL_LOCATION and SRC_FILE):
```podman run  --security-opt=label=disable  --ipc=host --ulimit=host -it --rm -v /tmp:/tmp  elinar.ai/ct2-ppc64le --model MODEL_LOCATION --src SRC_FILE --intra_threads 16```

## Install from sources
This build has been tested on RHEL 9 / ppc64le and requires IBM Advance Toolchain 17.0 ( https://www.ibm.com/support/pages/advance-toolchain-linux-power )
```
#sleef:
git clone -b 3.6.1 https://github.com/shibatch/sleef

cd sleef
mkdir build && cd build
cmake -DSLEEF_BUILD_INLINE_HEADERS=TRUE  -DCMAKE_CXX_FLAGS='-mcpu=power10 -mtune=power10 -O3 -std=gnu++11 -maltivec -mabi=altivec -mstrict-align ' -DCMAKE_C_COMPILER=/opt/at17.0/bin/gcc -DCMAKE_CXX_COMPILER=/opt/at17.0/bin/g++  -DAT_PATH=/opt/at17.0/ -DBUILD_SHARED_LIBS=FALSE -DBUILD_TESTS=FALSE -DENFORCE_VSX3=TRUE -DSLEEF_SHOW_CONFIG=1 -DCMAKE_BUILD_TYPE=Release   ..

cmake --build build -j --clean-first
sudo cmake --install build --prefix=/usr/


#OneDNN;
git clone  -b v3.2 --recursive https://github.com/oneapi-src/oneDNN
cd oneDNN
mkdir build && cd build
cmake -DCMAKE_CXX_FLAGS='-mcpu=power10 -mtune=power10 -O3 -maltivec' -DOPENMP_RUNTIME=COMP  ..
make -j16
sudo make install


git clone --recursive https://github.com/Dagamies/CTranslate2
cd CTranslate2
mkdir build
cd build
cmake -DWITH_CUDA=OFF -DWITH_MKL=OFF -DWITH_OPENBLAS=OFF -DWITH_DNNL=ON -DCMAKE_CXX_FLAGS='-mcpu=power10 -mtune=power10 -O3 -ffp-contract=off' -DOPENMP_RUNTIME=COMP ..
make -j16
sudo make install
sudo ldconfig -v
export LD_LIBRARY_PATH=/usr/local/lib64/

```