# Windows Ruy shutdown deadlock — minimal repro

Self-contained repro for the shutdown hang fixed by
[OpenNMT/CTranslate2#2076](https://github.com/OpenNMT/CTranslate2/pull/2076)
(reported downstream as jkawamoto/ctranslate2-rs#64).

`repro.cpp` builds a CPU int8 `Translator` (Ruy backend) with worker threads, runs one
large translation batch, then destroys the `Translator`. On an unpatched build the
per-thread `ruy::Context` destructor joins Ruy's thread pool from a worker thread that
is exiting under the Windows loader lock, and that join deadlocks.

## The one thing that matters: batch size

The batch must be large enough that Ruy actually spawns its internal thread pool. The
destructor only deadlocks when there are Ruy worker threads to join; a tiny batch runs
single-threaded, has nothing to join, and shuts down cleanly even unpatched. That is
easy to trip over when writing a test. This repro uses a 512-sentence batch.

It is *not* specific to a CRT model or link mode. Measured on Windows 11, MSVC 14.44,
x64, unpatched CTranslate2 `0d8bcd36`, Ruy int8, `intra_threads=4`, `inter_threads=2`,
512-sentence batch:

| Build of CTranslate2 + this repro | Result |
|---|---|
| static lib + static CRT (`/MT`)   | **hangs on shutdown** |
| shared lib + dynamic CRT (`/MD`)  | **hangs on shutdown** |

The `/MD` shared build is the configuration of the official wheels, so they are affected
too. Rebuilding either with #2076 applied, both exit cleanly (`SURVIVED`).

## Build & run

Needs CMake, MSVC, and a CTranslate2 source checkout with submodules initialized
(`git submodule update --init --recursive`). No CUDA / MKL / oneDNN required. `CT2_DIR`
points at a CTranslate2 source tree; from here in the tree that is the repo root,
`../..`. Run these from this directory (`tools/ruy_shutdown_repro`):

```sh
cmake -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCT2_DIR=../.. \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
  -S . -B build
cmake --build build --config Release --target repro
build/Release/repro.exe ../../tests/data/models/v2/aren-transliteration
```

For the shared `/MD` variant use `-DBUILD_SHARED_LIBS=ON
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`, and put the built `ctranslate2.dll`
(under `build/ctranslate2_build/Release`) on `PATH` before running.

Unpatched, the run prints `destroying Translator ...` and then hangs (kill it). With
#2076 applied it prints `SURVIVED: clean shutdown, no deadlock`.
