#! /bin/bash
set -eux

OPENMPI_VERSION=${1:?Open MPI version required (e.g. 4.1.6)}
major_minor=${OPENMPI_VERSION%.*}

workdir=$(mktemp -d)
trap "rm -rf $workdir" EXIT

curl -L https://download.open-mpi.org/release/open-mpi/v${major_minor}/openmpi-${OPENMPI_VERSION}.tar.bz2 | tar xj -C "$workdir"
cd "$workdir/openmpi-${OPENMPI_VERSION}"
./configure
make -j"$(nproc)" install
