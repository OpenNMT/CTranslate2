#! /bin/bash
set -eux

MKL_VERSION=${1:?MKL version required (e.g. 2025.3.0)}

if command -v apt-get >/dev/null 2>&1; then
    curl -L https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/intel-oneapi.gpg
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y "intel-oneapi-mkl-devel-$MKL_VERSION"
elif command -v dnf >/dev/null 2>&1; then
    dnf config-manager --add-repo https://yum.repos.intel.com/oneapi
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    dnf install -y "intel-oneapi-mkl-devel-$MKL_VERSION"
else
    echo "No supported package manager found (apt-get or dnf)" >&2
    exit 1
fi
