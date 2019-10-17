#!/bin/bash
#
# Build latest:
# ./build_all.sh
#
# Build and push version X.Y.Z:
# ./build_all.sh X.Y.Z 1

set -e

VERSION=${1:-latest}
PUSH=${2:-0}
IMAGE=opennmt/ctranslate2

build()
{
    PLAT=$1
    LATEST=$IMAGE:latest-$PLAT
    TAGGED=$IMAGE:$VERSION-$PLAT
    docker build --no-cache -t $LATEST -f docker/Dockerfile.$PLAT .
    if [ $PUSH -eq 1 ]; then
        docker push $LATEST
    fi
    if [ "$TAGGED" != "$LATEST" ]; then
        docker tag $LATEST $TAGGED
        if [ $PUSH -eq 1 ]; then
            docker push $TAGGED
        fi
    fi
}

build ubuntu18
build ubuntu18-gpu
build centos7
build centos7-gpu
