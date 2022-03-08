#!/usr/bin/env bash
set -e

export HTTP_PROXY=http://127.0.0.1:8118
export HTTPS_PROXY=http://127.0.0.1:8118
export http_proxy=http://127.0.0.1:8118
export https_proxy=http://127.0.0.1:8118

BASE_IMG="tensorflow_base:1.15.3-cuda10-cudnn7-py3"
TF_CC_IMG="tensorflow_cc:1.15.3-cuda10-cudnn7-py3"
SPIEL_IMG="my_spiel_cc:tf1.15.3-cuda10-cudnn7-py3"

# build tensorflow_base
docker build . -f Dockerfile.tensorflow_base -t ${BASE_IMG} \
--build-arg USE_PYTHON_3_NOT_2
# --network host \
# --build-arg https_proxy=$HTTP_PROXY --build-arg http_proxy=$HTTP_PROXY \
# --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTP_PROXY


# # build tensorflow_cc
docker build . -f Dockerfile.tensorflow_cc -t ${TF_CC_IMG} \
--build-arg BASE=${BASE_IMG} \
--network host \
--build-arg https_proxy=$HTTP_PROXY --build-arg http_proxy=$HTTP_PROXY \
--build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTP_PROXY


# build my_spiel
docker build . -f Dockerfile.from_tensorflow_cc -t ${SPIEL_IMG} \
--build-arg TF_CC=${TF_CC_IMG} \
--build-arg BASE=${BASE_IMG}
# --network host \
# --build-arg https_proxy=$HTTP_PROXY --build-arg http_proxy=$HTTP_PROXY \
# --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTP_PROXY

