#!/usr/bin/env bash

IMAGE_NAME="cnn_to_tree:latest"

USER_NAME=$( whoami )
CONTAINER_NAME=${USER_NAME}-cnn_to_tree

docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}

nvidia-docker run -it -d --net=host --ipc=host \
-v ${HOME}/cnn_to_tree:/cnn_to_tree \
-v /mnt/SSD/${USER_NAME}/cnn_to_tree:/data \
-w /cnn_to_tree --name ${CONTAINER_NAME} ${IMAGE_NAME} bash
