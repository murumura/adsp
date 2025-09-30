#!/bin/bash
# Expose the X server on the host.
container_id=$(docker ps -aq --filter name=pysim_container)
if [ ! -z "$container_id" ]
then
    docker rm -f ${container_id}
fi
sudo xhost +local:root
# --rm: Make the container ephemeral (delete on exit).
# -it: Interactive TTY.
# --gpus all: Expose all GPUs to the container.
# access jupyter notebook via http://localhost:1234/
if [[ $(uname -m) == 'arm64' ]]; then
  docker run \
  --rm \
  -it \
  -v $(pwd):/pysim \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 1234:8888 -p 6006:6006 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  --name pysim_container pysim-dev
else
  docker run \
  --rm \
  -it \
  --gpus all \
  -v $(pwd):/pysim \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 1234:8888 -p 6006:6006 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  --name pysim_container pysim-dev
fi

