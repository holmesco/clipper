xhost +local:docker
docker run -it --rm\
  --network=host\
  -v $(pwd):/workspace \
  --shm-size=1g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  clipper_dev 