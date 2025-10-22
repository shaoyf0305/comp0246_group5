xhost +local:root

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $HOME/COMP0246_Labs:/home/ros_ws/src/COMP0246_Labs \
    -v $HOME/.Xauthority:/root/.Xauthority \
    --net=host \
    osrf/ros:humble-desktop-full
