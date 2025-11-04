P1
	T1
xhost +local:root

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $HOME/COMP0246_Labs:/home/ros_ws/src/COMP0246_Labs \
    -v $HOME/.Xauthority:/root/.Xauthority \
    --net=host \
    osrf/ros:humble-desktop-full


cd home/ros_ws
colcon build --symlink-install
sudo apt-get update && sudo apt-get install ros-humble-python-orocos-kdl-vendor ros-humble-orocos-kdl-vendor ros-humble-kdl-parser ros-humble-joint-state-publisher-gui ros-humble-controller-manager ros-humble-joint-trajectory-controller ros-humble-joint-state-broadcaster ros-humble-ros2-control ros-humble-ros2-controllers 

source install/setup.bash

ros2 launch youbot_kinematics bringup.launch.py


	T2
docker exec -it $(docker ps -lq) bash

cd home/ros_ws
source install/setup.bash


ros2 run youbot_kinematics main_student


P2/3/4
	T1
ros2 launch youbot_kinematics bringup.launch.py use_jspg:=false

	T2
ros2 run youbot_kinematics plan_trajectory

	T3
ros2 run youbot_kinematics follow_trajectory
