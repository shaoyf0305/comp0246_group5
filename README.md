Do everytime when modified

	colcon build --symlink-install
 
 P1 T1-1

	xhost +local:root

	docker run -it --rm \
	    --env="DISPLAY=$DISPLAY" \
	    --env="QT_X11_NO_MITSHM=1" \
	    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	    -v $HOME/COMP0246_Labs:/home/ros_ws/src/COMP0246_Labs \
	    -v $HOME/.Xauthority:/root/.Xauthority \
	    --net=host \
	    osrf/ros:humble-desktop-full

﻿P1 T1-2
 
	cd home/ros_ws
	colcon build --symlink-install

﻿P1 T1-3
	 
	sudo apt-get update && sudo apt-get install ros-humble-python-orocos-kdl-vendor ros-humble-orocos-kdl-vendor ros-humble-kdl-parser ros-humble-joint-state-publisher-gui ros-humble-controller-manager ros-humble-joint-trajectory-controller ros-humble-joint-state-broadcaster ros-humble-ros2-control ros-humble-ros2-controllers 
		
	source install/setup.bash
		
﻿P1 T1-4
 
	ros2 launch youbot_kinematics bringup.launch.py


P1 T2-1

	docker exec -it $(docker ps -lq) bash

P1 T2-2
	
	cd home/ros_ws
	source install/setup.bash

P1 T2-3

	ros2 run youbot_kinematics main_student

P2/3/4 T1

	ros2 launch youbot_kinematics bringup.launch.py use_jspg:=false

P2/3/4 T2

	ros2 run youbot_kinematics plan_trajectory

P2/3/4 T3

	ros2 run youbot_kinematics follow_trajectory
