unset ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH RMW_IMPLEMENTATION
export LD_LIBRARY_PATH=

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib

source .venv/bin/activate