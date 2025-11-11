# NonPlanarIsaacSim

Quick notes on how to set up the environment, launch the simulator, and extend it with custom tracks.

## Maps Package
- Premade tracks live in the shared archive on [drive](https://drive.google.com/file/d/1UB569kuXtFDBQgYuY2cueMN8XaaIgi8i/view?usp=sharing)  
  Download and unzip it so each bundle sits under `maps/` (e.g., `maps/l_track/l_track.usd`).

## Setup & Launch
0. **Clone the repository and install uv:**
   ```bash
   git clone git@github.com:AhmadAmine998/NonPlanarIsaacSim.git
   pip install uv
   ```
   You will need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to install and run the simulator.
1. **Install dependencies (one-time):**
   ```bash
   bash install_env.sh
   ```
2. **Load the environment for every new shell before running the sim:**
   ```bash
   source setup_env.sh
   ```
3. **Start the ROS-enabled Isaac sim:**
   ```bash
   uv run python ros_sim.py
   ```
   The script picks the map referenced by `MAP_IDX` in `ros_sim.py` and will open the simulator GUI once the assets load.

## Driving the Car
- Run keyboard teleop with `ros2 run teleop_twist_keyboard teleop_twist_keyboard`, which publishes `geometry_msgs/Twist` on `cmd_vel`.
- Any higher-level controller should publish `AckermannDriveStamped` to either `ackermann_cmd` or `/drive` to command speed/steering directly, or publish `Twist` on `cmd_vel`.
- The node republishes odometry on `/fixposition/odometry` and ground-truth drive states on `/ground_truth/state` for downstream consumers.

## Adding New Maps
1. Place each new asset in its own subfolder inside `maps/` (e.g., `maps/new_map/new_map.usd` plus any textures or dependencies) to keep things organized.
2. Supported formats are `.usd`, `.usda`, `.usdz`, `.usdc`, `.obj`, and `.ply`. Meshes are auto-converted to USD on first use and cached in `.converted_maps/`.
3. Add the new file path to `AVAILABLE_MAPS` in `ros_sim.py`, set `MAP_IDX` to point to it, and update `INITIAL_POSITION_PER_MAP` so the rig spawns above the correct location/scale.

That is all that is required; just remember to re-run `source setup_env.sh` in any new terminal before launching the simulator.
