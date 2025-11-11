from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import asyncio
import math, os, threading, time
from pathlib import Path
import numpy as np

# Core API (5.1)
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

# Optional UI
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

# ROS 2, imported in-process (use Isaac’s Python 3.11 rclpy)
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Viewport camera binding
import omni.usd
from pxr import Usd, Sdf, UsdGeom, Gf, UsdLux
from omni.kit.viewport.utility import get_active_viewport

ODOM_TOPIC = "/fixposition/odometry"
GT_STATE_TOPIC = "/ground_truth/state"
MAP_FRAME_ID = "map"
BASE_FRAME_ID = "base_link"
STEER_RATE_LIMIT_RAD = math.radians(180.0)
EPS = 1e-6

USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
MESH_EXTENSIONS = {".obj", ".ply"}
CONVERTED_MAP_DIR = Path(__file__).resolve().parent / ".converted_maps"

MAP_IDX = 2  # change to select different map
AVAILABLE_MAPS = [
    None,
    "/home/ahmad/isaacsim/environments/nonPlanarSim/maps/motorcross/motorcross_track_demo.usd",
    "/home/ahmad/isaacsim/environments/nonPlanarSim/maps/l_track/l_track.usd",
    "/home/ahmad/isaacsim/environments/nonPlanarSim/maps/oval/oval.usd",
    "/home/ahmad/isaacsim/environments/nonPlanarSim/maps/kidney/kidney.usd",
]
INITIAL_POSITION_PER_MAP = [
    Gf.Vec3d(0.0, 0.0, 0.0),            # default ground plane
    Gf.Vec3d(0.0, 0.0, 45.7),          # motorcross track
    Gf.Vec3d(0.0, 0.0, 0.5),          # l_track track
    Gf.Vec3d(0.0, 0.0, 0.5),          # oval track
    Gf.Vec3d(0.0, 0.0, 0.5),          # kidney track
]
CHOSEN_MAP = AVAILABLE_MAPS[MAP_IDX]
INITIAL_POSITION = INITIAL_POSITION_PER_MAP[MAP_IDX]

LEATHERBACK_RIG_PATH = "/World/LeatherbackRig"
LEATHERBACK_ROBOT_PATH = f"{LEATHERBACK_RIG_PATH}/Robot"
if "motorcross" in (CHOSEN_MAP or ""):
    RIG_SCALE = Gf.Vec3f(100.0, 100.0, 100.0)
else:
    RIG_SCALE = Gf.Vec3f(1.0, 1.0, 1.0)

def _resolve_local_asset(path_str: str) -> Path:
    resolved = Path(os.path.expandvars(path_str)).expanduser()
    if not resolved.is_absolute():
        resolved = (Path(__file__).resolve().parent / resolved).resolve()
    return resolved

def _run_async_conversion(coro_factory):
    """Run an async converter coroutine even if an event loop is already running."""
    try:
        asyncio.run(coro_factory())
        return
    except RuntimeError as exc:
        if "asyncio.run()" not in str(exc):
            raise
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro_factory())
    finally:
        loop.close()

def convert_mesh_to_usd(mesh_path: Path) -> str:
    CONVERTED_MAP_DIR.mkdir(parents=True, exist_ok=True)
    usd_dir = CONVERTED_MAP_DIR / mesh_path.stem
    usd_dir.mkdir(parents=True, exist_ok=True)
    usd_path = usd_dir / f"{mesh_path.stem}.usd"
    if usd_path.exists():
        return str(usd_path)

    async def _convert_async():
        enable_extension("omni.kit.asset_converter")
        import omni.kit.asset_converter

        ctx = omni.kit.asset_converter.AssetConverterContext()
        ctx.ignore_materials = False
        ctx.ignore_animations = True
        ctx.ignore_camera = True
        ctx.ignore_light = True
        ctx.merge_all_meshes = True
        ctx.use_meter_as_world_unit = True
        ctx.baking_scales = True
        ctx.use_double_precision_to_usd_transform_op = True

        instance = omni.kit.asset_converter.get_instance()
        task = instance.create_converter_task(str(mesh_path), str(usd_path), None, ctx)
        success = await task.wait_until_finished()
        if not success:
            raise RuntimeError(f"Failed to convert {mesh_path} to USD: {task.get_error_message()}")

    _run_async_conversion(_convert_async)
    print(f"[load_map] Converted mesh {mesh_path} -> {usd_path}")
    return str(usd_path)

def load_map(world: World, chosen_map: str | None, prim_path: str = "/World/Track"):
    """Load the desired map asset into the stage, defaulting to ground plane."""
    stage = omni.usd.get_context().get_stage()
    # Add dome light
    distantLight = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    distantLight.CreateIntensityAttr(5000.0)
    if not chosen_map:
        world.scene.add_default_ground_plane()
        print("[load_map] Using default ground plane.")
        return None

    if "://" in chosen_map:
        asset_ext = os.path.splitext(chosen_map)[1].lower()
        if asset_ext in MESH_EXTENSIONS:
            raise ValueError("Mesh conversion is only supported for local files.")
        usd_asset_path = chosen_map
    else:
        local_path = _resolve_local_asset(chosen_map)
        print(f"[load_map] Loading map from {local_path}")
        if not local_path.exists():
            raise FileNotFoundError(f"Map asset not found: {local_path}")
        ext = local_path.suffix.lower()
        if ext in MESH_EXTENSIONS:
            usd_asset_path = convert_mesh_to_usd(local_path)
        elif ext in USD_EXTENSIONS:
            usd_asset_path = str(local_path)
        else:
            raise ValueError(f"Unsupported map format '{ext}' for {local_path}")

    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))
    stage_utils.add_reference_to_stage(usd_asset_path, prim_path)
    print(f"[load_map] Loaded map from {usd_asset_path} into {prim_path}")
    return prim_path

def find_camera_under(root_path: str, name_hint: str = "Camera_Chase") -> str | None:
    """Return prim path of a Camera under root_path. Prefer *name_hint* if present."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    fallback = None
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Camera":
            if name_hint and name_hint in prim.GetName():
                return prim.GetPath().pathString
            if fallback is None:
                fallback = prim.GetPath().pathString
    return fallback

def set_active_camera(cam_path: str):
    vp = get_active_viewport()
    if not vp:
        raise RuntimeError("No active viewport")
    vp.camera_path = cam_path  # make it the default view

def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array([x, y, z, w], dtype=float)

def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def world_to_body_xy(linear_world: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx, vy = linear_world[0], linear_world[1]
    return np.array([c * vx + s * vy, -s * vx + c * vy], dtype=float)

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

# ---------- ROS command state ----------
class CommandState:
    def __init__(self):
        self.steering = 0.0            # rad
        self.steering_velocity = 0.0   # rad/s
        self.speed_mps = 0.0           # m/s
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.latest_state = None

    def set_state(self, state: dict):
        with self.state_lock:
            self.latest_state = state

def make_ros_thread(cmd: CommandState, wheel_base_m: float, max_steer_rad: float):
    def twist_to_ackermann(twist: Twist):
        v = float(twist.linear.x)
        theta = math.radians(float(twist.angular.z))
        return v, max(-max_steer_rad, min(max_steer_rad, theta))

    def spin():
        rclpy.init()
        node = rclpy.create_node("leatherback_ackermann_driver")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        odom_pub = node.create_publisher(Odometry, ODOM_TOPIC, qos)
        state_pub = node.create_publisher(AckermannDriveStamped, GT_STATE_TOPIC, qos)

        def ack_cb(msg: AckermannDriveStamped):
            with cmd.lock:
                cmd.speed_mps = float(msg.drive.speed)
                cmd.steering = max(-max_steer_rad, min(max_steer_rad, float(msg.drive.steering_angle)))
                cmd.steering_velocity = float(msg.drive.steering_angle_velocity)

        def drive_cb(msg: AckermannDriveStamped):
            with cmd.lock:
                cmd.speed_mps = float(msg.drive.speed)
                cmd.steering = max(-max_steer_rad, min(max_steer_rad, float(msg.drive.steering_angle)))
                cmd.steering_velocity = float(msg.drive.steering_angle_velocity)

        def twist_cb(msg: Twist):
            v, theta = twist_to_ackermann(msg)
            with cmd.lock:
                cmd.speed_mps = v
                cmd.steering = theta
                cmd.steering_velocity = 0.0

        def publish_state():
            with cmd.state_lock:
                state = cmd.latest_state
            if not state:
                return
            stamp = node.get_clock().now().to_msg()
            odom_msg = Odometry()
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = state.get("frame_id", MAP_FRAME_ID)
            odom_msg.child_frame_id = state.get("child_frame_id", BASE_FRAME_ID)
            pos = state["position"]
            quat_xyzw = state["orientation_xyzw"]
            lin = state["linear_velocity"]
            ang = state["angular_velocity"]
            odom_msg.pose.pose.position.x = float(pos[0])
            odom_msg.pose.pose.position.y = float(pos[1])
            odom_msg.pose.pose.position.z = float(pos[2])
            odom_msg.pose.pose.orientation.x = float(quat_xyzw[0])
            odom_msg.pose.pose.orientation.y = float(quat_xyzw[1])
            odom_msg.pose.pose.orientation.z = float(quat_xyzw[2])
            odom_msg.pose.pose.orientation.w = float(quat_xyzw[3])
            odom_msg.twist.twist.linear.x = float(lin[0])
            odom_msg.twist.twist.linear.y = float(lin[1])
            odom_msg.twist.twist.linear.z = float(lin[2])
            odom_msg.twist.twist.angular.x = float(ang[0])
            odom_msg.twist.twist.angular.y = float(ang[1])
            odom_msg.twist.twist.angular.z = float(ang[2])
            odom_pub.publish(odom_msg)

            state_msg = AckermannDriveStamped()
            state_msg.header = odom_msg.header
            state_msg.drive.steering_angle = float(state["steering"])
            state_msg.drive.steering_angle_velocity = float(state["steering_rate"])
            state_msg.drive.speed = float(state["speed"])
            state_pub.publish(state_msg)

        node.create_subscription(AckermannDriveStamped, "ackermann_cmd", ack_cb, qos)
        node.create_subscription(AckermannDriveStamped, "/drive", drive_cb, qos)
        node.create_subscription(Twist, "cmd_vel", twist_cb, qos)
        node.create_timer(1.0 / 60.0, publish_state)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    return th

# ---------- shared state ----------
class _S:
    world = None
    robot = None
    controller = None
    steering_idx = None
    wheel_idx = None
    cmd = None
    WHEEL_R = 0.25
    max_steer = math.radians(24.0)
    delta = 0.0
    delta_rate = 0.0
    physics_dt = None
    prev_pose = None
    prev_theta = None
    prev_time = None
S = _S()

# ---------- async setup ----------
async def setup():
    if World.instance():
        World.instance().clear_instance()
    S.world = World()
    await S.world.initialize_simulation_context_async()

    TRACK_ROOT = "/World/Track"
    load_map(S.world, CHOSEN_MAP, prim_path=TRACK_ROOT)

    # Leatherback rig + reference, following test_usd_create pattern
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim(LEATHERBACK_RIG_PATH, "Xform")
    assets_root = get_assets_root_path()
    leatherback_usd = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    stage_utils.add_reference_to_stage(leatherback_usd, LEATHERBACK_ROBOT_PATH)
    rig_prim = stage.GetPrimAtPath(LEATHERBACK_RIG_PATH)
    xf = UsdGeom.XformCommonAPI(rig_prim)
    xf.SetScale(RIG_SCALE)
    xf.SetTranslate(INITIAL_POSITION)

    # Register Robot wrapper at the referenced prim
    S.robot = S.world.scene.add(Robot(prim_path=LEATHERBACK_ROBOT_PATH, name="leatherback"))
    print(f"[spawn] Leatherback rig scaled to {tuple(RIG_SCALE)} at translate {tuple(INITIAL_POSITION)}")

    await S.world.reset_async()
    await S.world.play_async()

    # viewport -> use built-in chase camera under Leatherback
    cam_path = find_camera_under(LEATHERBACK_ROBOT_PATH, "Camera_Chase")
    if cam_path:
        set_active_camera(cam_path)
        print(f"Active camera: {cam_path}")
    else:
        print("No Camera_Chase found; using default viewport camera")

    # joints per Leatherback tutorial
    steering_joint_names = ["Knuckle__Upright__Front_Left", "Knuckle__Upright__Front_Right"]
    wheel_joint_names    = ["Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right",
                            "Wheel__Upright__Rear_Left", "Wheel__Upright__Rear_Right"]
    S.steering_idx = [S.robot.get_dof_index(n) for n in steering_joint_names]
    S.wheel_idx    = [S.robot.get_dof_index(n) for n in wheel_joint_names]

    # controller - reference params: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/mobile_robot_controllers.html
    WHEEL_BASE = 1.65
    TRACK_W    = 1.25
    S.WHEEL_R  = 0.25
    S.controller = AckermannController(
        name="ackermann",
        wheel_base=WHEEL_BASE,
        track_width=TRACK_W,
        front_wheel_radius=S.WHEEL_R,
        back_wheel_radius=S.WHEEL_R,
    )

    # ROS
    MAX_STEER = math.radians(24.0)
    S.max_steer = MAX_STEER
    S.physics_dt = float(S.world.get_physics_dt())
    S.cmd = CommandState()
    make_ros_thread(S.cmd, WHEEL_BASE, MAX_STEER)

from isaacsim.simulation_app import SimulationApp as _SimApp
simulation_app.run_coroutine(setup(), run_until_complete=True)
S.prev_time = time.perf_counter()
S.prev_pose = None
S.prev_theta = None

# ---------- frame loop ----------
while simulation_app.is_running():
    now = time.perf_counter()
    wall_dt = None
    if S.prev_time is not None:
        wall_dt = max(now - S.prev_time, EPS)
    S.prev_time = now
    dt = S.physics_dt if S.physics_dt else (wall_dt or (1.0 / 600.0))
    dt = max(dt, EPS)

    with S.cmd.lock:
        speed_mps = S.cmd.speed_mps
        steer_target = S.cmd.steering
        steer_vel_cmd = S.cmd.steering_velocity

    prev_delta = S.delta
    if abs(steer_vel_cmd) > EPS:
        desired_rate = clamp(steer_vel_cmd, -STEER_RATE_LIMIT_RAD, STEER_RATE_LIMIT_RAD)
    else:
        rate_from_error = (steer_target - S.delta) / dt
        desired_rate = clamp(rate_from_error, -STEER_RATE_LIMIT_RAD, STEER_RATE_LIMIT_RAD)
    S.delta = clamp(S.delta + desired_rate * dt, -S.max_steer, S.max_steer)
    S.delta_rate = (S.delta - prev_delta) / dt

    wheel_omega = speed_mps / max(S.WHEEL_R, 1e-6)

    # Ackermann: [steer, steer_vel, forward_wheel_omega, accel, dt]
    actions = S.controller.forward([S.delta, S.delta_rate, wheel_omega, 0.0, dt])

    joint_pos = np.zeros(S.robot.num_dof)
    joint_vel = np.zeros(S.robot.num_dof)

    # steering positions
    jp = actions.joint_positions
    joint_pos[S.steering_idx[0]] = jp[0]
    joint_pos[S.steering_idx[1]] = jp[1]

    # wheel angular velocities
    jv = actions.joint_velocities
    joint_vel[S.wheel_idx[0]] = jv[0]
    joint_vel[S.wheel_idx[1]] = jv[1]
    joint_vel[S.wheel_idx[2]] = jv[2]
    joint_vel[S.wheel_idx[3]] = jv[3]

    S.robot.apply_action(ArticulationAction(joint_positions=joint_pos, joint_velocities=joint_vel))
    simulation_app.update()

    # Ground-truth pose and twist
    pos, quat_wxyz = S.robot.get_world_pose()
    pos = np.asarray(pos, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    yaw = yaw_from_quat_wxyz(quat_wxyz)

    lv = S.robot.get_linear_velocity()
    lin_vel_world = np.asarray(lv, dtype=float)
    av = S.robot.get_angular_velocity()
    ang_vel_world = np.asarray(av, dtype=float) if av is not None else None

    body_xy = world_to_body_xy(lin_vel_world, yaw)
    lin_body = np.array([body_xy[0], body_xy[1], lin_vel_world[2]], dtype=float)
    ang_body = np.array([0.0, 0.0, ang_vel_world[2]], dtype=float)
    speed = math.hypot(body_xy[0], body_xy[1])

    state_payload = {
        "frame_id": MAP_FRAME_ID,
        "child_frame_id": BASE_FRAME_ID,
        "position": pos.tolist(),
        "orientation_xyzw": quat_xyzw.tolist(),
        "linear_velocity": lin_body.tolist(),
        "angular_velocity": ang_body.tolist(),
        "steering": S.delta,
        "steering_rate": S.delta_rate,
        "speed": speed,
    }
    S.cmd.set_state(state_payload)

simulation_app.close()
