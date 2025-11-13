from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import asyncio
import math
import os
from pathlib import Path

import numpy as np

# Core API (5.1)
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

import omni.usd
from omni.kit.viewport.utility import get_active_viewport

from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf

# --------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------

ODOM_TOPIC = "/fixposition/odometry"
GT_STATE_TOPIC = "/ground_truth/state"
MAP_FRAME_ID = "map"
BASE_FRAME_ID = "base_link"
STEER_RATE_LIMIT_RAD = math.radians(360.0)
EPS = 1e-6

USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
MESH_EXTENSIONS = {".obj", ".ply"}
CONVERTED_MAP_DIR = Path(__file__).resolve().parent / ".converted_maps"

CHOSEN_MAP = None  # default ground plane
INITIAL_POSITION = Gf.Vec3d(0.0, 0.0, 0.0)

LEATHERBACK_RIG_PATH = "/World/LeatherbackRig"
LEATHERBACK_ROBOT_PATH = f"{LEATHERBACK_RIG_PATH}/Robot"

LEATHERBACK_ROOT = LEATHERBACK_ROBOT_PATH
CHASSIS_PRIM_PATH = f"{LEATHERBACK_ROOT}/Rigid_Bodies/Chassis"
RIGID_BODIES_ROOT_PATH = f"{LEATHERBACK_ROOT}/Rigid_Bodies"

# From your screenshot
WHEEL_PRIMS = {
    "rear_right":  f"{LEATHERBACK_ROOT}/Rigid_Bodies/Wheel_Rear_Right",
    "rear_left":   f"{LEATHERBACK_ROOT}/Rigid_Bodies/Wheel_Rear_Left",
    "front_right": f"{LEATHERBACK_ROOT}/Rigid_Bodies/Wheel_Front_Right",
    "front_left":  f"{LEATHERBACK_ROOT}/Rigid_Bodies/Wheel_Front_Left",
}

RIG_SCALE = Gf.Vec3f(1.0, 1.0, 1.0)


# --------------------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------------------

def _resolve_local_asset(path_str: str) -> Path:
    resolved = Path(os.path.expandvars(path_str)).expanduser()
    if not resolved.is_absolute():
        resolved = (Path(__file__).resolve().parent / resolved).resolve()
    return resolved


def convert_mesh_to_usd(local_path: Path) -> str:
    raise NotImplementedError(
        f"Mesh-to-USD conversion is not implemented in this script. Got: {local_path}"
    )


def _run_async_conversion(coro_factory):
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


def load_map(world: World, chosen_map: str | None, prim_path: str = "/World/Track"):
    stage = omni.usd.get_context().get_stage()

    # simple dome light
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(5000.0)

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
    vp.camera_path = cam_path


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


# --------------------------------------------------------------------------------------
# ASYNC SETUP
# --------------------------------------------------------------------------------------

async def setup():
    if World.instance():
        World.instance().clear_instance()

    S.world = World()
    await S.world.initialize_simulation_context_async()

    TRACK_ROOT = "/World/Track"
    load_map(S.world, CHOSEN_MAP, prim_path=TRACK_ROOT)

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim(LEATHERBACK_RIG_PATH, "Xform")
    assets_root = get_assets_root_path()
    leatherback_usd = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    stage_utils.add_reference_to_stage(leatherback_usd, LEATHERBACK_ROBOT_PATH)

    rig_prim = stage.GetPrimAtPath(LEATHERBACK_RIG_PATH)
    xf = UsdGeom.XformCommonAPI(rig_prim)
    xf.SetScale(RIG_SCALE)
    xf.SetTranslate(INITIAL_POSITION)

    S.robot = S.world.scene.add(
        Robot(prim_path=LEATHERBACK_ROBOT_PATH, name="leatherback")
    )
    print(f"[setup] Leatherback rig scaled to {tuple(RIG_SCALE)} at translate {tuple(INITIAL_POSITION)}")

    await S.world.reset_async()
    await S.world.play_async()

    cam_path = find_camera_under(LEATHERBACK_ROBOT_PATH, "Camera_Chase")
    if cam_path:
        set_active_camera(cam_path)
        print(f"[setup] Active camera: {cam_path}")
    else:
        print("[setup] No Camera_Chase found; using default viewport camera")

    steering_joint_names = ["Knuckle__Upright__Front_Left", "Knuckle__Upright__Front_Right"]
    wheel_joint_names = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Left",
        "Wheel__Upright__Rear_Right",
    ]
    S.steering_idx = [S.robot.get_dof_index(n) for n in steering_joint_names]
    S.wheel_idx = [S.robot.get_dof_index(n) for n in wheel_joint_names]

    WHEEL_BASE = 1.65
    TRACK_W = 1.25
    S.WHEEL_R = 0.25
    S.controller = AckermannController(
        name="ackermann",
        wheel_base=WHEEL_BASE,
        track_width=TRACK_W,
        front_wheel_radius=S.WHEEL_R,
        back_wheel_radius=S.WHEEL_R,
    )

    S.max_steer = math.radians(24.0)
    print("[setup] World initialized and Leatherback spawned.")


# --------------------------------------------------------------------------------------
# PARAMETER EXTRACTION HELPERS
# --------------------------------------------------------------------------------------

def get_stage():
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No stage loaded.")
    return stage


def get_total_mass_under(root_prim):
    total = 0.0
    for prim in Usd.PrimRange(root_prim):
        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api:
            continue
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr.IsValid():
            continue
        m = mass_attr.Get()
        if m is not None:
            total += float(m)
    return total


def get_bbox_cache(stage):
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
        ignoreVisibility=False,
    )


def get_center_and_size(stage, bbox_cache, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedBox()
    mn = rng.GetMin()
    mx = rng.GetMax()
    size = mx - mn
    center = 0.5 * (mn + mx)
    return center, size


def get_wheel_radius_from_size(size):
    return 0.5 * max(size[1], size[2])


def _find_collision_prim_recursive(prim: Usd.Prim):
    """Depth-first search for a prim with CollisionAPI under `prim`."""
    coll_api = UsdPhysics.CollisionAPI(prim)
    if coll_api:
        return prim
    for child in prim.GetChildren():
        found = _find_collision_prim_recursive(child)
        if found:
            return found
    return None


def get_wheel_material_friction(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[warn] Wheel prim not found for friction: {prim_path}")
        return None

    coll_prim = _find_collision_prim_recursive(prim)
    if coll_prim is None:
        print(f"[warn] No CollisionAPI found under wheel prim subtree: {prim_path}")
        target_prim = prim
    else:
        target_prim = coll_prim

    # Materials are bound via UsdShade.MaterialBindingAPI
    bind_api = UsdShade.MaterialBindingAPI(target_prim)
    bound_mat, _ = bind_api.ComputeBoundMaterial("physics")
    if not bound_mat:
        bound_mat, _ = bind_api.ComputeBoundMaterial()

    if not bound_mat:
        print(f"[warn] No bound physics material on: {target_prim.GetPath().pathString}")
        return None

    mat_prim = bound_mat.GetPrim()
    mat_api = UsdPhysics.MaterialAPI(mat_prim)
    if not mat_api:
        print(f"[warn] Material prim has no UsdPhysics.MaterialAPI: {mat_prim.GetPath().pathString}")
        return None

    mu_s = mat_api.GetStaticFrictionAttr().Get()
    mu_d = mat_api.GetDynamicFrictionAttr().Get()
    rest = mat_api.GetRestitutionAttr().Get()

    return {
        "material_prim": mat_prim.GetPath().pathString,
        "mu_static": float(mu_s) if mu_s is not None else None,
        "mu_dynamic": float(mu_d) if mu_d is not None else None,
        "restitution": float(rest) if rest is not None else None,
    }


def get_chassis_mass_com_inertia(stage, prim_path):
    """Try to read MassAPI on chassis; may return None if not authored."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[warn] Chassis prim not found: {prim_path}")
        return None

    mass_api = UsdPhysics.MassAPI(prim)
    if not mass_api:
        print(f"[warn] No MassAPI on chassis prim: {prim_path}")
        return None

    m_attr = mass_api.GetMassAttr()
    com_attr = mass_api.GetCenterOfMassAttr()
    diagI_attr = mass_api.GetDiagonalInertiaAttr()

    mass = m_attr.Get() if m_attr.IsValid() else None
    com_local = com_attr.Get() if com_attr.IsValid() else None
    diagI = diagI_attr.Get() if diagI_attr.IsValid() else None

    xform = UsdGeom.Xformable(prim)
    M_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    com_world = None
    if com_local is not None:
        v = Gf.Vec3d(*com_local)
        com_world = M_world.TransformPoint(v)

    return {
        "mass": float(mass) if mass is not None else None,
        "com_local": tuple(com_local) if com_local is not None else None,
        "com_world": (com_world[0], com_world[1], com_world[2]) if com_world is not None else None,
        "diag_inertia": tuple(diagI) if diagI is not None else None,
    }


def get_articulation_com_approx(stage, bbox_cache, rigid_bodies_root_path: str):
    """
    Approximate articulation CoM by volume-weighted average of all rigid-body
    Xforms under Rigid_Bodies (ignores Joints, materials, etc.).
    """
    root = stage.GetPrimAtPath(rigid_bodies_root_path)
    if not root.IsValid():
        print(f"[warn] Rigid_Bodies root not found: {rigid_bodies_root_path}")
        return None

    num = Gf.Vec3d(0.0, 0.0, 0.0)
    denom = 0.0

    for body in root.GetChildren():
        name = body.GetName()
        if name in ("Joints", "Rubber_Asphalt"):
            continue
        if not body.IsA(UsdGeom.Xformable):
            continue

        bbox = bbox_cache.ComputeWorldBound(body)
        rng = bbox.ComputeAlignedBox()
        mn = rng.GetMin()
        mx = rng.GetMax()
        center = 0.5 * (mn + mx)
        size = mx - mn
        vol = abs(size[0] * size[1] * size[2])
        if vol <= 0.0:
            continue

        num += center * vol
        denom += vol

    if denom <= 0.0:
        print("[warn] Could not compute approximate CoM (zero total volume).")
        return None

    com = num / denom
    return {
        "com_world": (com[0], com[1], com[2]),
        "geom_weight": denom,
    }


# --------------------------------------------------------------------------------------
# MAIN PARAMETER EXTRACTION
# --------------------------------------------------------------------------------------

def compute_leatherback_parameters():
    stage = get_stage()
    root_prim = stage.GetPrimAtPath(LEATHERBACK_ROOT)
    if not root_prim.IsValid():
        raise RuntimeError(f"Leatherback root prim not found at {LEATHERBACK_ROOT}")

    print(f"\n== Leatherback parameters under {LEATHERBACK_ROOT} ==")

    total_mass = get_total_mass_under(root_prim)
    print(f"Total authored mass (kg) [may be 0 if masses are implicit]: {total_mass}")

    bbox_cache = get_bbox_cache(stage)

    # Try chassis MassAPI (may be None), and geometric CoM approximation
    chassis_props = get_chassis_mass_com_inertia(stage, CHASSIS_PRIM_PATH)
    print("\nChassis mass/CoM/inertia from MassAPI:", chassis_props)

    artic_com = get_articulation_com_approx(stage, bbox_cache, RIGID_BODIES_ROOT_PATH)
    print("Approx articulation CoM from geometry:", artic_com)

    # --- Wheel geometry: centers, radii, wheelbase, track ---
    wheel_info = {}
    for name, path in WHEEL_PRIMS.items():
        try:
            center, size = get_center_and_size(stage, bbox_cache, path)
        except RuntimeError as e:
            print(f"[warn] {e}")
            continue
        radius = get_wheel_radius_from_size(size)
        wheel_info[name] = {
            "center": (center[0], center[1], center[2]),
            "size": (size[0], size[1], size[2]),
            "radius": float(radius),
        }

    print("\nWheel geometry:")
    for name, info in wheel_info.items():
        print(f"  {name}: center={info['center']}, radius={info['radius']}, size={info['size']}")

    # Wheelbase / track
    if all(k in wheel_info for k in ("front_left", "front_right", "rear_left", "rear_right")):
        wheelbase = abs(
            wheel_info["front_left"]["center"][0] - wheel_info["rear_left"]["center"][0]
        )
        front_track = abs(
            wheel_info["front_left"]["center"][1] - wheel_info["front_right"]["center"][1]
        )
        rear_track = abs(
            wheel_info["rear_left"]["center"][1] - wheel_info["rear_right"]["center"][1]
        )
        print(f"\nWheelbase (m): {wheelbase}")
        print(f"Front track (m): {front_track}")
        print(f"Rear track (m):  {rear_track}")
    else:
        print("\n[warn] Incomplete wheel_info; cannot compute wheelbase/track.")

    # Choose CoM source: prefer geometric articulation CoM, fall back to chassis if needed
    com_world = None
    if artic_com and artic_com.get("com_world"):
        com_world = artic_com["com_world"]
        com_source = "articulation geometric CoM"
    elif chassis_props and chassis_props.get("com_world"):
        com_world = chassis_props["com_world"]
        com_source = "chassis MassAPI CoM"
    else:
        com_source = None

    if com_world and all(
        k in wheel_info for k in ("front_left", "front_right", "rear_left", "rear_right")
    ):
        com_x, com_y, com_z = com_world
        x_front_axle = 0.5 * (
            wheel_info["front_left"]["center"][0] + wheel_info["front_right"]["center"][0]
        )
        x_rear_axle = 0.5 * (
            wheel_info["rear_left"]["center"][0] + wheel_info["rear_right"]["center"][0]
        )

        L = abs(x_front_axle - x_rear_axle)
        Lf = x_front_axle - com_x
        Lr = com_x - x_rear_axle
        H = com_z  # CoM height

        print(f"\nBicycle-model geometry using {com_source}:")
        print(f"  L   (wheelbase)      ≈ {L}")
        print(f"  Lf  (CoM -> front axle) ≈ {Lf}")
        print(f"  Lr  (rear axle -> CoM)  ≈ {Lr}")
        print(f"  H   (CoM height)        ≈ {H}")
    else:
        print("\n[warn] Could not compute Lf/Lr/H (missing CoM or wheel data).")

    print("\nWheel material friction:")
    for name, path in WHEEL_PRIMS.items():
        mat = get_wheel_material_friction(stage, path)
        print(f"  {name}: {mat}")

    print("\n[done] Leatherback parameter extraction complete.")


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    simulation_app.run_coroutine(setup(), run_until_complete=True)
    compute_leatherback_parameters()

    # keep sim running so you can inspect in GUI
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()
