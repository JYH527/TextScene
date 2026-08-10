import json
import math
import time
from pathlib import Path
from typing import Tuple

import carla


JSON_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\driving_paths_true_lane_points.json")
TARGET_MAP_KEY = "Town01_T-junction_01"
TARGET_CHAIN_INDEX = 3
TARGET_POINT_SPACING = 0.8

# CARLA 服务器地址
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

# 坐标轴修正
FLIP_Y = True

# 跟随参数
TARGET_SPEED_MPS = 6.0
POINT_HIT_DISTANCE = 1.2
LOOKAHEAD_DISTANCE = 6.0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_points_for_map():
    data = load_json(JSON_PATH)
    if TARGET_MAP_KEY not in data:
        raise KeyError(f"找不到 {TARGET_MAP_KEY}")

    lane_chains = data[TARGET_MAP_KEY]
    if not lane_chains:
        raise ValueError(f"{TARGET_MAP_KEY} 没有轨迹数据")

    if TARGET_CHAIN_INDEX < 0 or TARGET_CHAIN_INDEX >= len(lane_chains):
        raise IndexError(
            f"TARGET_CHAIN_INDEX={TARGET_CHAIN_INDEX} 超出范围，当前共有 {len(lane_chains)} 条轨迹"
        )

    chain = lane_chains[TARGET_CHAIN_INDEX]
    ordered_points = chain.get("ordered_points")
    if not ordered_points:
        points = chain["points"]
        ordered_points = [
            {"name": "from_start", "point": points["from_start"]},
            {"name": "from_end", "point": points["from_end"]},
            {"name": "via_start", "point": points["via_start"]},
            {"name": "via_end", "point": points["via_end"]},
            {"name": "to_start", "point": points["to_start"]},
            {"name": "to_end", "point": points["to_end"]},
        ]
    return chain, ordered_points


def to_carla_location(point_dict, z_offset=0.5):
    y = float(point_dict["y"])
    if FLIP_Y:
        y = -y
    return carla.Location(
        x=float(point_dict["x"]),
        y=y,
        z=float(point_dict.get("z", 0.0)) + z_offset,
    )


def distance_2d(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def get_yaw_to_target(current: carla.Location, target: carla.Location) -> float:
    return math.degrees(math.atan2(target.y - current.y, target.x - current.x))


def lerp_point(a: dict, b: dict, t: float) -> dict:
    return {
        "x": float(a["x"]) + (float(b["x"]) - float(a["x"])) * t,
        "y": float(a["y"]) + (float(b["y"]) - float(a["y"])) * t,
        "z": float(a.get("z", 0.0)) + (float(b.get("z", 0.0)) - float(a.get("z", 0.0))) * t,
    }


def cubic_hermite(p0: dict, p1: dict, m0: Tuple[float, float, float], m1: Tuple[float, float, float], t: float) -> dict:
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return {
        "x": h00 * p0["x"] + h10 * m0[0] + h01 * p1["x"] + h11 * m1[0],
        "y": h00 * p0["y"] + h10 * m0[1] + h01 * p1["y"] + h11 * m1[1],
        "z": h00 * p0.get("z", 0.0) + h10 * m0[2] + h01 * p1.get("z", 0.0) + h11 * m1[2],
    }


def build_smooth_trajectory(ordered_points):
    pts = [(item["name"], item["point"]) for item in ordered_points]
    if len(pts) < 2:
        return pts

    # Catmull-Rom style tangents for smooth transitions
    tangents = []
    for i in range(len(pts)):
        prev_pt = pts[max(i - 1, 0)][1]
        next_pt = pts[min(i + 1, len(pts) - 1)][1]
        tangents.append((
            (float(next_pt["x"]) - float(prev_pt["x"])) * 0.5,
            (float(next_pt["y"]) - float(prev_pt["y"])) * 0.5,
            (float(next_pt.get("z", 0.0)) - float(prev_pt.get("z", 0.0))) * 0.5,
        ))

    smoothed = []
    for idx in range(len(pts) - 1):
        start_name, start_pt = pts[idx]
        end_name, end_pt = pts[idx + 1]
        seg_len = distance_2d(to_carla_location(start_pt, z_offset=0.0), to_carla_location(end_pt, z_offset=0.0))
        steps = max(2, int(math.ceil(seg_len / TARGET_POINT_SPACING)))
        m0 = tangents[idx]
        m1 = tangents[idx + 1]
        for step in range(steps):
            t = step / float(steps)
            p = cubic_hermite(start_pt, end_pt, m0, m1, t)
            smoothed.append((f"{start_name}->{end_name}:{step}", p))
    smoothed.append((pts[-1][0], pts[-1][1]))
    return smoothed


def apply_control_to_target(vehicle, target_location, target_speed_mps=8.0):
    transform = vehicle.get_transform()
    current_location = transform.location
    current_yaw = transform.rotation.yaw

    desired_yaw = get_yaw_to_target(current_location, target_location)
    yaw_error = normalize_angle_deg(desired_yaw - current_yaw)

    steer = max(-1.0, min(1.0, yaw_error / 35.0))

    current_speed = vehicle.get_velocity()
    speed_mps = math.sqrt(
        current_speed.x ** 2 + current_speed.y ** 2 + current_speed.z ** 2
    )

    speed_error = target_speed_mps - speed_mps
    throttle = max(0.0, min(1.0, speed_error * 0.20))

    brake = 0.0
    if speed_error < -0.8:
        brake = max(0.0, min(1.0, -speed_error * 0.25))
        throttle = 0.0

    control = carla.VehicleControl()
    control.throttle = throttle
    control.brake = brake
    control.steer = steer
    control.reverse = False
    control.hand_brake = False
    vehicle.apply_control(control)

    return control, speed_mps, yaw_error


def spawn_vehicle(world, blueprint_library, start_location, yaw):
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_transform = carla.Transform(
        carla.Location(x=start_location.x, y=start_location.y, z=start_location.z + 0.5),
        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
    )
    vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
    return vehicle


def main():
    chain, ordered_points = get_points_for_map()

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle = None
    try:
        first_point = to_carla_location(ordered_points[0]["point"])
        second_point = to_carla_location(ordered_points[1]["point"])
        yaw = get_yaw_to_target(first_point, second_point)

        vehicle = spawn_vehicle(world, blueprint_library, first_point, yaw)
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))

        smooth_points = build_smooth_trajectory(ordered_points)

        print(f"开始跟随轨迹: {chain['lane_text']}")
        print("原始轨迹点顺序:")
        for item in ordered_points:
            p = item["point"]
            print(f"  {item['name']}: ({p['x']:.3f}, {p['y']:.3f}, {p.get('z', 0.0):.3f})")
        print(f"平滑后的追踪点数量: {len(smooth_points)}")

        target_index = 1
        max_ticks = 3000

        for tick in range(max_ticks):
            world.tick()

            if target_index >= len(smooth_points):
                print("已到达全部平滑轨迹点")
                break

            target_name, target_point_dict = smooth_points[target_index]
            target_location = to_carla_location(target_point_dict)

            current_location = vehicle.get_transform().location
            dist = distance_2d(current_location, target_location)

            if dist < POINT_HIT_DISTANCE:
                print(f"到达点 {target_name}，距离 {dist:.2f}m")
                target_index += 1
                continue

            control, speed_mps, yaw_error = apply_control_to_target(
                vehicle,
                target_location,
                TARGET_SPEED_MPS,
            )

            print(
                f"tick={tick:04d} target={target_name} "
                f"dist={dist:.2f}m speed={speed_mps:.2f}m/s yaw_err={yaw_error:.1f} "
                f"throttle={control.throttle:.2f} steer={control.steer:.2f} brake={control.brake:.2f}"
            )

            time.sleep(0.02)

        print("轨迹跟随结束")

    finally:
        if vehicle is not None:
            vehicle.destroy()

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()