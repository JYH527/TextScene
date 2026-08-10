import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


JSON_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\driving_paths_true_lane_links.json")
XODR_DIR = Path(r"d:\ProjectVW\developers_files\project\backend\maps")
OUTPUT_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\driving_paths_true_lane_points.json")

ROAD_ID_RE = re.compile(r"Road_(\d+)")


@dataclass
class GeometrySegment:
    s0: float
    x0: float
    y0: float
    hdg: float
    length: float
    kind: str
    curvature: float = 0.0
    curv_start: float = 0.0
    curv_end: float = 0.0
    a_u: float = 0.0
    b_u: float = 0.0
    c_u: float = 0.0
    d_u: float = 0.0
    a_v: float = 0.0
    b_v: float = 0.0
    c_v: float = 0.0
    d_v: float = 0.0
    p_range_start: float = 0.0
    p_range_end: float = 1.0
    p_is_arc_length: bool = False


@dataclass
class ElevationRecord:
    s0: float
    a: float
    b: float
    c: float
    d: float

    def z_at(self, ds: float) -> float:
        return self.a + self.b * ds + self.c * ds * ds + self.d * ds * ds * ds


@dataclass
class LaneWidthRecord:
    s_offset: float
    a: float
    b: float
    c: float
    d: float

    def width_at(self, ds: float) -> float:
        return self.a + self.b * ds + self.c * ds * ds + self.d * ds * ds * ds


@dataclass
class LaneInfo:
    lane_id: str
    lane_type: str
    widths: List[LaneWidthRecord]


@dataclass
class RoadInfo:
    road_id: str
    length: float
    junction: str
    road_links: Dict[str, Optional[dict]]
    geometries: List[GeometrySegment]
    elevations: List[ElevationRecord]
    lanes: Dict[str, LaneInfo]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_xodr_path_for_map(map_key: str) -> Path:
    town = map_key.split("_", 1)[0]
    return XODR_DIR / f"{town}.xodr"


def road_id_from_label(label: str) -> str:
    match = ROAD_ID_RE.search(label)
    if not match:
        raise ValueError(f"无法解析 road label: {label}")
    return match.group(1)


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def parse_geometry(road: ET.Element) -> List[GeometrySegment]:
    plan_view = road.find("planView")
    if plan_view is None:
        return []

    geometries: List[GeometrySegment] = []
    for geom in plan_view.findall("geometry"):
        kind = "line"
        curvature = 0.0
        if geom.find("arc") is not None:
            kind = "arc"
            curvature = parse_float(geom.find("arc").attrib.get("curvature"), 0.0)
        elif geom.find("spiral") is not None:
            kind = "spiral"
        elif geom.find("poly3") is not None:
            kind = "poly3"
        elif geom.find("paramPoly3") is not None:
            kind = "paramPoly3"

        spiral = geom.find("spiral")
        poly3 = geom.find("poly3")
        param_poly3 = geom.find("paramPoly3")

        geometry = GeometrySegment(
            s0=parse_float(geom.attrib.get("s")),
            x0=parse_float(geom.attrib.get("x")),
            y0=parse_float(geom.attrib.get("y")),
            hdg=parse_float(geom.attrib.get("hdg")),
            length=parse_float(geom.attrib.get("length")),
            kind=kind,
            curvature=curvature,
        )

        if spiral is not None:
            geometry.curv_start = parse_float(spiral.attrib.get("curvStart"), 0.0)
            geometry.curv_end = parse_float(spiral.attrib.get("curvEnd"), 0.0)

        if poly3 is not None:
            geometry.a_u = parse_float(poly3.attrib.get("aU"), 0.0)
            geometry.b_u = parse_float(poly3.attrib.get("bU"), 0.0)
            geometry.c_u = parse_float(poly3.attrib.get("cU"), 0.0)
            geometry.d_u = parse_float(poly3.attrib.get("dU"), 0.0)
            geometry.a_v = parse_float(poly3.attrib.get("aV"), 0.0)
            geometry.b_v = parse_float(poly3.attrib.get("bV"), 0.0)
            geometry.c_v = parse_float(poly3.attrib.get("cV"), 0.0)
            geometry.d_v = parse_float(poly3.attrib.get("dV"), 0.0)

        if param_poly3 is not None:
            geometry.a_u = parse_float(param_poly3.attrib.get("aU"), 0.0)
            geometry.b_u = parse_float(param_poly3.attrib.get("bU"), 0.0)
            geometry.c_u = parse_float(param_poly3.attrib.get("cU"), 0.0)
            geometry.d_u = parse_float(param_poly3.attrib.get("dU"), 0.0)
            geometry.a_v = parse_float(param_poly3.attrib.get("aV"), 0.0)
            geometry.b_v = parse_float(param_poly3.attrib.get("bV"), 0.0)
            geometry.c_v = parse_float(param_poly3.attrib.get("cV"), 0.0)
            geometry.d_v = parse_float(param_poly3.attrib.get("dV"), 0.0)
            geometry.p_range_start = parse_float(param_poly3.attrib.get("pRangeStart"), 0.0)
            geometry.p_range_end = parse_float(param_poly3.attrib.get("pRangeEnd"), 1.0)
            geometry.p_is_arc_length = param_poly3.attrib.get("pIsArcLength", "false").lower() == "true"

        geometries.append(geometry)

    geometries.sort(key=lambda g: g.s0)
    return geometries


def parse_elevations(road: ET.Element) -> List[ElevationRecord]:
    elevation_profile = road.find("elevationProfile")
    if elevation_profile is None:
        return []

    records: List[ElevationRecord] = []
    for elev in elevation_profile.findall("elevation"):
        records.append(
            ElevationRecord(
                s0=parse_float(elev.attrib.get("s")),
                a=parse_float(elev.attrib.get("a")),
                b=parse_float(elev.attrib.get("b")),
                c=parse_float(elev.attrib.get("c")),
                d=parse_float(elev.attrib.get("d")),
            )
        )

    records.sort(key=lambda e: e.s0)
    return records


def parse_lane_widths(lane: ET.Element) -> List[LaneWidthRecord]:
    widths: List[LaneWidthRecord] = []
    for width in lane.findall("width"):
        widths.append(
            LaneWidthRecord(
                s_offset=parse_float(width.attrib.get("sOffset")),
                a=parse_float(width.attrib.get("a")),
                b=parse_float(width.attrib.get("b")),
                c=parse_float(width.attrib.get("c")),
                d=parse_float(width.attrib.get("d")),
            )
        )
    widths.sort(key=lambda w: w.s_offset)
    return widths


def parse_road(road: ET.Element) -> RoadInfo:
    road_id = road.attrib["id"]
    lane_data: Dict[str, LaneInfo] = {}
    road_links = {"predecessor": None, "successor": None}

    link = road.find("link")
    if link is not None:
        predecessor = link.find("predecessor")
        successor = link.find("successor")
        if predecessor is not None:
            road_links["predecessor"] = dict(predecessor.attrib)
        if successor is not None:
            road_links["successor"] = dict(successor.attrib)

    lanes = road.find("lanes")
    if lanes is not None:
        for lane_section in lanes.findall("laneSection"):
            for side_name in ("left", "center", "right"):
                side = lane_section.find(side_name)
                if side is None:
                    continue
                for lane in side.findall("lane"):
                    lane_id = lane.attrib.get("id")
                    if lane_id is None:
                        continue
                    if lane_id not in lane_data:
                        lane_data[lane_id] = LaneInfo(
                            lane_id=lane_id,
                            lane_type=lane.attrib.get("type", ""),
                            widths=parse_lane_widths(lane),
                        )

    return RoadInfo(
        road_id=road_id,
        length=parse_float(road.attrib.get("length")),
        junction=road.attrib.get("junction", "-1"),
        road_links=road_links,
        geometries=parse_geometry(road),
        elevations=parse_elevations(road),
        lanes=lane_data,
    )


def parse_open_drive(xodr_path: Path) -> Dict[str, RoadInfo]:
    tree = ET.parse(xodr_path)
    root = tree.getroot()
    roads: Dict[str, RoadInfo] = {}
    for road in root.findall("road"):
        info = parse_road(road)
        roads[info.road_id] = info
    return roads


def get_geometry_at_s(road: RoadInfo, s: float) -> GeometrySegment:
    if not road.geometries:
        raise ValueError(f"Road_{road.road_id} 没有 planView geometry")
    candidate = road.geometries[0]
    for geom in road.geometries:
        if geom.s0 <= s:
            candidate = geom
        else:
            break
    return candidate


def get_elevation_at_s(road: RoadInfo, s: float) -> float:
    if not road.elevations:
        return 0.0
    candidate = road.elevations[0]
    for elev in road.elevations:
        if elev.s0 <= s:
            candidate = elev
        else:
            break
    ds = max(0.0, s - candidate.s0)
    return candidate.z_at(ds)


def _safe_asin(v: float) -> float:
    return math.asin(max(-1.0, min(1.0, v)))


def _safe_acos(v: float) -> float:
    return math.acos(max(-1.0, min(1.0, v)))


def _integrate_simpson(func, a: float, b: float, n: int = 200) -> float:
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        x = a + i * h
        s += func(x) * (4 if i % 2 == 1 else 2)
    return s * h / 3.0


def eval_geometry_xy(geom: GeometrySegment, s: float) -> Tuple[float, float, float]:
    ds = max(0.0, min(s - geom.s0, geom.length))

    if geom.kind == "line":
        x = geom.x0 + ds * math.cos(geom.hdg)
        y = geom.y0 + ds * math.sin(geom.hdg)
        return x, y, geom.hdg

    if geom.kind == "arc":
        k = geom.curvature
        if abs(k) < 1e-12:
            x = geom.x0 + ds * math.cos(geom.hdg)
            y = geom.y0 + ds * math.sin(geom.hdg)
            return x, y, geom.hdg

        radius = 1.0 / k
        theta = ds * k
        x = geom.x0 + radius * (math.sin(geom.hdg + theta) - math.sin(geom.hdg))
        y = geom.y0 - radius * (math.cos(geom.hdg + theta) - math.cos(geom.hdg))
        hdg = geom.hdg + theta
        return x, y, hdg

    if geom.kind == "spiral":
        # Fresnel-based clothoid integration: curvature changes linearly from curvStart to curvEnd.
        # We numerically integrate heading to keep the implementation robust without special functions.
        if geom.length <= 1e-12:
            return geom.x0, geom.y0, geom.hdg

        k0 = geom.curv_start
        k1 = geom.curv_end
        dk = (k1 - k0) / geom.length

        def heading(u: float) -> float:
            return geom.hdg + k0 * u + 0.5 * dk * u * u

        def dx(u: float) -> float:
            return math.cos(heading(u))

        def dy(u: float) -> float:
            return math.sin(heading(u))

        x = geom.x0 + _integrate_simpson(dx, 0.0, ds, 200)
        y = geom.y0 + _integrate_simpson(dy, 0.0, ds, 200)
        hdg = heading(ds)
        return x, y, hdg

    if geom.kind == "poly3":
        # OpenDRIVE poly3: x = u, y = v where the local path is (u(s), v(s)) in road frame.
        # Here s is the longitudinal parameter along the geometry.
        u = geom.a_u + geom.b_u * ds + geom.c_u * ds * ds + geom.d_u * ds * ds * ds
        v = geom.a_v + geom.b_v * ds + geom.c_v * ds * ds + geom.d_v * ds * ds * ds
        x = geom.x0 + u * math.cos(geom.hdg) - v * math.sin(geom.hdg)
        y = geom.y0 + u * math.sin(geom.hdg) + v * math.cos(geom.hdg)

        du = geom.b_u + 2.0 * geom.c_u * ds + 3.0 * geom.d_u * ds * ds
        dv = geom.b_v + 2.0 * geom.c_v * ds + 3.0 * geom.d_v * ds * ds
        hdg = geom.hdg + math.atan2(dv, du if abs(du) > 1e-12 else 1e-12)
        return x, y, hdg

    if geom.kind == "paramPoly3":
        # paramPoly3 uses parametric polynomial curves in u/v space.
        # If pIsArcLength=true, p is normalized by arc length; otherwise by parameter range.
        if geom.p_is_arc_length:
            p = ds / geom.length if geom.length > 1e-12 else 0.0
            p = max(0.0, min(1.0, p))
        else:
            span = geom.p_range_end - geom.p_range_start
            if abs(span) < 1e-12:
                p = 0.0
            else:
                p = geom.p_range_start + (geom.p_range_end - geom.p_range_start) * (ds / geom.length if geom.length > 1e-12 else 0.0)

        u = geom.a_u + geom.b_u * p + geom.c_u * p * p + geom.d_u * p * p * p
        v = geom.a_v + geom.b_v * p + geom.c_v * p * p + geom.d_v * p * p * p
        x = geom.x0 + u * math.cos(geom.hdg) - v * math.sin(geom.hdg)
        y = geom.y0 + u * math.sin(geom.hdg) + v * math.cos(geom.hdg)

        du = geom.b_u + 2.0 * geom.c_u * p + 3.0 * geom.d_u * p * p
        dv = geom.b_v + 2.0 * geom.c_v * p + 3.0 * geom.d_v * p * p
        hdg = geom.hdg + math.atan2(dv, du if abs(du) > 1e-12 else 1e-12)
        return x, y, hdg

    # fallback: keep running even for unsupported geometry
    x = geom.x0 + ds * math.cos(geom.hdg)
    y = geom.y0 + ds * math.sin(geom.hdg)
    return x, y, geom.hdg


def lane_width_at(lane: LaneInfo, s: float) -> float:
    if lane.lane_type != "driving":
        return 0.0
    if not lane.widths:
        return 0.0

    chosen = lane.widths[0]
    for width in lane.widths:
        if width.s_offset <= s:
            chosen = width
        else:
            break

    ds = max(0.0, s - chosen.s_offset)
    return chosen.width_at(ds)


def lane_center_offset(road: RoadInfo, lane_id: str, s: float) -> float:
    """
    返回 lane 中心相对 reference line 的横向偏移。

    规则：
    - right lane id < 0，偏移为负
    - left lane id > 0，偏移为正
    - lane 中心 = 参考线到该 lane 外边界的累计宽度 + 本车道半宽
    """
    lane_num = int(lane_id)
    if lane_num == 0:
        return 0.0

    sign = 1.0 if lane_num > 0 else -1.0
    count = abs(lane_num)

    if lane_num > 0:
        ordered_ids = [str(i) for i in range(1, count + 1)]
    else:
        ordered_ids = [str(-i) for i in range(1, count + 1)]

    total = 0.0
    for idx, ordered_lane_id in enumerate(ordered_ids, start=1):
        lane = road.lanes.get(ordered_lane_id)
        if lane is None:
            continue
        width = lane_width_at(lane, s)
        if idx < count:
            total += width
        else:
            total += width / 2.0
    return sign * total


def lane_center_point(road: RoadInfo, lane_id: str, s: float) -> Tuple[float, float, float]:
    geom = get_geometry_at_s(road, s)
    x_ref, y_ref, hdg = eval_geometry_xy(geom, s)
    offset = lane_center_offset(road, lane_id, s)
    x = x_ref + offset * math.cos(hdg + math.pi / 2.0)
    y = y_ref + offset * math.sin(hdg + math.pi / 2.0)
    z = get_elevation_at_s(road, s)
    return x, y, z


def point_dict_from_xyz(xyz: Tuple[float, float, float]) -> dict:
    return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}


def parse_lane_text_order(lane_text: str) -> Tuple[str, str, str, str, str, str]:
    """解析形如 `From Road_0 lane -1 -> Road_40 lane -1 -> Road_1 lane -1` 的文本。"""
    pattern = re.compile(
        r"^From\s+(Road_\d+)\s+lane\s+(-?\d+)\s+->\s+(Road_\d+)\s+lane\s+(-?\d+)\s+->\s+(Road_\d+)\s+lane\s+(-?\d+)$"
    )
    match = pattern.match(lane_text.strip())
    if not match:
        raise ValueError(f"无法解析 lane_text: {lane_text}")
    return match.group(1), match.group(2), match.group(3), match.group(4), match.group(5), match.group(6)


def road_point(road: RoadInfo, lane_id: str, side: str) -> Tuple[float, float, float]:
    s = 0.0 if side == "start" else road.length
    return lane_center_point(road, lane_id, s)


def road_entry_side(road: RoadInfo, junction_id: str) -> Optional[str]:
    for side in ("predecessor", "successor"):
        link = road.road_links.get(side)
        if link and link.get("elementType") == "junction" and str(link.get("elementId")) == str(junction_id):
            return "start" if side == "predecessor" else "end"
    return None


def road_travel_direction(road: RoadInfo, junction_id: str, is_incoming: bool) -> Optional[Tuple[str, str]]:
    """
    返回 (first_side, second_side)
    first_side = 行驶进入/离开 junction 之前的那一端
    second_side = 另一个端点
    """
    entry_side = road_entry_side(road, junction_id)
    if entry_side is None:
        return None
    if is_incoming:
        return ("start" if entry_side == "end" else "end", entry_side)
    return (entry_side, "start" if entry_side == "end" else "end")


def ordered_edge_points(road: RoadInfo, lane_id: str, first_side: str, second_side: str) -> List[dict]:
    first = road_point(road, lane_id, first_side)
    second = road_point(road, lane_id, second_side)
    return [
        {"name": f"{road.road_id}_{first_side}", "point": point_dict_from_xyz(first)},
        {"name": f"{road.road_id}_{second_side}", "point": point_dict_from_xyz(second)},
    ]


def extract_points_for_lane_chain(roads: Dict[str, RoadInfo], conn: dict) -> Optional[dict]:
    lane_text = conn.get("text", "")
    from_road_label, from_lane, via_road_label, via_lane, to_road_label, to_lane = parse_lane_text_order(lane_text)

    from_road_id = road_id_from_label(from_road_label)
    via_road_id = road_id_from_label(via_road_label)
    to_road_id = road_id_from_label(to_road_label)

    from_road = roads.get(from_road_id)
    via_road = roads.get(via_road_id)
    to_road = roads.get(to_road_id)
    if from_road is None or via_road is None or to_road is None:
        return None

    junction_id = str(conn.get("junction_id", ""))
    from_dir = road_travel_direction(from_road, junction_id, is_incoming=True)
    to_dir = road_travel_direction(to_road, junction_id, is_incoming=False)

    if from_dir is None or to_dir is None:
        # fallback：保留原点位，但不再承诺方向
        from_start = road_point(from_road, from_lane, "start")
        from_end = road_point(from_road, from_lane, "end")
        via_start = road_point(via_road, via_lane, "start")
        via_end = road_point(via_road, via_lane, "end")
        to_start = road_point(to_road, to_lane, "start")
        to_end = road_point(to_road, to_lane, "end")
        ordered_points = [
            {"name": "from_start", "point": point_dict_from_xyz(from_start)},
            {"name": "from_end", "point": point_dict_from_xyz(from_end)},
            {"name": "via_start", "point": point_dict_from_xyz(via_start)},
            {"name": "via_end", "point": point_dict_from_xyz(via_end)},
            {"name": "to_start", "point": point_dict_from_xyz(to_start)},
            {"name": "to_end", "point": point_dict_from_xyz(to_end)},
        ]
        return {
            "original": conn.get("original", ""),
            "lane_text": lane_text,
            "from_road": f"Road_{from_road_id}",
            "from_lane": from_lane,
            "via_road": f"Road_{via_road_id}",
            "via_lane": via_lane,
            "to_road": f"Road_{to_road_id}",
            "to_lane": to_lane,
            "points": {
                "from_start": point_dict_from_xyz(from_start),
                "from_end": point_dict_from_xyz(from_end),
                "via_start": point_dict_from_xyz(via_start),
                "via_end": point_dict_from_xyz(via_end),
                "to_start": point_dict_from_xyz(to_start),
                "to_end": point_dict_from_xyz(to_end),
            },
            "ordered_points": ordered_points,
            "travel_order": [p["name"] for p in ordered_points],
            "junction_id": junction_id,
            "direction_resolved": False,
        }

    from_first_side, from_second_side = from_dir
    to_first_side, to_second_side = to_dir

    # 连接路 via_road：优先从与 from_road 相接的一端走到与 to_road 相接的一端
    via_conn_contact = str(conn.get("connection_contact_point", "start"))
    via_first_side = via_conn_contact if via_conn_contact in ("start", "end") else "start"
    via_second_side = "end" if via_first_side == "start" else "start"

    from_ordered = ordered_edge_points(from_road, from_lane, from_first_side, from_second_side)
    via_ordered = ordered_edge_points(via_road, via_lane, via_first_side, via_second_side)
    to_ordered = ordered_edge_points(to_road, to_lane, to_first_side, to_second_side)

    ordered_points = from_ordered + via_ordered + to_ordered

    # 兼容旧结构，依然保留这 6 个字段，但它们现在表示真实方向下的两端
    points_map = {
        "from_start": from_ordered[0]["point"],
        "from_end": from_ordered[1]["point"],
        "via_start": via_ordered[0]["point"],
        "via_end": via_ordered[1]["point"],
        "to_start": to_ordered[0]["point"],
        "to_end": to_ordered[1]["point"],
    }

    return {
        "original": conn.get("original", ""),
        "lane_text": lane_text,
        "from_road": f"Road_{from_road_id}",
        "from_lane": from_lane,
        "via_road": f"Road_{via_road_id}",
        "via_lane": via_lane,
        "to_road": f"Road_{to_road_id}",
        "to_lane": to_lane,
        "points": points_map,
        "ordered_points": ordered_points,
        "travel_order": [p["name"] for p in ordered_points],
        "junction_id": junction_id,
        "direction_resolved": True,
        "from_direction": from_dir,
        "to_direction": to_dir,
    }


def main() -> None:
    data = load_json(JSON_PATH)
    cache: Dict[Path, Dict[str, RoadInfo]] = {}
    output: Dict[str, List[dict]] = {}

    total_chains = 0
    total_points = 0
    failed_chains = 0

    for map_key, lane_groups in data.items():
        xodr_path = get_xodr_path_for_map(map_key)
        if not xodr_path.exists():
            print(f"[WARN] 找不到对应 OpenDRIVE 文件: {xodr_path}")
            continue

        if xodr_path not in cache:
            print(f"[INFO] 解析 {xodr_path.name}")
            cache[xodr_path] = parse_open_drive(xodr_path)

        roads = cache[xodr_path]
        map_results: List[dict] = []

        for lane_group in lane_groups:
            for conn in lane_group.get("connections", []):
                total_chains += 1
                try:
                    item = extract_points_for_lane_chain(roads, conn)
                    if item is None:
                        failed_chains += 1
                        map_results.append({
                            "original": conn.get("original", ""),
                            "lane_text": conn.get("text", ""),
                            "error": "road not found in xodr",
                            "points": None,
                        })
                        continue

                    map_results.append(item)
                    total_points += 6
                except Exception as exc:
                    failed_chains += 1
                    map_results.append({
                        "original": conn.get("original", ""),
                        "lane_text": conn.get("text", ""),
                        "error": str(exc),
                        "points": None,
                    })

        output[map_key] = map_results

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成，输出文件: {OUTPUT_PATH}")
    print(f"处理 lane chain 数: {total_chains}")
    print(f"成功输出点数: {total_points}")
    print(f"失败 lane chain 数: {failed_chains}")


if __name__ == "__main__":
    main()
