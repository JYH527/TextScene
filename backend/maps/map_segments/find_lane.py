import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


JSON_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\all_Town_maps_filtered.json")
XODR_DIR = Path(r"d:\ProjectVW\developers_files\project\backend\maps")
OUTPUT_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\driving_paths_true_lane_links.json")


DRIVING_PATH_RE = re.compile(r"From\s+Road_(\d+)\s+to\s+Road_(\d+)\s+via\s+Road_(\d+)")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_driving_path(text: str) -> Tuple[str, str, str]:
    """解析 `From Road_A to Road_B via Road_C`。"""
    match = DRIVING_PATH_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"无法解析 driving_path: {text}")
    return match.group(1), match.group(2), match.group(3)


def get_xodr_path_for_map(map_key: str) -> Path:
    town = map_key.split("_", 1)[0]
    return XODR_DIR / f"{town}.xodr"


def parse_lane_links(lane: ET.Element) -> Dict[str, List[str]]:
    lane_link = lane.find("link")
    links = {"predecessor": [], "successor": []}
    if lane_link is None:
        return links

    for predecessor in lane_link.findall("predecessor"):
        lane_id = predecessor.attrib.get("id")
        if lane_id is not None:
            links["predecessor"].append(lane_id)

    for successor in lane_link.findall("successor"):
        lane_id = successor.attrib.get("id")
        if lane_id is not None:
            links["successor"].append(lane_id)

    return links


def parse_road_lane_sections(road: ET.Element) -> List[dict]:
    lane_sections = []
    lanes = road.find("lanes")
    if lanes is None:
        return lane_sections

    for lane_section in lanes.findall("laneSection"):
        section = {
            "s": float(lane_section.attrib.get("s", "0")),
            "lanes": {},
        }
        for side_name in ("left", "center", "right"):
            side = lane_section.find(side_name)
            if side is None:
                continue
            for lane in side.findall("lane"):
                lane_id = lane.attrib.get("id")
                if lane_id is None:
                    continue
                section["lanes"][lane_id] = {
                    "id": lane_id,
                    "type": lane.attrib.get("type", ""),
                    "links": parse_lane_links(lane),
                }
        lane_sections.append(section)

    lane_sections.sort(key=lambda item: item["s"])
    return lane_sections


def parse_road_endpoint_links(road: ET.Element) -> Dict[str, Optional[dict]]:
    link = road.find("link")
    result = {"predecessor": None, "successor": None}
    if link is None:
        return result

    predecessor = link.find("predecessor")
    successor = link.find("successor")
    if predecessor is not None:
        result["predecessor"] = dict(predecessor.attrib)
    if successor is not None:
        result["successor"] = dict(successor.attrib)
    return result


def parse_open_drive(xodr_path: Path) -> Dict[str, object]:
    """
    解析 OpenDRIVE。

    这里的关键逻辑是：
    1. `<junction>/<connection>/<laneLink>` 只给出 incomingRoad lane -> connectingRoad lane。
    2. `connectingRoad` 后续接到哪条 road，要看 connecting road 自己的 `<road>/<link>`。
    3. connecting road lane 接到目标 road 哪条 lane，要看 connecting road lane 的
       `<predecessor>` 或 `<successor>`，方向由 road link 中 predecessor/successor 指向的 road 决定。
    """
    tree = ET.parse(xodr_path)
    root = tree.getroot()

    roads = {}
    connections_by_incoming_and_connecting = defaultdict(list)

    for road in root.findall("road"):
        road_id = road.attrib["id"]
        roads[road_id] = {
            "id": road_id,
            "name": road.attrib.get("name", f"Road {road_id}"),
            "junction": road.attrib.get("junction", "-1"),
            "road_links": parse_road_endpoint_links(road),
            "lane_sections": parse_road_lane_sections(road),
        }

    for junction in root.findall("junction"):
        junction_id = junction.attrib.get("id", "")
        for connection in junction.findall("connection"):
            incoming_road = connection.attrib.get("incomingRoad")
            connecting_road = connection.attrib.get("connectingRoad")
            if incoming_road is None or connecting_road is None:
                continue

            lane_links = []
            for lane_link in connection.findall("laneLink"):
                from_lane = lane_link.attrib.get("from")
                to_lane = lane_link.attrib.get("to")
                if from_lane is None or to_lane is None:
                    continue
                lane_links.append({"from": from_lane, "to": to_lane})

            conn_info = {
                "junction_id": junction_id,
                "connection_id": connection.attrib.get("id", ""),
                "incomingRoad": incoming_road,
                "connectingRoad": connecting_road,
                "contactPoint": connection.attrib.get("contactPoint", ""),
                "laneLinks": lane_links,
            }
            connections_by_incoming_and_connecting[(incoming_road, connecting_road)].append(conn_info)

    return {
        "roads": roads,
        "connections_by_incoming_and_connecting": connections_by_incoming_and_connecting,
    }


def get_lane_type(road_data: dict, lane_id: str) -> Optional[str]:
    """获取指定 road 上 lane_id 的类型。"""
    for section in road_data.get("lane_sections", []):
        lane_info = section["lanes"].get(lane_id)
        if lane_info is not None:
            return lane_info.get("type")
    return None


def lane_at_connecting_road_exit(road_data: dict, via_lane: str, target_side: str) -> List[str]:
    """
    从 connecting road 的 lane link 中取通往目标 road 的 lane。

    target_side 为：
    - `successor`：目标 road 是 connecting road 的 successor road
    - `predecessor`：目标 road 是 connecting road 的 predecessor road

    CARLA/RoadRunner 的 junction connectingRoad 通常只有一个 laneSection；
    为了稳健，这里会扫描所有 laneSection，只取 via_lane 对应的真实 lane link。
    """
    result = []
    seen = set()

    for section in road_data.get("lane_sections", []):
        lane_info = section["lanes"].get(via_lane)
        if lane_info is None:
            continue
        if lane_info.get("type") != "driving":
            continue

        for lane_id in lane_info["links"].get(target_side, []):
            if get_lane_type(road_data, lane_id) != "driving":
                continue
            if lane_id not in seen:
                seen.add(lane_id)
                result.append(lane_id)

    return result


def target_side_for_via_road(via_road_data: dict, from_road_id: str, to_road_id: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    判断 via road 是通过 predecessor 还是 successor 接到 to_road。

    例如 Road_40：
    <predecessor elementType="road" elementId="0" contactPoint="end"/>
    <successor elementType="road" elementId="1" contactPoint="start"/>

    对 `From Road_0 to Road_1 via Road_40`，目标侧就是 successor。
    """
    road_links = via_road_data.get("road_links", {})
    predecessor = road_links.get("predecessor")
    successor = road_links.get("successor")

    if successor and successor.get("elementType") == "road" and successor.get("elementId") == to_road_id:
        return "successor", successor
    if predecessor and predecessor.get("elementType") == "road" and predecessor.get("elementId") == to_road_id:
        return "predecessor", predecessor

    # 有些数据的语义方向和 road link 前后方向相反；如果 from road 在 successor，
    # 那么 to road 很可能在 predecessor。
    if successor and successor.get("elementType") == "road" and successor.get("elementId") == from_road_id:
        if predecessor and predecessor.get("elementType") == "road":
            return "predecessor", predecessor
    if predecessor and predecessor.get("elementType") == "road" and predecessor.get("elementId") == from_road_id:
        if successor and successor.get("elementType") == "road":
            return "successor", successor

    return None, None


def resolve_true_lane_paths(open_drive: dict, from_road_id: str, to_road_id: str, via_road_id: str) -> List[dict]:
    """
    根据真实 OpenDRIVE 拓扑解析：
    `from road lane -> via road lane -> to road lane`。

    注意：不会把所有 lane 做笛卡尔积；只使用 `<junction>/<connection>/<laneLink>` 中真实存在的
    `from -> via`，再使用 via road 自身 lane `<link>` 中真实存在的 `via -> to`。
    """
    roads = open_drive["roads"]
    connection_map = open_drive["connections_by_incoming_and_connecting"]
    via_road = roads.get(via_road_id)
    if via_road is None:
        return []

    first_connections = connection_map.get((from_road_id, via_road_id), [])
    if not first_connections:
        return []

    target_side, target_road_link = target_side_for_via_road(via_road, from_road_id, to_road_id)
    if target_side is None or target_road_link is None:
        return []

    # 严格确认 via road 的 road link 最终确实接到语义路径中的 to_road。
    actual_to_road = target_road_link.get("elementId")
    if actual_to_road != to_road_id:
        return []

    results = []
    seen = set()

    for connection in first_connections:
        for lane_link in connection["laneLinks"]:
            from_lane = lane_link["from"]
            via_lane = lane_link["to"]
            to_lanes = lane_at_connecting_road_exit(via_road, via_lane, target_side)

            for to_lane in to_lanes:
                key = (from_road_id, from_lane, via_road_id, via_lane, to_road_id, to_lane)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "from_road": f"Road_{from_road_id}",
                    "from_lane": from_lane,
                    "via_road": f"Road_{via_road_id}",
                    "via_lane": via_lane,
                    "to_road": f"Road_{to_road_id}",
                    "to_lane": to_lane,
                    "text": (
                        f"From Road_{from_road_id} lane {from_lane} "
                        f"-> Road_{via_road_id} lane {via_lane} "
                        f"-> Road_{to_road_id} lane {to_lane}"
                    ),
                    "junction_id": connection["junction_id"],
                    "connection_id": connection["connection_id"],
                    "connection_contact_point": connection["contactPoint"],
                    "via_road_target_side": target_side,
                    "via_road_target_contact_point": target_road_link.get("contactPoint"),
                })

    return results


def iter_driving_paths(item: dict):
    """从 junctions_semantic 中遍历所有 driving_paths。"""
    junctions_semantic = item.get("junctions_semantic") or {}
    for junction_name, junction_data in junctions_semantic.items():
        for path_text in junction_data.get("driving_paths", []):
            yield junction_name, path_text


def main() -> None:
    data = load_json(JSON_PATH)
    xodr_cache = {}
    results = {}

    total_paths = 0
    resolved_paths = 0
    total_lane_chains = 0

    for map_key, item in data.items():
        xodr_path = get_xodr_path_for_map(map_key)
        if not xodr_path.exists():
            print(f"[WARN] 找不到对应 xodr: {xodr_path}")
            continue

        if xodr_path not in xodr_cache:
            print(f"[INFO] 解析 {xodr_path.name}")
            xodr_cache[xodr_path] = parse_open_drive(xodr_path)

        open_drive = xodr_cache[xodr_path]
        per_map = []

        for junction_name, path_text in iter_driving_paths(item):
            total_paths += 1
            try:
                from_road_id, to_road_id, via_road_id = parse_driving_path(path_text)
                lane_chains = resolve_true_lane_paths(open_drive, from_road_id, to_road_id, via_road_id)
                if lane_chains:
                    resolved_paths += 1
                    total_lane_chains += len(lane_chains)

                per_map.append({
                    "junction": junction_name,
                    "original": path_text,
                    "connections": lane_chains,
                })
            except Exception as exc:
                per_map.append({
                    "junction": junction_name,
                    "original": path_text,
                    "connections": [],
                    "error": str(exc),
                })

        results[map_key] = per_map

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成，结果已写入: {OUTPUT_PATH}")
    print(f"driving_paths 总数: {total_paths}")
    print(f"成功解析路径数: {resolved_paths}")
    print(f"真实 lane chain 总数: {total_lane_chains}")


if __name__ == "__main__":
    main()
