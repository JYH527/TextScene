import json
import re
from pathlib import Path

INPUT_PATH = Path(r"D:\ProjectVW\developers_files\project\backend\maps\map_segments\all_Town_maps.json")
BACKUP_PATH = INPUT_PATH.with_suffix(".backup.json")

def extract_via_roads(driving_paths):
    """
    从 driving_paths 中提取 via 后面的道路名，例如：
    "From Road_0 to Road_1 via Road_40" -> "Road_40"
    """
    via_roads = set()
    for item in driving_paths:
        m = re.search(r"\bvia\s+(Road_\d+)\b", item)
        if m:
            via_roads.add(m.group(1))
    return via_roads

def extract_used_roads(driving_paths):
    """
    从 driving_paths 中提取所有参与路径的道路：
    - From 后面的起点道路
    - to 后面的终点道路
    - via 后面的内部道路
    """
    used_roads = set()
    for item in driving_paths:
        m1 = re.search(r"\bFrom\s+(Road_\d+)\b", item)
        m2 = re.search(r"\bto\s+(Road_\d+)\b", item)
        m3 = re.search(r"\bvia\s+(Road_\d+)\b", item)

        if m1:
            used_roads.add(m1.group(1))
        if m2:
            used_roads.add(m2.group(1))
        if m3:
            used_roads.add(m3.group(1))
    return used_roads


def has_long_road(raw_roads, min_length=100.0):
    """判断 raw_roads 中是否存在长度大于 min_length 的道路。"""
    for road_obj in raw_roads.values():
        info = road_obj.get("info", "")
        m = re.search(r"(?:Len|length)\s*:\s*(\d+(?:\.\d+)?)\s*m?", info, re.IGNORECASE)
        if m and float(m.group(1)) > min_length:
            return True
    return False


def clean_scene(scene_obj):
    junctions = scene_obj.get("junctions_semantic", {})
    raw_roads = scene_obj.get("raw_roads", {})

    # 无论 junctions_semantic 是否为空，都先删除 raw_roads 中的 tunnels
    for road_obj in raw_roads.values():
        if "tunnels" in road_obj:
            del road_obj["tunnels"]

    # 无论 junctions_semantic 是否为空，都删除 map_description 中的 core_topology
    map_description = scene_obj.get("map_description", {})
    if isinstance(map_description, dict) and "core_topology" in map_description:
        del map_description["core_topology"]

    # junctions_semantic 为空时：如果没有长度大于 100m 的道路，则删除该场景
    if not junctions:
        summary = scene_obj.get("summary", {})
        if isinstance(summary, dict):
            summary["total_roads"] = len(raw_roads)
            summary["total_junctions"] = 0
        return has_long_road(raw_roads)

    # 收集本场景中所有 driving_paths 的道路
    all_driving_paths = []
    for junc in junctions.values():
        all_driving_paths.extend(junc.get("driving_paths", []))

    via_roads = extract_via_roads(all_driving_paths)
    used_roads = extract_used_roads(all_driving_paths)

    # 1) 删除 raw_roads 中不在 driving_paths 里的道路
    for road_name in list(raw_roads.keys()):
        if road_name not in used_roads:
            del raw_roads[road_name]

    # 3) 删除 internal_parts 中不在 via 后面的道路
    for junc in junctions.values():
        internal_parts = junc.get("internal_parts", [])
        junc["internal_parts"] = [r for r in internal_parts if r in via_roads]

    # 保持 summary 中的统计数量与实际数据一致（必须放在删除 raw_roads 之后）
    summary = scene_obj.get("summary", {})
    if isinstance(summary, dict):
        summary["total_roads"] = len(raw_roads)
        summary["total_junctions"] = len(junctions)

    return True

def main():
    # 备份原文件
    BACKUP_PATH.write_text(INPUT_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    for scene_name in list(data.keys()):
        scene_obj = data[scene_name]
        if isinstance(scene_obj, dict):
            keep_scene = clean_scene(scene_obj)
            if not keep_scene:
                del data[scene_name]

    # 写回文件
    INPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"处理完成：{INPUT_PATH}")
    print(f"已生成备份：{BACKUP_PATH}")

if __name__ == "__main__":
    main()