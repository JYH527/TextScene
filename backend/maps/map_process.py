import json
import re
from pathlib import Path

INPUT_PATH = Path(r"d:\finetune-model\Carla_Map\map_segments\Town01_annotation_segments2.json")
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

def clean_scene(scene_obj):
    junctions = scene_obj.get("junctions_semantic", {})

    # 删除 map_description 中的 core_topology
    map_description = scene_obj.get("map_description", {})
    if isinstance(map_description, dict) and "core_topology" in map_description:
        del map_description["core_topology"]

    # 收集本场景中所有 driving_paths 的道路
    all_driving_paths = []
    for junc in junctions.values():
        all_driving_paths.extend(junc.get("driving_paths", []))

    via_roads = extract_via_roads(all_driving_paths)
    used_roads = extract_used_roads(all_driving_paths)

    # 1) 删除 raw_roads 中不在 driving_paths 里的道路
    raw_roads = scene_obj.get("raw_roads", {})
    for road_name in list(raw_roads.keys()):
        if road_name not in used_roads:
            del raw_roads[road_name]
        else:
            # 2) 删除 tunnels 字段
            if "tunnels" in raw_roads[road_name]:
                del raw_roads[road_name]["tunnels"]

    # 3) 删除 internal_parts 中不在 via 后面的道路
    for junc in junctions.values():
        internal_parts = junc.get("internal_parts", [])
        junc["internal_parts"] = [r for r in internal_parts if r in via_roads]

def main():
    # 备份原文件
    BACKUP_PATH.write_text(INPUT_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    for scene_name, scene_obj in data.items():
        if isinstance(scene_obj, dict):
            clean_scene(scene_obj)

    # 写回文件
    INPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"处理完成：{INPUT_PATH}")
    print(f"已生成备份：{BACKUP_PATH}")

if __name__ == "__main__":
    main()