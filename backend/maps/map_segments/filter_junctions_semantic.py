import json
import re
from pathlib import Path


INPUT_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\all_Town_maps copy.json")
OUTPUT_PATH = Path(r"d:\ProjectVW\developers_files\project\backend\maps\map_segments\all_Town_maps_filtered.json")


def load_json_relaxed(path: Path) -> dict:
    """优先按标准 JSON 读取；如果文件里有少量不规范写法，则做一次宽松解析。"""
    text = path.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 去掉注释（如果文件里混入了注释）
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)

    # 去掉对象/数组末尾多余逗号
    text = re.sub(r",\s*(\}|\])", r"\1", text)

    # 某些条目里的 driving_route 不是合法 JSON（例如未加引号的键/值），
    # 但这个脚本只需要 junctions_semantic，因此直接移除该字段再解析。
    text = re.sub(r'\s*,?\s*"driving_route"\s*:\s*\{[\s\S]*?\}\s*(?=,\s*"|\s*\})', "", text)

    return json.loads(text)


def filter_data(data: dict) -> dict:
    """保留 junctions_semantic 为空的元素；非空时仅保留其中恰好 1 个 junction 的元素。"""
    filtered = {}

    for key, value in data.items():
        junctions_semantic = value.get("junctions_semantic")

        # 空字典 / None：保持不变
        if not junctions_semantic:
            filtered[key] = value
            continue

        # 非空：只保留 junctions_semantic 恰好只有 1 个元素的条目
        if isinstance(junctions_semantic, dict) and len(junctions_semantic) == 1:
            filtered[key] = value

    return filtered


def main() -> None:
    data = load_json_relaxed(INPUT_PATH)

    filtered_data = filter_data(data)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成，输出文件: {OUTPUT_PATH}")
    print(f"原始条目数: {len(data)}")
    print(f"保留条目数: {len(filtered_data)}")


if __name__ == "__main__":
    main()
