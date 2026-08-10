import json
import os
import traceback
from typing import List, Dict, Any, Union
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil 

# --- 路径与配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")
FAISS_INDEX_BASE_DIR = os.path.join(SCRIPT_DIR, "faiss_indices")
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "bge-m3")

# --- 统一的库配置 ---
LIBRARY_CONFIG = {
    "cidas": {"file": "scenarios.json", "name": "CIDAS (Public)", "type": "scenario"},
    "project": {"file": "scenarios2.json", "name": "Project (Private)", "type": "scenario"},
    "cda": {"file": "scenarios3.json", "name": "CDA (Public)", "type": "scenario"},
    "triggers1": {"file": "triggers.json", "name": "触发条件库", "type": "trigger"},
    "maps": {"file": "map_semantics.json", "name": "地图语义库", "type": "map"},
    "maps_carla": {"file": "all_Town_maps.json", "name": "Carla地图语义库 (Carla)", "type": "map"}
}
# --- 核心函数 ---

def create_document_from_scenario(scenario: Dict[str, Any]) -> str:
    """
    将单个场景转换为向量文本。
    【逻辑保持】：为了保证匹配精度，这里只提取 'L4_DynamicObjects' 进行向量化，
    忽略 L0/L1/L6 等非动态交互信息，以免干扰 AI 匹配。
    但请放心，完整的 JSON 会通过 metadata 存入索引，前端依然能显示所有层级。
    """
    if not isinstance(scenario, dict): return ""
    
    # 1. 提取标签 (Annotations)
    annotations = scenario.get('annotations', {})
    tags_parts = [f"{k}: {v}" for k, v in annotations.items() if v]
    tags_text = ". ".join(tags_parts)

    # 2. 仅提取 L4 层 (Dynamic Objects)
    l4_description = "无动态行为描述"
    layers = scenario.get('layers', {})
    if layers and 'L4_DynamicObjects' in layers:
        l4_description = layers['L4_DynamicObjects']

    return (
        f"场景名称: {scenario.get('name', '未命名')}. "
        f"场景描述: {scenario.get('description', '无描述')}. "
        f"关键特征: {tags_text}. "
        f"核心动态交互: {l4_description}"
    )

def create_document_from_trigger(trigger: Dict[str, Any]) -> str:
    """将触发条件转换为文本"""
    if not isinstance(trigger, dict): return ""
    
    source_parts = []
    for key, value in trigger.get('trigger_source', {}).items():
        if isinstance(value, dict):
            details = ' '.join(f"{k}:{v}" for k, v in value.items())
            source_parts.append(f"{key} {details}")
    source_text = ". ".join(source_parts) if source_parts else "未知"

    mechanism_parts = [f"{m.get('dimension', '')} {m.get('mechanism', '')} 导致 {m.get('result', '')}" for m in trigger.get('trigger_mechanism', [])]
    mechanism_text = ". ".join(mechanism_parts)

    behavior_parts = [f"{h.get('type', '')}: {h.get('description', '')}" for h in trigger.get('hazardous_behavior', [])]
    behavior_text = ". ".join(behavior_parts)

    return (
        f"触发条件名称: {trigger.get('name', '未命名')}. "
        f"ID: {trigger.get('id', '无ID')}. "
        f"相关功能: {trigger.get('function_group', '未指定')}. "
        f"触发源: {source_text}. "
        f"触发机制: {mechanism_text}. "
        f"相关危险行为: {behavior_text}."
    )

def create_document_from_map(map_data: Dict[str, Any]) -> str:
    """将地图数据转换为文本"""
    if not isinstance(map_data, dict): return ""

    parts = []
    map_id = map_data.get('map_id', '未知ID')
    topology = map_data.get('topology', '未知拓扑')
    parts.append(f"地图ID: {map_id}. 地图拓扑结构类型: {topology}.")

    if 'junctions' in map_data and map_data['junctions']:
        for j in map_data['junctions']:
            parts.append(f"路口ID: {j.get('junction_id')}.")
            features = []
            if j.get('has_unprotected_left_turn'): features.append("支持无保护左转")
            if features: parts.append(f"特征: {', '.join(features)}.")

    return " ".join(parts)

def create_document_from_map_carla(map_key: str, map_data: Dict[str, Any]) -> str:
    """
    从 Carla 地图标注中提取 map_description 的四个核心语义字段，
    拼成用于检索的文本。
    """
    if not isinstance(map_data, dict):
        return ""

    map_description = map_data.get("map_description", {})

    # 兼容异常格式：个别数据可能直接是字符串
    if isinstance(map_description, str):
        return f"map_key: {map_key}. map_description: {map_description.strip()}"

    if not isinstance(map_description, dict):
        map_description = {}

    lane_and_direction = str(map_description.get("lane_and_direction", "")).strip()
    environment_and_geometry = str(map_description.get("environment_and_geometry", "")).strip()

    return (
        f"lane_and_direction: {lane_and_direction}. "
        f"environment_and_geometry: {environment_and_geometry}. "
    )


def build_and_save_all_indices():
    """主函数：构建并保存所有库的索引"""
    print("="*60)
    print("=== 开始构建向量索引 (L4核心匹配 + 全量数据存储) ===")
    print("="*60)

    print(f"[1] 正在加载嵌入模型: '{LOCAL_MODEL_PATH}'...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"!! 错误: 模型路径不存在: '{LOCAL_MODEL_PATH}'")
        return
    
    os.makedirs(FAISS_INDEX_BASE_DIR, exist_ok=True)

    try:
        embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
        print("  - 模型加载成功。")
    except Exception as e:
        print(f"!! 模型加载严重错误: {e}")
        traceback.print_exc()
        return

    # --- 遍历库 ---
    for lib_id, config in LIBRARY_CONFIG.items():
        print("\n" + "-"*50)
        print(f" 处理库: '{config['name']}' (ID: {lib_id}) ")
        print("-"*50)

        library_path = os.path.join(PUBLIC_DIR, config['file'])
        index_path = os.path.join(FAISS_INDEX_BASE_DIR, f"index_{lib_id}")

        if not os.path.exists(library_path):
            print(f"  - 警告: 文件未找到: '{library_path}'")
            continue
        
        try:
            with open(library_path, 'r', encoding='utf-8') as f:
                items_to_index = json.load(f)
            if not items_to_index:
                print("  - 库为空。")
                continue
            print(f"  - 加载了 {len(items_to_index)} 条记录。")
        except Exception as e:
            print(f"  - JSON 读取失败: {e}")
            continue
        
        # 选择处理函数
        if config['type'] == 'scenario':
            doc_creation_func = create_document_from_scenario
            id_field = 'id'
            valid_items = [item for item in items_to_index if isinstance(item, dict)]
            documents = [doc_creation_func(item) for item in valid_items]
            metadatas = valid_items
            ids = [str(item.get(id_field, f"{lib_id}-{i}")) for i, item in enumerate(valid_items)]

        elif config['type'] == 'trigger':
            doc_creation_func = create_document_from_trigger
            id_field = 'id'
            valid_items = [item for item in items_to_index if isinstance(item, dict)]
            documents = [doc_creation_func(item) for item in valid_items]
            metadatas = valid_items
            ids = [str(item.get(id_field, f"{lib_id}-{i}")) for i, item in enumerate(valid_items)]

        elif config['type'] == 'map':
            if lib_id == 'maps_carla':
                doc_creation_func = create_document_from_map_carla
                # map_semantics_carla.json 为 dict: {map_key: map_obj}
                if isinstance(items_to_index, dict):
                    valid_items = [(str(k), v) for k, v in items_to_index.items() if isinstance(v, dict)]
                else:
                    valid_items = []
                documents = [doc_creation_func(map_key, map_obj) for map_key, map_obj in valid_items]
                metadatas = [map_obj for _, map_obj in valid_items]
                ids = [map_key for map_key, _ in valid_items]
            else:
                doc_creation_func = create_document_from_map
                id_field = 'map_id'
                valid_items = [item for item in items_to_index if isinstance(item, dict)]
                documents = [doc_creation_func(item) for item in valid_items]
                metadatas = valid_items
                ids = [str(item.get(id_field, f"{lib_id}-{i}")) for i, item in enumerate(valid_items)]
        else:
            continue
        
        print(f"  - 正在生成索引...")
        try:
            vector_store = FAISS.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas, ids=ids)
            vector_store.save_local(index_path)
            print(f"  - 🎉 成功保存索引至 '{index_path}'")
        except Exception as e:
            print(f"  - !! 索引创建失败: {e}")

    print("\n" + "="*60)
    print("✅ 所有索引构建完成。请重启后端 app.py 以应用更改。")
    print("="*60)

if __name__ == "__main__":
    build_and_save_all_indices()