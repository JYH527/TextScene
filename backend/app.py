from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
import os
import time
import subprocess
import uuid
import json
import re
import zipfile
import io
import contextlib
from flask import Response, stream_with_context, send_file   
from werkzeug.utils import secure_filename
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tkinter as tk
import sys
from tkinter import filedialog
from werkzeug.utils import secure_filename
import queue
import threading
import time

# --- 导入业务模块 (确保文件名与你提供的文件一致) ---
from logical_scenario_generator import (
        initialize_llm_and_chain as initialize_dsg_chain,
        generate_multiple_dangerous_scenarios,
        LogicalScenarioOutput,
        refine_scenario_for_library,
        create_document_from_scenario # 确保引用了辅助函数
    )
from physical_scenario_generator import (
        init_physical_chain as initialize_understanding_chain,
        generate_physical_scenario
    )
# 注意：这里假设文件名是 coder_generate.py，如果你的文件名是 coder_generate2.py 请自行修改
from openscenario_code_generator import (
        initialize_chain as initialize_code_gen_chain,
        generate_openscenario as generate_openscenario_code
    )
from esmini_to_carla import (
    generate_openscenario as convert_esmini_to_carla
)

# --- 路径与常量定义 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")
SIMULATIONS_SAVE_DIR = os.path.join(PROJECT_ROOT, "simulations")
XOSC_SAVE_DIR = os.path.join(os.path.dirname(__file__), "xosc")
PREDEFINED_XODR_DIR = os.path.join(os.path.dirname(__file__), "maps")
FAISS_INDEX_BASE_DIR = os.path.join(os.path.dirname(__file__), "faiss_indices")

# ================== [新增] 本地 VLLM 环境变量配置 ==================
#VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://172.20.200.91:8000/v1")
#VLLM_MODEL_NAME_CODE_GEN = os.getenv("VLLM_MODEL_NAME_CODE_GEN", "Qwen3-32B-AWQ")  
#VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://172.20.200.91:8000/v1")
VLLM_MODEL_NAME_CODE_GEN = os.getenv("VLLM_MODEL_NAME_CODE_GEN", "Qwen3.6-35B-A3B") 
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "cqu-123456") 
# ================================================================


# --- 模型注册表 ---
# 这里定义 Base URL 和 模型名称，API Key 将从前端请求中获取 (Local除外)
MODEL_REGISTRY = {
    # 1. 原有的云端模型
    "deepseekv3.2": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat" 
    },
    "gemini3_pro": {
        "base_url": "https://xinghuapi.com/v1",
        "model": "gemini-3.1-pro-preview" 
    },
    # 2. [新增] 本地模型
    "local": {
        "base_url": VLLM_BASE_URL,
        "model": VLLM_MODEL_NAME_CODE_GEN,
        "api_key": VLLM_API_KEY
    }
}
DEFAULT_MODEL_KEY = "deepseekv3.2" # 默认值，如果前端没传且想优先用本地，可改为 "local"
ADMIN_MASTER_PASSWORD = os.getenv("ADMIN_MASTER_PASSWORD", "admin888")

# --- 库配置 ---
LIBRARY_CONFIG = {
    "cidas": {"file": "scenarios.json", "name": "CIDAS (Public)", "password": None, "type": "scenario"},
    "project": {"file": "scenarios2.json", "name": "Project (Private)", "password": None, "type": "scenario"},
    "cda": {"file": "scenarios3.json", "name": "CDA (Public)", "password": None, "type": "scenario"},
    "triggers1": {"file": "triggers.json", "name": "触发条件库", "password": None, "type": "trigger"},
    "maps_carla": {"file": "all_Town_maps.json", "name": "Carla地图语义库 (Carla)", "password": None, "type": "map"}
}

REGISTRY_FILE = os.path.join(PROJECT_ROOT, "library_registry.json")

def load_registry():
    """读取名称注册表"""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}
    return {}

def check_permission(library_password, input_password):

    if input_password == ADMIN_MASTER_PASSWORD:
        return True
    if library_password is None:
        return True
    return library_password == input_password

def save_registry(registry):
    """保存名称注册表"""
    try:
        with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f" [ERROR] Failed to save registry: {e}")

# === [修改] 动态加载逻辑 (支持读取注册表中的名字) ===
def load_dynamic_libraries():
    print(" [INFO] Scanning for dynamic libraries...")
    if not os.path.exists(PUBLIC_DIR): return
    
    # 1. 加载“记事本”
    registry = load_registry()
    
    for filename in os.listdir(PUBLIC_DIR):
        if filename.endswith(".json"):
            # 排除系统保留文件
            if filename in ["1.json", "triggers.json", "map_semantics_carla.json", "scenarios.json", "scenarios2.json", "scenarios3.json"]:
                continue
            
            lib_id = os.path.splitext(filename)[0]
            
            # 2. 从注册表中获取元数据 (兼容旧格式：旧格式只是字符串，新格式是字典)
            entry = registry.get(filename)
            
            display_name = f"{lib_id} (Custom)"
            lib_password = None
            
            if entry:
                if isinstance(entry, dict):
                    # 新格式: {"name": "...", "password": "..."}
                    display_name = entry.get("name", display_name)
                    lib_password = entry.get("password", None)
                elif isinstance(entry, str):
                    # 旧格式兼容: "Display Name"
                    display_name = entry
            
            if lib_id not in LIBRARY_CONFIG:
                LIBRARY_CONFIG[lib_id] = {
                    "file": filename,
                    "name": display_name, 
                    "password": lib_password, # <--- 加载密码
                    "type": "scenario"
                }
                status = "LOCKED" if lib_password else "OPEN"
                print(f" [LOADED] Custom lib: {lib_id} -> {display_name} ({status})")

# 程序启动时加载
load_dynamic_libraries()

# 创建所有必要的目录
for directory in [SIMULATIONS_SAVE_DIR, PREDEFINED_XODR_DIR, XOSC_SAVE_DIR, PUBLIC_DIR, FAISS_INDEX_BASE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Esmini 配置
ESMINI_DEMO_BASE_PATH = os.path.join(PROJECT_ROOT,"esmini-demo")
ESMINI_EXECUTABLE_NAME = "esmini.exe" if os.name == 'nt' else "esmini"
ESMINI_EXECUTABLE_PATH = os.path.join(ESMINI_DEMO_BASE_PATH, "bin", ESMINI_EXECUTABLE_NAME)
ESMINI_WORKING_DIR = ESMINI_DEMO_BASE_PATH
ESMINI_AVAILABLE = os.path.isfile(ESMINI_EXECUTABLE_PATH)

if ESMINI_AVAILABLE:
    print(f"INFO: esmini executable found at: {ESMINI_EXECUTABLE_PATH}")
else:
    print(f"WARNING: esmini executable NOT FOUND at {ESMINI_EXECUTABLE_PATH}. Simulation endpoint will not work.")

# --- LLM初始化状态标志 ---
DSG_INITIALIZED, UNDERSTANDING_INITIALIZED, CODE_GEN_INITIALIZED = (False,)*3
RULES_FILE_PATH = "scenario_rules.md"
EMBEDDING_MODEL_NAME = os.path.join(PROJECT_ROOT, "bge-m3")
SCENARIO_RULES: str = ""
EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
VECTOR_STORES: Dict[str, Optional[FAISS]] = {}

#  1.json 本地缓存加载模块
LIBRARY_JSON_PATH = os.path.join(os.path.dirname(__file__), "1.json")
LIBRARY_SCENARIO_CACHE = {}

# 全局地图语义缓存
MAP_SEMANTICS_CACHE = {} 

def load_local_data():
    """统一加载本地数据：场景缓存和地图语义"""
    global LIBRARY_SCENARIO_CACHE, MAP_SEMANTICS_CACHE
    
    print("\n" + "="*50)
    print(f" [STARTUP] 正在加载本地数据...")

    # 1. 加载场景缓存 (1.json)
    LIBRARY_SCENARIO_CACHE = {} 
    if os.path.exists(LIBRARY_JSON_PATH):
        try:
            with open(LIBRARY_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = 0
            for item in data:
                if 'id' in item and 'formatted_scenario' in item:
                    s_id = str(item['id']).strip()
                    LIBRARY_SCENARIO_CACHE[s_id] = item['formatted_scenario']
                    count += 1
            print(f" [SUCCESS] 1.json 加载成功，缓存了 {count} 个场景。")
        except Exception as e:
            print(f" [ERROR] 1.json 加载失败: {e}")
            traceback.print_exc()
    else:
        print(f" [WARNING] 未找到 1.json，库匹配功能将不可用。")

    # 2. 加载地图语义库
    MAP_SEMANTICS_CACHE = {}
    map_config = LIBRARY_CONFIG.get("maps_carla")
    if map_config:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), map_config["file"]),
            os.path.join(PUBLIC_DIR, map_config["file"]),
            map_config["file"]
        ]
        
        found_map_file = False
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        MAP_SEMANTICS_CACHE = json.load(f)
                    print(f" [SUCCESS] 地图语义库加载成功: {len(MAP_SEMANTICS_CACHE)} 个地图定义 (From: {p})")
                    found_map_file = True
                    break
                except Exception as e:
                    print(f" [ERROR] 地图文件 {p} 解析失败: {e}")
        
        if not found_map_file:
            print(f" [WARNING] 未找到地图标注文档 ({map_config['file']})，直接查找功能将受限。")

    print("="*50 + "\n")

load_local_data()

def build_map_info_from_map_key(map_key: str, map_annotation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    将 scenario_generator2 产出的 map_key（例如 Town01_T-junction_01）
    归一化为 app 后续阶段需要的 map_info（至少包含 map_id / xodr_file）。
    """
    key = str(map_key or "").strip()
    ann = dict(map_annotation or {})

    if key and "_" in key:
        town_name = key.split("_", 1)[0]
    else:
        town_name = key

    map_id = ann.get("map_id") or key or town_name or "Town01"
    xodr_file = ann.get("xodr_file")

    if not xodr_file:
        base_name = town_name or key or "Town01"
        xodr_file = f"{base_name}.xodr"

    ann["map_id"] = map_id
    ann["xodr_file"] = xodr_file
    ann["map_key"] = key or ann.get("map_key", "")
    return ann

def extract_map_filename(text: str) -> str:
    if not text: return ""
    match = re.search(r'([\w-]+\.xodr)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

class XOSCParamHandler:
    @staticmethod
    def _get_namespace_url(xosc_content):
        try:
            import re
            match = re.search(r'xmlns="([^"]+)"', xosc_content)
            if match: return match.group(1)
        except: pass
        return None

    @staticmethod
    def _strip_namespace(tag):
        if '}' in tag: return tag.split('}', 1)[1]
        return tag

    @staticmethod
    def _normalize_key(name):
        return str(name).strip() if name else ""

    @staticmethod
    def _find_element_recursive(root, target_tag):
        """深度查找（忽略命名空间）"""
        for child in root.iter():
            if XOSCParamHandler._strip_namespace(child.tag) == target_tag:
                return child
        return None

    @staticmethod
    def parse_and_extract(xosc_content):
        params = {"ego": None, "targets": {}}
        try:
            root = ET.fromstring(xosc_content)
            storyboard = XOSCParamHandler._find_element_recursive(root, 'Storyboard')
            if storyboard is None: return params

            entities_data = {}
            init_node = XOSCParamHandler._find_element_recursive(storyboard, 'Init')
            
            if init_node:
                for private in init_node.iter():
                    if XOSCParamHandler._strip_namespace(private.tag) == 'Private':
                        entity_ref = private.get('entityRef', '')
                        if not entity_ref: continue
                        
                        # [修改] 初始化字典结构，增加 road_id 和 lane_id
                        entities_data[entity_ref] = {
                            'name': entity_ref, 
                            'init_speed': 0.0, 
                            'init_s': 0.0, 
                            'init_road_id': '', # [新增]
                            'init_lane_id': '', # [新增]
                            'speed_actions': []
                        }
                        
                        speed_node = XOSCParamHandler._find_element_recursive(private, 'AbsoluteTargetSpeed')
                        if speed_node is not None:
                            entities_data[entity_ref]['init_speed'] = float(speed_node.get('value', 0))
                        
                        pos_node = XOSCParamHandler._find_element_recursive(private, 'LanePosition')
                        if pos_node is not None:
                            entities_data[entity_ref]['init_s'] = float(pos_node.get('s', 0))
                            # [新增] 提取 roadId 和 laneId
                            entities_data[entity_ref]['init_road_id'] = pos_node.get('roadId', '')
                            entities_data[entity_ref]['init_lane_id'] = pos_node.get('laneId', '')

            def get_children(node, tag_name):
                return [c for c in node if XOSCParamHandler._strip_namespace(c.tag) == tag_name]

            stories = get_children(storyboard, 'Story')
            for story in stories:
                acts = get_children(story, 'Act')
                for act in acts:
                    groups = get_children(act, 'ManeuverGroup')
                    for group in groups:
                        current_group_actors = []
                        actors_container = XOSCParamHandler._find_element_recursive(group, 'Actors')
                        if actors_container is not None:
                            for entity_ref_node in actors_container.iter():
                                if XOSCParamHandler._strip_namespace(entity_ref_node.tag) == 'EntityRef':
                                    ref = entity_ref_node.get('entityRef')
                                    if ref: current_group_actors.append(ref)
                        
                        if not current_group_actors: continue

                        maneuvers = get_children(group, 'Maneuver')
                        for maneuver in maneuvers:
                            events = get_children(maneuver, 'Event')
                            for event in events:
                                event_name = event.get('name', 'Unknown')
                                speed_node = XOSCParamHandler._find_element_recursive(event, 'AbsoluteTargetSpeed')
                                if speed_node is not None:
                                    speed_val = float(speed_node.get('value', 0))
                                    for actor_ref in current_group_actors:
                                        if actor_ref in entities_data:
                                            existing_events = [e['event_name'] for e in entities_data[actor_ref]['speed_actions']]
                                            if event_name not in existing_events:
                                                entities_data[actor_ref]['speed_actions'].append({
                                                    'event_name': event_name,
                                                    'target_speed': speed_val
                                                })

            for ref, data in entities_data.items():
                if ref.lower() in ['ego', 'hero', 'ego_vehicle']:
                    params['ego'] = data
                else:
                    params['targets'][ref] = data

            return params
        except Exception as e:
            print(f"[XOSC Extract Error] {e}")
            return params

    @staticmethod
    def update_xml_params(xosc_content, modification_dict):
        if not modification_dict: return xosc_content
        print("\n--- [XOSC Update Debug Start] ---")
        try:
            ns_url = XOSCParamHandler._get_namespace_url(xosc_content)
            if ns_url: ET.register_namespace('', ns_url)
            
            root = ET.fromstring(xosc_content)
            storyboard = XOSCParamHandler._find_element_recursive(root, 'Storyboard')
            if not storyboard: return xosc_content

            init_node = XOSCParamHandler._find_element_recursive(storyboard, 'Init')
            if init_node:
                for elem in init_node.iter():
                    if XOSCParamHandler._strip_namespace(elem.tag) == 'Private':
                        entity_ref = elem.get('entityRef')
                        clean_ref = XOSCParamHandler._normalize_key(entity_ref)
                        
                        updates = None
                        if clean_ref.lower() in ['ego', 'hero'] and 'ego' in modification_dict:
                            updates = modification_dict['ego']
                        elif 'targets' in modification_dict:
                            if entity_ref in modification_dict['targets']:
                                updates = modification_dict['targets'][entity_ref]
                            elif clean_ref in modification_dict['targets']:
                                updates = modification_dict['targets'][clean_ref]

                        if updates:
                            if 'init_speed' in updates:
                                speed_node = XOSCParamHandler._find_element_recursive(elem, 'AbsoluteTargetSpeed')
                                if speed_node is not None: 
                                    speed_node.set('value', str(updates['init_speed']))
                            
                            # [修改] 查找 LanePosition 并更新所有相关属性
                            pos_node = XOSCParamHandler._find_element_recursive(elem, 'LanePosition')
                            if pos_node is not None: 
                                if 'init_s' in updates:
                                    pos_node.set('s', str(updates['init_s']))
                                # [新增] 更新 RoadID
                                if 'init_road_id' in updates:
                                    pos_node.set('roadId', str(updates['init_road_id']))
                                # [新增] 更新 LaneID
                                if 'init_lane_id' in updates:
                                    pos_node.set('laneId', str(updates['init_lane_id']))

            event_speed_map = {}
            def collect(d):
                if d and 'speed_actions' in d:
                    for act in d['speed_actions']:
                        k = XOSCParamHandler._normalize_key(act.get('event_name'))
                        v = act.get('target_speed')
                        if k and v is not None: event_speed_map[k] = v

            if 'ego' in modification_dict: collect(modification_dict['ego'])
            if 'targets' in modification_dict:
                for t in modification_dict['targets'].values(): collect(t)

            for elem in storyboard.iter():
                if XOSCParamHandler._strip_namespace(elem.tag) == 'Event':
                    raw_name = elem.get('name')
                    clean_name = XOSCParamHandler._normalize_key(raw_name)
                    if clean_name in event_speed_map:
                        new_val = str(event_speed_map[clean_name])
                        speed_node = XOSCParamHandler._find_element_recursive(elem, 'AbsoluteTargetSpeed')
                        if speed_node is not None:
                            speed_node.set('value', new_val)

            print("--- [XOSC Update Finished] ---")
            return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode', method='xml')

        except Exception as e:
            print(f"[XOSC Update Error] {e}")
            return xosc_content

# --- 初始化函数 ---
def initialize_rag_components():
    global VECTOR_STORES, SCENARIO_RULES, EMBEDDINGS
    print("\n--- Initializing RAG Components (Multi-Index Mode) ---")
    try:
        print(f"INFO: Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("INFO: Embedding model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load embedding model. RAG will be disabled. Error: {e}\n{traceback.format_exc()}")
        return
    
    for lib_id, config in LIBRARY_CONFIG.items():
        if config.get("type") == "map":
            print(f"INFO: Skipping vector index for '{lib_id}' (using direct lookup).")
            VECTOR_STORES[lib_id] = None
            continue

        index_path = os.path.join(FAISS_INDEX_BASE_DIR, f"index_{lib_id}")
        if os.path.exists(index_path):
            try:
                VECTOR_STORES[lib_id] = FAISS.load_local(index_path, EMBEDDINGS, allow_dangerous_deserialization=True)
                print(f"INFO: FAISS vector store for '{lib_id}' loaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to load FAISS index for '{lib_id}': {e}")
                VECTOR_STORES[lib_id] = None
        else:
            VECTOR_STORES[lib_id] = None
    
    try:
        if os.path.exists(RULES_FILE_PATH):
            with open(RULES_FILE_PATH, 'r', encoding='utf-8') as f: SCENARIO_RULES = f.read()
        else: SCENARIO_RULES = "无特定规则提供。"
    except Exception as e: SCENARIO_RULES = "加载规则文件时出错。"
    print("--- RAG Components Initialization Finished ---\n")

def initialize_llms():
    global DSG_INITIALIZED, UNDERSTANDING_INITIALIZED, CODE_GEN_INITIALIZED
    # 模块预热，实际连接在请求时建立
    print("Initializing Logical Scenario Generator Modules...")
    DSG_INITIALIZED = initialize_dsg_chain() 
    print("Initializing Scene Understander Modules...")
    UNDERSTANDING_INITIALIZED = initialize_understanding_chain()
    print("Initializing Code Generator Modules...")
    CODE_GEN_INITIALIZED = initialize_code_gen_chain()

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# --- [新增] 辅助函数：从请求中提取 LLM 配置 ---
def get_llm_config_from_request(request_data: Dict) -> Dict:
    """
    从前端请求中提取 model_selection 和 api_key，
    结合 MODEL_REGISTRY 生成完整的配置字典。
    """
    selected_key = request_data.get('model_selection', DEFAULT_MODEL_KEY)
    user_api_key = request_data.get('api_key', '').strip()
    
    # 获取注册表中的基础配置 (URL, Model Name)
    registry_config = MODEL_REGISTRY.get(selected_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    
    # 构造最终配置
    config = {
        "base_url": registry_config["base_url"],
        "model": registry_config["model"],
        "api_key": registry_config.get("api_key", user_api_key) # 本地模型可能已有Key，否则用用户的
    }
    
    # 如果不是本地模型，且用户没传key，使用默认或EMPTY
    if selected_key != "local" and not config["api_key"] and not user_api_key:
         config["api_key"] = "EMPTY"
    elif selected_key != "local" and user_api_key:
         config["api_key"] = user_api_key
         
    return config

# --- [新增] 手动导入场景的 API ---
@app.route('/api/manual_import_scenario', methods=['POST'])
def api_manual_import_scenario_route():
    try:
        # 1. 获取表单数据
        scenario_name = request.form.get('name')
        description = request.form.get('description', '')
        target_lib_id = request.form.get('target_library_id')
        password = request.form.get('password')

        # 2. 获取文件
        xosc_file = request.files.get('xosc_file')
        xodr_file = request.files.get('xodr_file')

        if not all([scenario_name, target_lib_id, xosc_file]):
            return jsonify({"error": "缺少必要参数：名称、库ID或 .xosc 文件"}), 400

        # 3. 验证库权限
        if target_lib_id not in LIBRARY_CONFIG:
            return jsonify({"error": "目标库ID无效"}), 400
        
        lib_config = LIBRARY_CONFIG[target_lib_id]
        
        # 权限校验 (支持管理员密码)
        if not check_permission(lib_config.get("password"), password):
            return jsonify({"error": "权限验证失败：密码错误"}), 403
        
        # 4. 生成唯一ID
        scenario_id = f"MANUAL-{int(time.time())}-{uuid.uuid4().hex[:4]}"

        # 5. 保存 .xodr 地图文件 (如果有)
        map_filename = "unknown.xodr"
        if xodr_file and xodr_file.filename:
            try:
                safe_map_name = secure_filename(xodr_file.filename)
                # 防止空文件名
                if not safe_map_name: safe_map_name = f"map_{int(time.time())}.xodr"
                
                if not os.path.exists(PREDEFINED_XODR_DIR):
                    os.makedirs(PREDEFINED_XODR_DIR)

                map_save_path = os.path.join(PREDEFINED_XODR_DIR, safe_map_name)
                xodr_file.save(map_save_path)
                map_filename = safe_map_name
                print(f" [IMPORT] 地图已保存: {map_save_path}")
            except Exception as e_map:
                print(f" [WARNING] 地图保存失败: {e_map}")
                # 地图保存失败不应阻止场景保存，继续执行

        # 6. 保存 .xosc 场景文件
        try:
            xosc_filename = f"{scenario_id}.xosc"
            if not os.path.exists(XOSC_SAVE_DIR):
                os.makedirs(XOSC_SAVE_DIR)
            xosc_save_path = os.path.join(XOSC_SAVE_DIR, xosc_filename)
            
            # [关键修复] 使用 errors='replace' 防止因编码问题导致的崩溃
            file_bytes = xosc_file.read()
            try:
                xosc_content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # 如果 utf-8 失败，尝试 gbk 或忽略错误
                try:
                    xosc_content = file_bytes.decode('gbk')
                except:
                    xosc_content = file_bytes.decode('utf-8', errors='ignore')
            
            # 简单的正则替换，确保 xosc 指向刚才上传的地图
            if map_filename != "unknown.xodr":
                 xosc_content = re.sub(
                    r'filepath\s*=\s*["\']?([^"\']+\.xodr)["\']?', 
                    f'filepath="../maps/{map_filename}"', 
                    xosc_content, 
                    flags=re.IGNORECASE
                )

            with open(xosc_save_path, 'w', encoding='utf-8') as f:
                f.write(xosc_content)
            print(f" [IMPORT] 场景文件已保存: {xosc_save_path}")
            
        except Exception as e_xosc:
            print(f" [ERROR] XOSC 文件处理失败: {e_xosc}")
            return jsonify({"error": f"场景文件处理失败: {str(e_xosc)}"}), 500

        # 7. 构造逻辑场景 JSON 对象
        scenario_obj = {
            "id": scenario_id,
            "name": scenario_name,
            "description": description or "User manually imported scenario.",
            "source": "manual_import",
            "map_key": map_filename, 
            "layers": {
                "L1_Road": f"Based on imported map: {map_filename}",
                "L4_Dynamic_Objects": "Defined in imported OpenSCENARIO file."
            },
            "annotations": {
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "import_type": "manual_upload"
            }
        }

        # 8. 写入 JSON 库文件
        target_json_path = os.path.join(PUBLIC_DIR, lib_config["file"])
        # 如果配置指向的文件不在 public 下，尝试在根目录找
        if not os.path.exists(target_json_path):
             alt_path = os.path.join(os.path.dirname(__file__), lib_config["file"])
             if os.path.exists(alt_path) or not os.path.exists(PUBLIC_DIR):
                 target_json_path = alt_path
        
        library_data = []
        if os.path.exists(target_json_path):
            try:
                with open(target_json_path, 'r', encoding='utf-8') as f:
                    library_data = json.load(f)
            except: library_data = []
        
        if not isinstance(library_data, list): library_data = []
        
        library_data.insert(0, scenario_obj) 
        
        # 确保目录存在
        os.makedirs(os.path.dirname(target_json_path), exist_ok=True)

        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, ensure_ascii=False, indent=2)

        # 9. 更新内存缓存
        global LIBRARY_SCENARIO_CACHE
        LIBRARY_SCENARIO_CACHE[scenario_id] = scenario_obj

        return jsonify({
            "message": f"场景导入成功！ID: {scenario_id}",
            "scenario_data": scenario_obj
        }), 201

    except Exception as e:
        print(f"CRITICAL ERROR in /api/manual_import_scenario: {e}\n{traceback.format_exc()}")
        # [关键] 即使发生未知错误，也要返回 JSON，防止前端无限等待
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

# --- [新增] 删除仿真文件接口 ---
@app.route('/api/delete_simulation_file', methods=['POST'])
def api_delete_simulation_file_route():
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path:
            return jsonify({"error": "Missing file path."}), 400

        # 安全检查：只允许删除 XOSC_SAVE_DIR 或 SIMULATIONS_SAVE_DIR 下的文件
        abs_path = os.path.abspath(file_path)
        abs_xosc_dir = os.path.abspath(XOSC_SAVE_DIR)
        abs_sim_dir = os.path.abspath(SIMULATIONS_SAVE_DIR)

        # 检查路径前缀是否匹配允许的目录
        is_safe = abs_path.startswith(abs_xosc_dir) or abs_path.startswith(abs_sim_dir)

        if not is_safe:
            print(f" [SECURITY BLOCK] Attempt to delete file outside safe dirs: {abs_path}")
            return jsonify({"error": "Permission denied: Cannot delete files outside allowed directories."}), 403

        if os.path.exists(abs_path):
            os.remove(abs_path)
            print(f" [INFO] Deleted file: {abs_path}")
            return jsonify({"message": "File deleted successfully.", "success": True}), 200
        else:
            return jsonify({"error": "File not found.", "success": False}), 404

    except Exception as e:
        print(f"ERROR in /api/delete_simulation_file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_library_list', methods=['GET'])
def api_get_library_list_route():
    """
    返回当前后端所有已加载的库配置列表给前端
    """
    # 确保返回前再扫描一次，防止文件变动没刷新
    load_dynamic_libraries()
    
    libraries = []
    for lib_id, config in LIBRARY_CONFIG.items():
        # 我们只返回 scenario 类型的库给侧边栏显示
        # (trigger 和 map 是特殊库，不在通用列表显示逻辑里)
        if config.get('type') == 'scenario':
            libraries.append({
                "id": lib_id,
                "name": config.get("name", lib_id),
                "file": config.get("file"),
                "password": config.get("password"),
                "type": "scenario"
            })
    return jsonify(libraries), 200

SYSTEM_PROTECTED_LIBS = ["cda", "cidas", "project", "triggers1", "maps_carla"]

@app.route('/api/delete_library', methods=['POST'])
def api_delete_library_route():
    try:
        data = request.get_json()
        lib_id = data.get('library_id')
        password = data.get('password') # <--- 获取前端传来的密码
        
        if not lib_id:
            return jsonify({"error": "缺少 library_id"}), 400

        # 1. 安全检查：禁止删除系统库
        if lib_id in SYSTEM_PROTECTED_LIBS:
            return jsonify({"error": "系统内置库不允许删除"}), 403

        # 2. 检查库是否存在于配置中
        if lib_id not in LIBRARY_CONFIG:
            return jsonify({"error": "找不到指定的库"}), 404
            
        lib_config = LIBRARY_CONFIG[lib_id]
        
        # === [新增] 密码校验 ===
        # 如果库有密码，且用户提供的密码不匹配，则拒绝删除
 
        if not check_permission(lib_config.get("password"), password):
            return jsonify({"error": "权限验证失败：密码错误"}), 403
        # ======================

        filename = lib_config.get("file")
        
        # 3. 物理删除 JSON 文件
        file_path = os.path.join(PUBLIC_DIR, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(PROJECT_ROOT, filename)
            
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f" [INFO] 已物理删除文件: {file_path}")
            except Exception as e:
                return jsonify({"error": f"文件删除失败: {str(e)}"}), 500

        # 4. 删除索引 (保持原有逻辑)
        index_path = os.path.join(FAISS_INDEX_BASE_DIR, f"index_{lib_id}")
        if os.path.exists(index_path):
            try:
                import shutil
                shutil.rmtree(index_path)
                if lib_id in VECTOR_STORES: del VECTOR_STORES[lib_id]
            except Exception as e: print(f" [WARNING] 索引删除失败: {e}")

        # 5. 从注册表中移除记录 (这是新增的重要步骤，防止残留)
        registry = load_registry()
        if filename in registry:
            del registry[filename]
            save_registry(registry)

        # 6. 从内存配置中移除
        del LIBRARY_CONFIG[lib_id]
        
        return jsonify({"message": f"库 {lib_id} 已成功删除", "success": True}), 200

    except Exception as e:
        print(f"ERROR in /api/delete_library: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/create_new_library', methods=['POST'])
def api_create_new_library_route():
    try:
        data = request.get_json()
        filename = data.get('filename')
        display_name = data.get('display_name')
        password = data.get('password') # <--- 获取密码，可以是空字符串或 None
        
        if not filename: return jsonify({"error": "Filename is required"}), 400
            
        safe_name = secure_filename(filename)
        if not safe_name.endswith('.json'): safe_name += '.json'
        
        if not display_name: display_name = safe_name.replace('.json', '')

        file_path = os.path.join(PUBLIC_DIR, safe_name)
        
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            # === [修改] 写入注册表 (支持密码) ===
            registry = load_registry()
            registry[safe_name] = {
                "name": display_name,
                "password": password if password and password.strip() else None
            }
            save_registry(registry)
            # ==================================

            # 立即更新内存配置
            lib_id = safe_name.replace('.json', '')
            LIBRARY_CONFIG[lib_id] = {
                "file": safe_name,
                "name": display_name,
                "password": password if password and password.strip() else None,
                "type": "scenario"
            }

            return jsonify({"message": "Library created", "filename": safe_name}), 201
        else:
            return jsonify({"message": "File already exists", "filename": safe_name}), 200

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_library_scenario', methods=['POST'])
def api_delete_library_scenario_route():
    """
    功能：从指定库中删除场景
    参数：library_id, scenario_id, password (如果库受保护)
    """
    try:
        data = request.get_json()
        lib_id = data.get('library_id')
        s_id = data.get('scenario_id')
        password = data.get('password')

        if not all([lib_id, s_id]):
            return jsonify({"error": "缺少 library_id 或 scenario_id"}), 400

        # 1. 检查库是否存在
        if lib_id not in LIBRARY_CONFIG:
            return jsonify({"error": "未知的 library_id"}), 404

        lib_config = LIBRARY_CONFIG[lib_id]

        # [修改] 校验密码（支持管理员）
        if not check_permission(lib_config.get("password"), password):
            return jsonify({"error": "权限验证失败：密码错误"}), 403

        # 3. 确定文件路径
        # 优先查找 PUBLIC_DIR，因为保存功能通常写在这里
        file_path = os.path.join(PUBLIC_DIR, lib_config["file"])
        if not os.path.exists(file_path):
            # 如果 public 下没有，尝试项目根目录或配置的原始路径
            file_path = os.path.join(os.path.dirname(__file__), lib_config["file"])
        
        if not os.path.exists(file_path):
            return jsonify({"error": "找不到对应的库文件"}), 404

        # 4. 读取并过滤数据
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
        except Exception:
            return jsonify({"error": "库文件损坏或无法读取"}), 500

        original_count = len(scenarios)
        # 过滤掉 ID 匹配的场景 (注意 ID 可能是数字或字符串，统一转字符比较)
        new_scenarios = [s for s in scenarios if str(s.get('id')) != str(s_id)]

        if len(new_scenarios) == original_count:
            return jsonify({"message": "未找到指定 ID 的场景，未执行删除。", "success": False}), 404

        # 5. 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_scenarios, f, ensure_ascii=False, indent=2)

        # 6. (可选) 同步更新内存缓存 LIBRARY_SCENARIO_CACHE
        global LIBRARY_SCENARIO_CACHE
        if str(s_id) in LIBRARY_SCENARIO_CACHE:
            del LIBRARY_SCENARIO_CACHE[str(s_id)]
            print(f" [INFO] 已从内存缓存中移除场景 {s_id}")

        # 注意：FAISS 向量库的实时删除比较复杂，这里暂时只处理了物理文件和简单缓存。
        # 建议前端在删除成功后提示用户刷新页面或重新加载模型。

        return jsonify({
            "message": f"场景 {s_id} 已从 {lib_config['name']} 删除成功。",
            "success": True
        }), 200

    except Exception as e:
        print(f"ERROR in /api/delete_library_scenario: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


@app.route('/api/export_scenario_package', methods=['POST'])
def api_export_scenario_package_route():
    try:
        data = request.get_json()
        xosc_content = data.get('openscenario_code')
        xodr_filename = data.get('opendrive_filename')
        scenario_name = data.get('scenario_name', 'scenario')

        if not xosc_content:
            return jsonify({"error": "Missing OpenSCENARIO code content."}), 400

        # --- Create hidden Tkinter root window ---
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Make dialog appear on top

        # Default filename
        default_name = f"{secure_filename(scenario_name)}_package.zip"

        # Open the Save Dialog (Blocking)
        print(" [EXPORT] Waiting for user to select path...")
        save_path = filedialog.asksaveasfilename(
            title="Select Save Location",
            initialfile=default_name,
            defaultextension=".zip",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )
        root.destroy()  # Cleanup

        # Check if user cancelled
        if not save_path:
            return jsonify({"status": "cancelled", "message": "User cancelled the save operation."}), 200

        print(f" [EXPORT] User selected save path: {save_path}")

        # Write the ZIP file locally
        try:
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. Write .xosc file
                # Fix relative path in xosc
                clean_xosc = re.sub(
                    r'filepath\s*=\s*["\']?([^"\']+\.xodr)["\']?',
                    lambda m: f'filepath="{os.path.basename(m.group(1))}"',
                    xosc_content,
                    flags=re.IGNORECASE
                )
                xosc_name_inside = f"{secure_filename(scenario_name)}.xosc"
                zf.writestr(xosc_name_inside, clean_xosc)

                # 2. Write .xodr map file
                # Handle PyInstaller frozen path vs Development path
                if getattr(sys, 'frozen', False):
                    base_dir = os.path.dirname(sys.executable)
                else:
                    base_dir = os.path.dirname(__file__)

                # Try to find the map file
                map_path = os.path.join(base_dir, "maps", xodr_filename)
                if not os.path.exists(map_path):
                    map_path = os.path.join(PREDEFINED_XODR_DIR, xodr_filename)

                if os.path.exists(map_path):
                    zf.write(map_path, arcname=os.path.basename(xodr_filename))
                else:
                    zf.writestr("MISSING_MAP_LOG.txt", f"Error: Map file '{xodr_filename}' not found at {map_path}")

            return jsonify({
                "status": "success",
                "message": "Export successful.",
                "saved_path": save_path
            }), 200

        except Exception as file_err:
            return jsonify({"error": f"Failed to write file locally: {str(file_err)}"}), 500

    except Exception as e:
        print(f"CRITICAL ERROR in /api/export_scenario_package: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server export logic failed: {str(e)}"}), 500

@app.route('/api/generate_structured_text', methods=['POST'])
def api_generate_structured_text_route():
    def _stream_payload(payload: Dict[str, Any]):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _event_stream():
        try:
            data = request.get_json(force=True)
            llm_config = get_llm_config_from_request(data)
            yield _stream_payload({"type": "status", "message": f"Connecting model {llm_config.get('model', '')}..."})

            logical_scenario_json = data.get('logical_scenario_json')
            if not logical_scenario_json:
                yield _stream_payload({"type": "error", "message": "请求中缺少 'logical_scenario_json' 数据。"})
                return

            received_id = str(logical_scenario_json.get('id', '')).strip()
            yield _stream_payload({"type": "status", "message": f"[API REQUEST] generate_structured_text. ID: '{received_id}'"})

            # 缓存查找逻辑
            if received_id and received_id in LIBRARY_SCENARIO_CACHE:
                cached_data = LIBRARY_SCENARIO_CACHE[received_id]
                if isinstance(cached_data, dict):
                    formatted_text = json.dumps(cached_data, indent=2, ensure_ascii=False)
                    map_key = cached_data.get('map_key', 'two-way-two-lane-straight-urban')
                    xodr_file = map_key if map_key.endswith('.xodr') else f"{map_key}.xodr"
                else:
                    formatted_text = str(cached_data)
                    xodr_file = extract_map_filename(formatted_text)
                if not xodr_file or xodr_file == 'unknown.xodr':
                    xodr_file = 'roundabout.xodr' if '环岛' in formatted_text else 'two-way-two-lane-straight-urban.xodr'
                matched_map_info = {"map_id": xodr_file.replace('.xodr', ''), "xodr_file": xodr_file}
                yield _stream_payload({
                    "type": "result", 
                    "payload": {
                        "message": f"已从库中加载场景 '{received_id}' (无AI生成)。", 
                        "structured_text": formatted_text, 
                        "matched_map_info": matched_map_info, 
                        "scenario_id": received_id
                    }
                })
                return
            yield _stream_payload({"type": "status", "message": f">>> [MISS] ID '{received_id}' 未在缓存中找到。开始 AI 生成流程..."})
            
            # 地图匹配逻辑
            matched_map_annotation = {}
            matched_source = 'unknown'
            target_map_key = None

            l1_selection = logical_scenario_json.get('l1_map_selection', {})
            if l1_selection and 'selected_map_key' in l1_selection:
                target_map_key = l1_selection.get('selected_map_key')
            if not target_map_key:
                layers = logical_scenario_json.get('layers', {})
                l1_text = layers.get('L1_Road', '')
                if l1_text:
                    target_map_key = str(l1_text).strip()
                    yield _stream_payload({"type": "status", "message": f"[COMPATIBILITY] 从 layers.L1_Road 提取到地图 Key: {target_map_key}"})

            if target_map_key:
                yield _stream_payload({"type": "status", "message": f"[MAP LOOKUP] 目标地图 Key: {target_map_key}"})
                if target_map_key in MAP_SEMANTICS_CACHE:
                    matched_map_annotation = MAP_SEMANTICS_CACHE[target_map_key]
                    matched_source = 'direct_key_lookup'
                    yield _stream_payload({"type": "status", "message": "[SUCCESS] 精确匹配成功。"})
                else:
                    normalized_target = target_map_key.lower().replace('_', '').replace('-', '').replace(' ', '')
                    best_match_key = None
                    for cached_key in MAP_SEMANTICS_CACHE.keys():
                        normalized_cache = cached_key.lower().replace('_', '').replace('-', '').replace(' ', '')
                        if normalized_target in normalized_cache or normalized_cache in normalized_target:
                            best_match_key = cached_key
                            break
                    if best_match_key:
                        matched_map_annotation = MAP_SEMANTICS_CACHE[best_match_key]
                        target_map_key = best_match_key
                        matched_source = f"fuzzy_match ({best_match_key})"
                        yield _stream_payload({"type": "status", "message": f"[SUCCESS] 模糊匹配成功: '{best_match_key}'"})
                    else:
                        yield _stream_payload({"type": "status", "message": f"[WARNING] Key '{target_map_key}' 模糊匹配也失败。"})

            if not matched_map_annotation:
                yield _stream_payload({"type": "status", "message": "[WARNING] Map matching completely failed. Using fallback from maps_carla cache."})

            # ================== [核心修改：拦截普通日志与节点摘要信息] ==================
            class StreamLogger:
                def __init__(self, q):
                    self.q = q
                    self.buffer = ""

                def write(self, msg):
                    self.buffer += msg
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            clean_line = line.strip()
                            if clean_line:
                                if clean_line.startswith("[NODE_SUMMARY]::"):
                                    self.q.put(clean_line)
                                else:
                                    self.q.put(f"[NORMAL_LOG]::{clean_line}")
                        self.buffer = lines[-1]

                def flush(self):
                    clean_line = self.buffer.strip()
                    if clean_line:
                        if clean_line.startswith("[NODE_SUMMARY]::"):
                            self.q.put(clean_line)
                        else:
                            self.q.put(f"[NORMAL_LOG]::{clean_line}")
                    self.buffer = ""
            log_queue = queue.Queue()
            logger = StreamLogger(log_queue)
            result_container = {}
            error_container = {}

            def target_task():
                try:
                    with contextlib.redirect_stdout(logger):
                        physical_dict = generate_physical_scenario(logical_scenario_json, matched_map_annotation, llm_config=llm_config)
                        result_container['data'] = physical_dict
                except Exception as e:
                    error_container['error'] = str(e)
                    print(f"Task Error: {traceback.format_exc()}")
                finally:
                    log_queue.put(None)

            task_thread = threading.Thread(target=target_task)
            task_thread.start()

            # ================== [修改：处理普通日志与 Node 摘要流] ==================
            while True:
                log_msg = log_queue.get()
                if log_msg is None:
                    break
                
                # 捕获我们在 refiner 里主动 print 的摘要内容
                if log_msg.startswith("[NODE_SUMMARY]::"):
                    summary_content = log_msg.split("::", 1)[1]
                    yield _stream_payload({"type": "node_summary", "content": summary_content})
                
                # 处理普通日志及步骤信令
                elif log_msg.startswith("[NORMAL_LOG]::"):
                    clean_msg = log_msg.split("::", 1)[1]
                    if clean_msg.startswith("[STEP_START]::"):
                        step = clean_msg.split("::")[1].strip()
                        yield _stream_payload({"type": "step_start", "step": step})
                    elif clean_msg.startswith("[STEP_END]::"):
                        step = clean_msg.split("::")[1].strip()
                        yield _stream_payload({"type": "step_end", "step": step})
                    elif "[Agent]" in clean_msg or "[DEBUG]" in clean_msg or "[Tool]" in clean_msg:
                        yield _stream_payload({"type": "log", "message": clean_msg})
            # ================================================================

            if 'error' in error_container:
                yield _stream_payload({"type": "error", "message": error_container['error']})
                return

            physical_dict = result_container.get('data', {})
            structured_text = json.dumps(physical_dict, ensure_ascii=False, indent=2)
            reasoning = {
                "environment_reasoning": physical_dict.get("environment_reasoning", ""),
                "duration_reasoning": physical_dict.get("duration_reasoning", ""),
                "base_entities_reasoning": physical_dict.get("base_entities_reasoning", ""),
                "final_entities_reasoning": physical_dict.get("final_entities_reasoning", ""),
            }
            
            yield _stream_payload({
                "type": "result", 
                "payload": {
                    "message": "场景理解完成", 
                    "structured_text": structured_text, 
                    "matched_map_info": matched_map_annotation, 
                    "physical_scenario": physical_dict, 
                    "reasoning": reasoning
                }
            })
            
        except Exception as e:
            print(f"CRITICAL ERROR in /api/generate_structured_text: {e}\n{traceback.format_exc()}")
            yield _stream_payload({"type": "error", "message": f"处理请求时发生意外的服务器错误: {str(e)}"})

    return Response(stream_with_context(_event_stream()), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/generate_code_from_text', methods=['POST'])
def api_generate_code_from_text_route():
    try:
        data = request.get_json()
        
        # [修改] 提取 LLM 配置
        llm_config = get_llm_config_from_request(data)
        print(f" [API] Code Gen Model: {llm_config['model']}")

        structured_text = data.get('structured_text')
        matched_map_info = data.get('matched_map_info')
        scenario_id = data.get('scenario_id') 
        
        openscenario_code = ""
        opendrive_code = "# ERROR: No xodr_file in matched map."
        is_library_file = False
        
        if not structured_text:
            return jsonify({"error": "请求中缺少 'structured_text' 数据。"}), 400
            
        print("\n--- [Step 2] Generating/Loading Final Code (.xosc) ---")
        
        def normalize_str(s):
            if not s: return ""
            if isinstance(s, dict): return "".join(json.dumps(s, ensure_ascii=False).split())
            return "".join(str(s).split())
        
        input_pure = normalize_str(structured_text)
        
        # 1. 尝试反向匹配 ID
        if not scenario_id:
            for cache_id, cache_data in LIBRARY_SCENARIO_CACHE.items():
                cache_str_norm = normalize_str(cache_data)
                if input_pure == cache_str_norm:
                    scenario_id = cache_id
                    break
            if not scenario_id:
                for cache_id, cache_data in LIBRARY_SCENARIO_CACHE.items():
                    cache_str_norm = normalize_str(cache_data)
                    if len(input_pure) > 50 and len(cache_str_norm) > 50:
                        if input_pure[:50] == cache_str_norm[:50]:
                            scenario_id = cache_id
                            break

        # 2. 根据 ID 加载或 AI 生成
        found_in_library = False
        library_data = None
        
        if scenario_id and scenario_id in LIBRARY_SCENARIO_CACHE:
            found_in_library = True
            library_data = LIBRARY_SCENARIO_CACHE[scenario_id]

        if found_in_library:
            target_filename = f"{scenario_id}.xosc"
            possible_paths = [
                os.path.join(PREDEFINED_XODR_DIR, target_filename), 
                os.path.join(XOSC_SAVE_DIR, target_filename),
                os.path.join(PROJECT_ROOT, "scenarios", target_filename)
            ]
            file_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f" [HIT] 成功加载本地物理场景文件: {path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        openscenario_code = f.read()
                    file_loaded = True
                    is_library_file = True
                    break
            
            if not file_loaded and isinstance(library_data, dict):
                print(f" [WARNING] 虽匹配到 ID 但无本地文件，将回退到 AI 生成。")
                is_library_file = False
        
        if not is_library_file:
             print(" [INFO] 开始调用 AI 生成 OpenSCENARIO 代码...")
             text_for_ai = structured_text
             if isinstance(structured_text, dict):
                 text_for_ai = json.dumps(structured_text, indent=2, ensure_ascii=False)
             
             # [修改] 传递 llm_config
             openscenario_code = generate_openscenario_code(
                 text_for_ai, 
                 llm_config=llm_config
             )
             
             if str(openscenario_code).strip().startswith("# ERROR:"):
                 return jsonify({"error": "生成OpenSCENARIO代码时失败。", "details": openscenario_code}), 500

        # 3. 读取 OpenDRIVE 地图代码
        if matched_map_info and 'xodr_file' in matched_map_info:
            xodr_filename = matched_map_info['xodr_file']
            if found_in_library and isinstance(library_data, dict):
                 map_key = library_data.get('map_key')
                 if map_key: 
                     if not map_key.endswith('.xodr'): xodr_filename = f"{map_key}.xodr"
                     else: xodr_filename = map_key
            xodr_path = os.path.join(PREDEFINED_XODR_DIR, xodr_filename)
            if os.path.isfile(xodr_path):
                with open(xodr_path, 'r', encoding='utf-8') as f: opendrive_code = f.read()
            else:
                found_xodr = False
                if os.path.exists(PREDEFINED_XODR_DIR):
                    for existing_file in os.listdir(PREDEFINED_XODR_DIR):
                        if existing_file.replace(" ", "").lower() == xodr_filename.replace(" ", "").replace("_", "").lower():
                            with open(os.path.join(PREDEFINED_XODR_DIR, existing_file), 'r', encoding='utf-8') as f:
                                opendrive_code = f.read()
                                matched_map_info['xodr_file'] = existing_file
                            found_xodr = True
                            break
                if not found_xodr: opendrive_code = f"# ERROR: Matched XODR file not found at: {xodr_path}"
        else:
            opendrive_code = "# ERROR: Matched XODR file not found on server."       
        
        extracted_params = {}
        if openscenario_code and not str(openscenario_code).startswith("# ERROR"):
            print(" [INFO] Extracting physics parameters from XOSC...")
            extracted_params = XOSCParamHandler.parse_and_extract(openscenario_code)

        return jsonify({
            "message": "已加载标准库文件 (本地)" if is_library_file else "场景代码已生成 (AI)",
            "openscenario_code": openscenario_code,
            "opendrive_code": opendrive_code,
            "physics_parameters": extracted_params
        }), 200

    except Exception as e:
        print(f"CRITICAL ERROR in /api/generate_code_from_text: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

# --- [新增] 场景转化接口 (Esmini to Carla) ---
@app.route('/api/convert_scenario', methods=['POST'])
def api_convert_scenario_route():
    try:
        data = request.get_json(force=True)
        original_xosc = data.get('original_xosc')
        target_simulator = str(data.get('target_simulator', 'Carla')).strip().lower()
        llm_config = get_llm_config_from_request(data)

        if not original_xosc:
            return jsonify({"error": "缺少需要转化的 OpenSCENARIO 内容。"}), 400
        if target_simulator != 'carla':
            return jsonify({"error": "当前仅支持 Carla 转化。"}), 400

        print(f" [API] Converting scenario to {target_simulator} ...")
        converted_xosc = convert_esmini_to_carla(original_xosc, llm_config=llm_config)
        if not converted_xosc or str(converted_xosc).startswith("# ERROR"):
            return jsonify({"error": "转化失败。", "details": converted_xosc}), 500

        return jsonify({
            "message": "场景转化成功。",
            "converted_xosc": converted_xosc,
            "target_simulator": "Carla"
        }), 200

    except Exception as e:
        print(f"CRITICAL ERROR in /api/convert_scenario: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"转化时发生错误: {str(e)}"}), 500

@app.route('/api/save_converted_xosc', methods=['POST'])
def api_save_converted_xosc_route():
    try:
        data = request.get_json()
        xosc_content = data.get('converted_xosc')
        default_name = data.get('default_filename', 'converted_carla.xosc')

        if not xosc_content:
            return jsonify({"error": "缺少需要保存的内容。"}), 400

        # 创建隐藏的 Tkinter 窗口，并将其置顶
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 确保对话框在浏览器最前端弹出

        print(" [SAVE] 等待用户选择保存路径...")
        # 弹出“另存为”对话框，阻塞等待用户操作
        save_path = filedialog.asksaveasfilename(
            title="Select Save Location (Carla OpenSCENARIO)",
            initialfile=default_name,
            defaultextension=".xosc",
            filetypes=[("OpenSCENARIO files", "*.xosc"), ("XML files", "*.xml"), ("All files", "*.*")]
        )
        root.destroy()  # 清理窗口

        # 检查用户是否点击了“取消”
        if not save_path:
            return jsonify({"status": "cancelled", "message": "用户取消了保存。"}), 200

        print(f" [SAVE] 用户选择的保存路径: {save_path}")

        # 将前端传来的代码直接写入用户选择的本地路径
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(xosc_content)

        return jsonify({
            "status": "success",
            "message": "文件保存成功。",
            "saved_path": save_path
        }), 200

    except Exception as e:
        print(f"CRITICAL ERROR in /api/save_converted_xosc: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"保存文件时发生错误: {str(e)}"}), 500

# --- [新增] Carla 场景运行 API ---
@app.route('/api/run_carla_simulation', methods=['POST'])
def api_run_carla_simulation_route():
    try:
        data = request.get_json(force=True)
        xosc_content = data.get('xosc_content')

        if not xosc_content:
            return jsonify({"error": "缺少需要运行的 OpenSCENARIO 内容。"}), 400

        # 定位 scenario_runner.py 脚本 (与 app.py 同级目录下的 scenario_runner 文件夹中)
        scenario_runner_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "scenario_runner"))
        scenario_runner_script = os.path.join(scenario_runner_dir, "scenario_runner.py")

        if not os.path.exists(scenario_runner_script):
            return jsonify({"error": f"未找到 scenario_runner.py 脚本: {scenario_runner_script}"}), 404

        import re

        # === 核心修复 1：自动将 Catalog 的相对路径修正为绝对正斜杠路径 ===
        catalog_abs_path = os.path.join(scenario_runner_dir, "srunner", "examples", "catalogs").replace('\\', '/')
        if not catalog_abs_path.endswith('/'):
            catalog_abs_path += '/'

        xosc_content = re.sub(
            r'<Directory\s+path=["\'][^"\']+["\']\s*/?>',
            f'<Directory path="{catalog_abs_path}"/>',
            xosc_content
        )

        # 直接保留前端传入的地图路径，仅统一为 XML 友好的正斜杠格式
        xosc_content = re.sub(
            r'<LogicFile\s+filepath=["\']([^"\']+)["\']\s*/?>',
            lambda match: f'<LogicFile filepath="{match.group(1).replace(chr(92), "/")}"/>',
            xosc_content,
            flags=re.IGNORECASE
        )

        # 1. 将前端传来的临时内容保存为一个实体文件供 Carla 读取
        temp_filename = f"temp_carla_run_{int(time.time())}.xosc"
        temp_filepath = os.path.abspath(os.path.join(XOSC_SAVE_DIR, temp_filename))

        with open(temp_filepath, 'w', encoding='utf-8') as f:
            f.write(xosc_content)

        # 3. 构造启动命令
        CARLA_PYTHON_PATH = os.getenv(
            "CARLA_PYTHON_PATH",
            r"D:\anaconda\envs\carla37\python.exe"
        )

        command = [
            CARLA_PYTHON_PATH,
            scenario_runner_script,
            "--openscenario", temp_filepath,
            "--host", "127.0.0.1",
            "--port", "4000",
            "--timeout", "2000.0"
        ]

        # 4. 异步启动 Carla 仿真测试
        cflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

        # === 终极修复：构建绝对安全的子进程环境 ===
        carla_env = os.environ.copy()

        # 强制注入底层解压所需的核心 Windows 变量 (防御性编程，彻底解决 KeyError)
        carla_env["windir"] = os.environ.get("windir", os.environ.get("WINDIR", r"C:\Windows"))
        carla_env["WINDIR"] = carla_env["windir"]
        carla_env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        carla_env["TEMP"] = os.environ.get("TEMP", os.environ.get("TMP", r"C:\Temp"))
        carla_env["TMP"] = carla_env["TEMP"]

        # 清理跨环境的 Python 干扰项
        carla_env.pop("PYTHONHOME", None)

        # 将 carla37 的路径置顶
        carla37_dir = os.path.dirname(CARLA_PYTHON_PATH)
        carla37_paths = f"{carla37_dir};{carla37_dir}\\Library\\mingw-w64\\bin;{carla37_dir}\\Library\\usr\\bin;{carla37_dir}\\Library\\bin;{carla37_dir}\\Scripts"
        carla_env["PATH"] = f"{carla37_paths};{carla_env.get('PATH', '')}"

        CARLA_ROOT_PATH = r"D:\CARLA_0.9.15\WindowsNoEditor"
        carla_env["CARLA_ROOT"] = CARLA_ROOT_PATH

        egg_file_name = "carla-0.9.15-py3.7-win-amd64.egg"
        carla_egg_path = os.path.join(CARLA_ROOT_PATH, "PythonAPI", "carla", "dist", egg_file_name)
        carla_agents_path = os.path.join(CARLA_ROOT_PATH, "PythonAPI", "carla")

        # 注入 Carla 专属的 PYTHONPATH
        carla_env["PYTHONPATH"] = f"{carla_egg_path};{carla_agents_path};{scenario_runner_dir}"

        # 5. 启动进程，捕获日志
        process = subprocess.Popen(
            command,
            cwd=scenario_runner_dir,
            creationflags=cflags,
            env=carla_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        print(f" [INFO] Carla scenario_runner initiated. PID: {process.pid}")
        print(f" [INFO] USING PYTHON: {CARLA_PYTHON_PATH}")
        print(f" [INFO] CARLA_ROOT: {carla_env['CARLA_ROOT']}")

        # 6. 后台日志读取
        def stream_carla_output(pipe):
            try:
                for line in pipe:
                    print(f"[CARLA-SR] {line.strip()}", flush=True)
            except Exception as e:
                print(f"[CARLA-SR ERROR] 日志读取失败: {e}")
            finally:
                pipe.close()

        import threading
        threading.Thread(target=stream_carla_output, args=(process.stdout,), daemon=True).start()

        return jsonify({
            "message": f"Carla 仿真指令已发送成功 (PID: {process.pid})！请在 Carla 窗口查看运行情况。",
            "temp_file": temp_filepath
        }), 200

    except Exception as e:
        print(f"CRITICAL ERROR in /api/run_carla_simulation: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"启动 Carla 仿真时发生错误: {str(e)}"}), 500

@app.route('/api/search_scenarios', methods=['POST'])
def api_search_scenarios_route():
    try:
        data = request.get_json()
        query = data.get('query')
        lib_id = data.get('library_id')
        k = data.get('k', 20)
        if not all([query, lib_id]):
            return jsonify({"error": "'query' 和 'library_id' 字段均为必填项。"}), 400
        vector_store = VECTOR_STORES.get(lib_id)
        if not vector_store:
            return jsonify({"results": [], "message": f"库 '{lib_id}' 的搜索服务不可用或索引为空。"}), 200
        
        docs = vector_store.similarity_search_with_score(query, k=k)
        
        dataset_label = "CIDAS"
        if lib_id == "cidas": dataset_label = "CIDAS"
        elif lib_id == "project": dataset_label = "Private"
        elif lib_id == "cda": dataset_label = "GB" 
            
        results = []
        for doc, score in docs:
            if score < 1.5:
                results.append({
                    "scenario_data": doc.metadata,
                    "relevance_score": round(1 - score / 2, 4),
                    "dataset_label": dataset_label
                })
        return jsonify({"results": results}), 200
    except Exception as e:
        print(f"ERROR in /api/search_scenarios: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器意外错误: {str(e)}"}), 500

@app.route('/api/search_triggers', methods=['POST'])
def api_search_triggers_route():
    try:
        data = request.get_json()
        query = data.get('query')
        lib_id = data.get('library_id')
        k = data.get('k', 30)
        if not all([query, lib_id]):
            return jsonify({"error": "'query' 和 'library_id' 字段均为必填项。"}), 400
        vector_store = VECTOR_STORES.get(lib_id)
        if not vector_store:
            return jsonify({"results": [], "message": f"触发条件库 '{lib_id}' 的搜索服务不可用或索引为空。"}), 200
        docs = vector_store.similarity_search_with_score(query, k=k)
        results = [{"data": doc.metadata, "relevance_score": round(1 - score / 2, 4)} for doc, score in docs if score < 1.5]
        return jsonify({"results": results}), 200
    except Exception as e:
        print(f"ERROR in /api/search_triggers: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器意外错误: {str(e)}"}), 500

@app.route('/api/save_selected_library_scenario', methods=['POST'])
def api_save_selected_library_scenario_route():
    try:
        data = request.get_json()
        scenario_to_save = data.get('scenario_to_save')
        target_lib_id = data.get('target_library_id')
        password = data.get('password')
        if not all([scenario_to_save, target_lib_id]):
            return jsonify({"error": "缺少 'scenario_to_save' 或 'target_library_id' 字段"}), 400
        try: LogicalScenarioOutput.model_validate(scenario_to_save) 
        except Exception as e_val: return jsonify({"error": f"场景数据结构无效: {e_val}"}), 400
        if target_lib_id not in LIBRARY_CONFIG:
            return jsonify({"error": "目标库ID无效"}), 400
        lib_config = LIBRARY_CONFIG[target_lib_id]
        if not check_permission(lib_config.get("password"), password):
            return jsonify({"error": "权限验证失败"})
        target_path = os.path.join(PUBLIC_DIR, secure_filename(lib_config["file"]))
        
        library_data = []
        if os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                try: library_data = json.load(f)
                except json.JSONDecodeError: library_data = []
        if not isinstance(library_data, list): library_data = []
        
        if 'id' not in scenario_to_save or not scenario_to_save['id']:
            scenario_to_save['id'] = f"SC-{int(time.time())}-{uuid.uuid4().hex[:4]}"
        
        library_data = [s for s in library_data if s.get('id') != scenario_to_save['id']]
        library_data.insert(0, scenario_to_save)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, ensure_ascii=False, indent=2)
        
        vector_store = VECTOR_STORES.get(target_lib_id)
        doc_text = create_document_from_scenario(scenario_to_save)
        
        if vector_store:
            vector_store.add_texts(texts=[doc_text], metadatas=[scenario_to_save], ids=[str(scenario_to_save['id'])])
        elif EMBEDDINGS:
            vector_store = FAISS.from_texts(texts=[doc_text], embedding=EMBEDDINGS, metadatas=[scenario_to_save], ids=[str(scenario_to_save['id'])])
            VECTOR_STORES[target_lib_id] = vector_store
        else:
            return jsonify({"message": f"场景已保存至 {lib_config['file']}，但搜索索引更新失败 (模型未加载)。"}), 207
        
        index_path = os.path.join(FAISS_INDEX_BASE_DIR, f"index_{target_lib_id}")
        try:
            vector_store.save_local(index_path)
        except Exception as e_save:
            return jsonify({"message": f"场景已保存至 {lib_config['file']}，但索引文件写入失败。"}), 207
        return jsonify({"message": f"场景已成功保存至 {lib_config['name']} 并已更新搜索索引。"}), 201
    except Exception as e:
        print(f"ERROR in /api/save_selected_library_scenario: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器意外错误: {str(e)}"}), 500

@app.route('/api/generate_dangerous_scenario', methods=['POST'])
def api_generate_dangerous_scenario_route():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Request body is empty."}), 400
        
        # [修改] 提取 LLM 配置
        llm_config = get_llm_config_from_request(data)
        print(f" [API] Scenario Gen Model: {llm_config['model']}")
        
        test_description = data.get('test_function_description')
        selected_l1_key = data.get('selected_l1_road_type_key')
        if not test_description: return jsonify({"error": "'test_function_description' is required."}), 400
        
        user_hints = data.get('layer_specific_inputs', {})
        l1_fallback_desc = user_hints.get('L1', '').strip()
        if not selected_l1_key and not l1_fallback_desc:
            selected_l1_key = "generic"
        elif not selected_l1_key and l1_fallback_desc:
            selected_l1_key = l1_fallback_desc[:20].replace(' ', '*') or "user_defined"
            
        generation_hint_parts = []
        if user_hints:
            generation_hint_parts.append("**用户提供的特定层级输入:**\n" + json.dumps(user_hints, ensure_ascii=False, indent=2))
            
        retrieved_scenarios_for_response = []
        all_retrieved_docs_with_source = []
        enhanced_query = test_description
        if l1_fallback_desc: enhanced_query += f" [道路环境: {l1_fallback_desc}]"
        
        # 1. 检索逻辑
        ALLOWED_RAG_LIBS = ["project", "cda"] 
        for lib_id, config in LIBRARY_CONFIG.items():
            if lib_id not in ALLOWED_RAG_LIBS:
                continue
            
            if config.get('type') == 'scenario':
                vector_store = VECTOR_STORES.get(lib_id)
                if vector_store:
                    try:
                        docs_with_scores = vector_store.similarity_search_with_score(enhanced_query, k=15)
                        for doc, score in docs_with_scores:
                            if score < 3.0:
                                all_retrieved_docs_with_source.append({
                                    "doc": doc, 
                                    "score": score, 
                                    "lib_id": lib_id, 
                                    "lib_name": config.get('name', lib_id)
                                })
                    except Exception as e_rag_lib: 
                        print(f"Error searching {lib_id}: {e_rag_lib}")
        
        all_retrieved_docs_with_source.sort(key=lambda x: x['score'])
        top_retrieved_docs = all_retrieved_docs_with_source[:10]
        
        context_parts = []
        for item in top_retrieved_docs:
            meta = item['doc'].metadata
            meta['library_id'] = item['lib_id']
            real_name = (
                meta.get('name') or 
                meta.get('scenario_name') or 
                meta.get('title') or 
                meta.get('Name') or 
                meta.get('id') or 
                "未命名场景"
            )
            meta['name'] = real_name
            
            retrieved_scenarios_for_response.append({"scenario_data": meta, "source": "library"})
            context_parts.append(f"参考示例:\n 名称: {real_name}\n 描述: {meta.get('description', '无描述')}")

        context_parts = context_parts[:6]
        if context_parts:
            generation_hint_parts.append(f"\n**系统检索到的相似场景:**\n" + "\n".join(context_parts))
        if SCENARIO_RULES:
            generation_hint_parts.append("\n**场景设计规则:**\n" + SCENARIO_RULES)
            
        final_generation_hint = "\n\n".join(generation_hint_parts)

        # 2. AI 生成逻辑 (传入 llm_config)
        requested_num_to_generate = data.get('num_to_generate', 4)
        try:
            num_to_generate = int(requested_num_to_generate)
        except (TypeError, ValueError):
            return jsonify({"error": "'num_to_generate' 必须是整数。"}), 400

        if num_to_generate < 1:
            return jsonify({"error": "'num_to_generate' 必须大于等于 1。"}), 400
        if num_to_generate > 20:
            return jsonify({"error": "'num_to_generate' 不能超过 20。"}), 400

        def event_stream():
            generated_items: List[Dict[str, Any]] = []
            progress_queue = queue.Queue()

            def stream_event(payload: Dict[str, Any]):
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            def on_progress(completed: int, total: int, success: bool):
                progress_queue.put({
                    "type": "progress",
                    "completed": completed,
                    "total": total,
                    "success": success
                })

            def run_generation():
                try:
                    result = generate_multiple_dangerous_scenarios(
                        test_function_description=test_description,
                        num_to_generate=num_to_generate,
                        user_provided_generation_hint=final_generation_hint.strip(),
                        selected_l1_road_type_key=selected_l1_key,
                        llm_config=llm_config,
                        progress_callback=on_progress
                    )
                    generated_items.extend(result)
                    progress_queue.put({"type": "done"})
                except Exception as e_inner:
                    progress_queue.put({"type": "error", "message": str(e_inner)})

            threading.Thread(target=run_generation, daemon=True).start()

            while True:
                item = progress_queue.get()
                if item["type"] == "progress":
                    yield stream_event(item)
                elif item["type"] == "error":
                    yield stream_event({"type": "error", "message": item["message"]})
                    return
                elif item["type"] == "done":
                    break

            generated_for_response = [{"scenario_data": s, "source": "ai_generated"} for s in generated_items if s]
            all_scenarios = retrieved_scenarios_for_response + generated_for_response
            if not all_scenarios:
                yield stream_event({"type": "error", "message": "未能生成任何场景。"})
                return
            yield stream_event({"type": "result", "payload": {"scenarios": all_scenarios}})

        return Response(stream_with_context(event_stream()), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    except Exception as e:
        print(f"ERROR in /api/generate_dangerous_scenario: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器意外错误: {str(e)}"}), 500

@app.route('/api/refine_library_scenario_draft', methods=['POST'])
def api_refine_library_scenario_draft_route():
    try:
        data = request.get_json()
        
        # [修改] 提取 LLM 配置
        llm_config = get_llm_config_from_request(data)
        
        user_draft = data.get('user_draft')
        if not user_draft: return jsonify({"error": "'user_draft' is required."}), 400
        
        similar_scenarios_context = "无相似场景可参考。"
        all_docs = []
        ALLOWED_RAG_LIBS = ["project", "cda"] 
        for lib_id, config in LIBRARY_CONFIG.items():
            if lib_id not in ALLOWED_RAG_LIBS:
                continue
            if config.get('type') == 'scenario' and VECTOR_STORES.get(lib_id):
                try:
                    docs = VECTOR_STORES[lib_id].similarity_search_with_score(user_draft, k=3)
                    for d, s in docs: all_docs.append((d, s, config.get('name')))
                except: pass
        all_docs.sort(key=lambda x: x[1])
        parts = [f"参考示例 (来自: {lib}):\n名称: {d.metadata.get('name')}\n描述: {d.metadata.get('description')}" for d, s, lib in all_docs[:3] if s < 1.2]
        if parts: similar_scenarios_context = "\n".join(parts)
        
        # 传入 llm_config
        refined_scenario = refine_scenario_for_library(
            user_draft, 
            similar_scenarios_context, 
            SCENARIO_RULES,
            llm_config=llm_config
        )
        if not refined_scenario: return jsonify({"error": "AI未能成功精炼您的草稿。"}), 500
        return jsonify({"refined_scenario": refined_scenario}), 200
    except Exception as e:
        print(f"ERROR in /api/refine_library_scenario_draft: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器意外错误: {str(e)}"}), 500

@app.route('/api/get_predefined_xodr_options', methods=['GET'])
def api_get_predefined_xodr_options_route():
    options = []
    try:
        if os.path.exists(PREDEFINED_XODR_DIR):
            for filename in sorted(os.listdir(PREDEFINED_XODR_DIR)):
                if filename.lower().endswith(".xodr"):
                    key = os.path.splitext(filename)[0]
                    label = f"{key.replace('*', ' ').title()} ({filename})"
                    options.append({"value": key, "label": label})
            return jsonify(options), 200
        else: return jsonify({"error": "Predefined XODR directory not found."}), 404
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/get_predefined_xodr_description/<string:xodr_key>', methods=['GET'])
def api_get_predefined_xodr_description_route(xodr_key: str):
    safe_key = secure_filename(xodr_key)
    desc_path = os.path.join(PREDEFINED_XODR_DIR, f"{safe_key}.description.txt")
    if os.path.isfile(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f: description_content = f.read()
        return jsonify({"description": description_content}), 200
    else: return jsonify({"description": f"（信息：未找到地图 '{xodr_key}' 的文字说明。）"}), 200

@app.route('/api/save_simulation_files', methods=['POST'])
def api_save_simulation_files_route():
    data = request.get_json()
    xosc_code = data.get('openscenario_code')
    scenario_name_input = data.get('scenario_name', 'unnamed')
    
    modified_params = data.get('modified_params', {}) 
    
    if not xosc_code:
        return jsonify({"error": "缺少 OpenSCENARIO 内容"}), 400
    
    try:
        if modified_params:
            print(f" [INFO] 应用物理参数修改: {modified_params}")
            xosc_code = XOSCParamHandler.update_xml_params(xosc_code, modified_params)
        
        safe_name_ascii = re.sub(r'[^\x00-\x7F]+', '', scenario_name_input)
        safe_name = re.sub(r'[^\w\s-]', '', safe_name_ascii).strip().replace(' ', '_') or "scenario"
        timestamp = time.strftime('%Y%m%d_%H%M%S') 
        
        xosc_filename = f"{safe_name}_{timestamp}.xosc"
        xosc_full_path = os.path.join(XOSC_SAVE_DIR, xosc_filename)
        
        def clean_map_path_callback(match):
            full_path_str = match.group(1)
            filename_only = full_path_str.replace("\\", "/").split("/")[-1]
            return f'filepath="../maps/{filename_only}"'

        xosc_code = re.sub(
            r'filepath\s*=\s*["\']?([^"\'>]+\.xodr)["\']?',
            clean_map_path_callback,
            xosc_code,
            flags=re.IGNORECASE
        )
        
        with open(xosc_full_path, 'w', encoding='utf-8') as f:
            f.write(xosc_code)
        
        print(f"SUCCESS: Saved to {xosc_full_path}")
        
        return jsonify({
            "message": "场景保存成功（参数已更新）。",
            "save_path": XOSC_SAVE_DIR,
            "saved_files": {'xosc_1_0_path_on_server': xosc_full_path}
        }), 200
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/run_simulation', methods=['POST'])
def api_run_simulation_route():
    if not ESMINI_AVAILABLE: return jsonify({"error": "esmini executable not configured."}), 503
    
    data = request.get_json()
    xosc_path = data.get('xosc_1_0_file_path_for_sim')
    
    if not xosc_path: return jsonify({"error": "Missing XOSC file path."}), 400
    abs_xosc_path = os.path.abspath(xosc_path)
    abs_sim_dir = os.path.abspath(SIMULATIONS_SAVE_DIR)
    abs_xosc_dir = os.path.abspath(XOSC_SAVE_DIR)
    is_safe_path = abs_xosc_path.startswith(abs_sim_dir) or abs_xosc_path.startswith(abs_xosc_dir)
    
    if not is_safe_path:
        print(f"Security blocked: {abs_xosc_path}")
        return jsonify({"error": "Invalid XOSC file path (access denied)."}), 403
        
    if not os.path.isfile(abs_xosc_path): return jsonify({"error": "XOSC file not found."}), 404
    command = [
        ESMINI_EXECUTABLE_PATH,
        "--window", "60", "60", "800", "400",
        "--osc", abs_xosc_path 
    ]
    
    try:
        cflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen(command, cwd=ESMINI_WORKING_DIR, creationflags=cflags)
        return jsonify({"message": f"esmini process (PID: {process.pid}) initiated."}), 200
    except Exception as e: return jsonify({"error": f"Server error running simulation: {str(e)}"}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        if not os.path.exists(os.path.join(app.static_folder, 'index.html')):
             return jsonify({"error": "Frontend entry point 'index.html' not found in public directory."}), 404
        return send_from_directory(app.static_folder, 'index.html')

# --- [新增] 重建索引的 API ---
@app.route('/api/rebuild_indices', methods=['POST'])
def api_rebuild_indices_route():
    global VECTOR_STORES
    try:
        data = request.get_json()
        password = data.get('password')
        
        # 简单校验管理员权限
        if password != ADMIN_MASTER_PASSWORD:
             return jsonify({"error": "权限拒绝：需要管理员密码"}), 403

        print("\n=== [API] 开始触发全量索引重建 (热更新) ===")
        
        # 确保 Embedding 模型已加载
        if not EMBEDDINGS:
             return jsonify({"error": "Embedding 模型未加载，无法构建索引。"}), 500

        rebuilt_count = 0
        
        # 遍历当前内存中所有的库配置 (LIBRARY_CONFIG 包含动态添加的库)
        for lib_id, config in LIBRARY_CONFIG.items():
            # 跳过地图库，只处理场景和Trigger
            if config.get('type') not in ['scenario', 'trigger']:
                continue

            print(f" -> 正在处理库: {config['name']} ({lib_id})...")
            
            # 1. 读取 JSON 数据
            file_path = os.path.join(PUBLIC_DIR, config['file'])
            if not os.path.exists(file_path):
                # 尝试备用路径
                file_path = os.path.join(PROJECT_ROOT, config['file'])
            
            if not os.path.exists(file_path):
                print(f"    [WARN] 文件不存在，跳过: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                try: items = json.load(f)
                except: items = []
            
            if not items:
                print("    [WARN] 数据为空，跳过。")
                continue

            # 2. 生成文档文本
            texts = []
            metadatas = []
            ids = []
            
            for item in items:
                if not isinstance(item, dict): continue
                
                # 根据类型生成不同的描述文本
                if config['type'] == 'trigger':
                    # 这里把 build_all_indices.py 里的逻辑搬过来，或者简化
                    # 为简单起见，这里复用 item 的 name 和 description
                    doc_text = f"触发条件: {item.get('name','')}. 描述: {item.get('description','')}"
                    # 如果您导入了 create_document_from_trigger 函数，可以用那个
                else:
                    # 场景类型：调用 scenario_generator 里现有的函数
                    doc_text = create_document_from_scenario(item)
                
                texts.append(doc_text)
                metadatas.append(item) # 存入完整元数据
                ids.append(str(item.get('id', uuid.uuid4().hex)))

            # 3. 构建 FAISS 索引
            if texts:
                print(f"    [INFO] 生成 {len(texts)} 条向量...")
                vector_store = FAISS.from_texts(texts=texts, embedding=EMBEDDINGS, metadatas=metadatas, ids=ids)
                
                # 4. 保存到磁盘
                index_save_path = os.path.join(FAISS_INDEX_BASE_DIR, f"index_{lib_id}")
                vector_store.save_local(index_save_path)
                
                # 5. [关键] 更新内存中的 VECTOR_STORES，实现热更新
                VECTOR_STORES[lib_id] = vector_store
                rebuilt_count += 1
        
        print(f"=== 索引重建完成。共更新 {rebuilt_count} 个库。 ===\n")
        return jsonify({"message": f"成功重建 {rebuilt_count} 个库的索引，搜索已更新。", "success": True}), 200

    except Exception as e:
        print(f"ERROR in /api/rebuild_indices: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"构建失败: {str(e)}"}), 500

if __name__ == '__main__':
    initialize_llms()
    initialize_rag_components()
    print("--- Flask Application Starting ---")
    print(f"--- Serving frontend from: {PUBLIC_DIR} ---")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)