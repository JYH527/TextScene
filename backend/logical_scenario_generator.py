import json
import os
import traceback
import time
import re
import math
import random
from typing import List, Optional, Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field


# ========================== 1. Paths and Configuration ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "bge-m3")
TOWN_SEGMENTS_PATH = os.path.join(SCRIPT_DIR, "maps", "map_segments", "all_Town_maps.json")
FAISS_INDEX_BASE_DIR = os.path.join(SCRIPT_DIR, "faiss_indices")
MAPS_CARLA_INDEX_PATH = os.path.join(FAISS_INDEX_BASE_DIR, "index_maps_carla")

MAP_INVENTORY_CONTEXT = """
[Available Map Classes (Map Inventory)]
You must choose the `map_class` that best matches the test requirement from the following list (do not invent new classes):

- T-junction
- Straight-Urban-Road
- Straight-Highway-Road
- Curve-Urban-Road
- Curve-Highway-Road
- Ramp
- Roundout
- Crossroad
- Complex-Junction
"""

# ========================== 2. Data Models ==========================
class MapDescription(BaseModel):
    selected_map_class: str = Field(description="A concise description of the map topology")
    key_words: List[str] = Field(
        description="A group of labels describing the map's core features. You must select at least one item from each of the following 5 dimensions:\n"
                    "1. Road shape: [Straight, Curve]. If any Curve segment exists, choose Curve.\n"
                    "2. Slope feature: [Slope, Flat]. If any road segment has a non-zero slope, choose Slope.\n"
                    "3. Facility feature: [Parking, ParkingAreas, NoParking]. If parkingAreas exists choose ParkingAreas; if parking exists choose Parking; otherwise choose NoParking.\n"
                    "4. Crosswalk: [Crosswalk, NoCrosswalk]. If crosswalks exist choose Crosswalk; otherwise choose NoCrosswalk.\n"
                    "5. Lane capacity: [1-Lane-Per-Dir, Muti-Lane-Per-Dir]. Check the driving lanes. If every one-way road has only one driving lane (e.g. driving:[-1, 1]), you must choose [1-Lane-Per-Dir]. If any one-way road has multiple driving lanes (e.g. driving:[-2, -1]), you must choose [Muti-Lane-Per-Dir].\n"
    )
    core_topology: str = Field(
        description="Describe the basic topology of the intersection or road. Example: This map is a typical T-junction connecting roads from the east, west, and south."
    )
    lane_and_direction: str = Field(
        description="Describe traffic direction and lane distribution. Example: Both the main road and side roads are two-way two-lane roads, with bidirectional driving lanes, shoulders, sidewalks, and parking lanes."
    )
    environment_and_geometry: str = Field(
        description="Describe the environment (urban/highway) and the geometric terrain features (ramp/curve/flat). Example: This is an urban road environment with generally flat terrain. The outer roads are straight, while the intersection interior is connected by multiple flat curved links."
    )
class LogicalScenarioGuideline(BaseModel):
    """
    Phase 1: Rough scene logic and map structure conception
    """
    scenario_name: str = Field(description="A short scenario name. Example: Left Vehicle Overtaking and Cutting in Followed by Emergency Braking")
    description: str = Field(description="A concise description of the scene logic. Example: The target vehicle overtakes the ego vehicle from the left lane, cuts in front of the ego vehicle, and immediately performs emergency braking, testing the ego vehicle's ability to respond to an emergency cut-in by a preceding vehicle.")
    l1_map_selection: MapDescription = Field(description="Map requirements description corresponding to the test scenario")


class ScenarioAnnotations(BaseModel):
    scenario_type: str = Field(description="Scenario category, e.g. Cut-in, CarFollowing", default="Unknown")
    risk_level: str = Field(description="Risk level, e.g. High, Medium, Low", default="High")
    expected_function: str = Field(description="Expected autonomous driving function under test, e.g. AEB, ACC", default="Unknown")
    weather: str = Field(description="Weather condition, e.g. Clear, Rain", default="Clear")
class ScenarioLayers(BaseModel):
    """
    Hierarchical scene description based on the scenario library elements table (Table 1).
    Note: Do not include any concrete coordinate values, road IDs, or lane IDs; use relative semantic descriptions only.
    """
    L0_EgoInternal: str = Field(
        description="Only describe the initial ego vehicle state. You must explicitly include the following elements:\n"
                    "1. Type (e.g. car, truck, van)\n"
                    "2. Position (e.g. middle lane, right lane, left lane, shoulder lane, parking lane, etc.)\n"
                    "3. Global behavior (e.g. go forward, turn left, turn right, stop, etc.)\n"
                    "Example: The ego vehicle (car) is initially in the right lane and going forward."
    )
    L3_TemporaryChanges: str = Field(
        description="Temporary static road changes. If present, you must explicitly include the following elements (use None if absent):\n"
                    "1. Type (e.g. cone barrel, warning sign, warning bucket, etc.)\n"
                    "2. Position (initial position relative to the ego vehicle: front, left, right)\n"
                    "Example: Cone barrels are placed in the front lane ahead of the ego vehicle.",
        default="No construction or temporary obstacles."
    )
    L4_DynamicObjects: str = Field(
        description="Dynamic traffic participants other than the ego vehicle and their interaction behavior (the ego vehicle keeps the initial L0 state unchanged). You must explicitly include the following for each participant:\n"
                    "1. Type (e.g. car, truck, van, pedestrian, etc.)\n"
                    "2. Position (initial position relative to the ego vehicle: front, left, right)\n"
                    "3. Oracle / interaction behavior (longitudinal actions such as yield/accelerate/decelerate/stop, lateral actions such as keep/change lane)\n"
                    "Example: A target vehicle (car) is initially positioned to the left behind the ego vehicle. It accelerates (longitudinally) to overtake, performs a lane change (lateral) maneuver to the right, cuts directly in front of the ego vehicle, and immediately decelerates (longitudinally) to a complete stop."
    )
    L5_Environment: str = Field(
        description="Environment and weather description. You must explicitly include the following elements:\n"
                    "1. Weather: Type (sunny, rainy, snowy) and density (strong, medium, weak)\n"
                    "2. Time: (daytime, nighttime, morning)\n"
                    "Example: Strong rainy weather during daytime."
    )


class LogicalScenarioOutput(BaseModel):
    """
    Phase 2: Final standardized scene structure
    """
    id: Optional[str] = None
    scenario_name: str = Field(description="A short scenario name. Example: Left Vehicle Overtaking and Cutting in Followed by Emergency Braking")
    description: str = Field(description="A concise description of the scene logic. Example: The target vehicle overtakes the ego vehicle from the left lane, cuts in front of the ego vehicle, and immediately performs emergency braking, testing the ego vehicle's ability to respond to an emergency cut-in by a preceding vehicle.")
    annotations: ScenarioAnnotations
    layers: ScenarioLayers


# ========================== 3. Prompt Definitions ==========================
guideline_prompt_template_str = """
You are a world-class autonomous driving test scenario logic designer.
Your task is to generate a `LogicalScenarioGuideline` based on the user's test function requirements. This entails defining high-level scene logic and mapping requirements.

[Input Context]
- Test function description: {test_function_description}
- Additional generation hint: {generation_hint}

[Map Inventory Context] 
{map_inventory_context}

[Reasoning and Constraints]
1. Scenario Definition: Provide a clear, one-sentence scene overview and a concise scenario name.
2. Map Selection (`l1_map_selection` rules):
   - `selected_map_class` MUST be strictly chosen from the provided Map Inventory. Do not hallucinate classes.
   - `key_words` MUST comprehensively cover both the map class and its core geometric/facility features to enable accurate hard-filtering.
   - `core_topology`, `lane_and_direction`, and `environment_and_geometry` MUST be semantically clear, detailed, and complete to facilitate downstream semantic similarity matching.

⚠️ [Format Requirements] ⚠️:
   - Output only the final JSON result.
   - Do not include any reasoning text, markdown fences, or extra commentary.
   - Your output must strictly match the expected JSON schema.
{format_instructions}
"""

final_prompt_template_str = """
You are an expert autonomous driving test scenario construction engineer.
Your task is to synthesize the final standardized `LogicalScenarioOutput` by combining the reference scene logic, user requirements, and the matched map annotations.

[Input Context]
- User Test Requirements: {test_function_description}
- LogicalScenarioGuideline (Reference logic): {guideline_json}
- Scene Differentiation Requirements: {generation_hint}
- Matched Map Annotations (best_map): {best_map}

[Reasoning and Constraints]
1. Abstract Semantic Constraint (CRITICAL RED LINE): 
   - You MUST NOT use any concrete numbers, road IDs, or lane IDs in the `layers`. 
   - Use ONLY relative, abstract semantics (e.g., "right lane", "front", "behind").
2. Mandatory Layer Specifications:
   - `L0_EgoInternal`: Must explicitly define Ego's Type (e.g., car/truck), Position (e.g., left lane), and Global Behavior (e.g., go forward).
   - `L3_TemporaryChanges`: Describe static obstacles (if any) with Type (e.g., traffic cone) and Position (e.g., front/left/right). State clearly if absent.
   - `L4_DynamicObjects`: Describe dynamic NPCs. For EACH participant, you must specify: Type, Initial Position relative to Ego, and subsequent Oracle trajectory (longitudinal actions like accelerate/brake; lateral actions like lane change).
     *Note: During the interaction, the Ego vehicle remains in its initial L0 state unchanged.*
   - `L5_Environment`: Must detail Weather (type and intensity, e.g., heavy rain) and Time of day (e.g., daytime, nighttime).
3. Diversity Control: You MUST strictly adhere to the [Scene Differentiation Requirements] to guarantee diversity among generated variants.

⚠️ [Format Requirements] ⚠️:
   - Output only the final JSON result.
   - Do not include any reasoning text, markdown fences, or extra commentary.
   - Your output must strictly match the expected JSON schema.
{format_instructions}
"""

refinement_prompt_template_str = """
You are a scene normalization expert.
Transform the user scene draft into a structured `LogicalScenarioOutput`.

--- User Draft ---
{user_draft}

--- Retrieval Context ---
{similar_scenarios_context}

--- Rules ---
{scenario_rules}

Please output data that conforms to the JSON Schema.
{format_instructions}
"""

# ========================== 4. Global Instances ==========================
CURRENT_LLM_CONFIG: Dict[str, Any] = {}
LLM_INSTANCE = None
EMBEDDING_MODEL = None
TOWN01_SEGMENTS_CACHE: Optional[Dict[str, Any]] = None
MAPS_CARLA_VECTORSTORE = None
MAPS_CARLA_VECTOR_BY_KEY: Dict[str, List[float]] = {}

GUIDELINE_PROMPT_TEMPLATE = None
FINAL_PROMPT_TEMPLATE = None
REFINE_PROMPT_TEMPLATE = None

GUIDELINE_PARSER = None
FINAL_OUTPUT_PARSER = None


# ========================== 5. Utility Functions ==========================

def _clean_json_from_response(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("LLM output is empty")

    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    json_match = re.search(r'```json(.*?)```', cleaned_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    elif '```' in cleaned_text:
        json_str = cleaned_text.replace('```', '').strip()
    else:
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_text[start_idx: end_idx + 1]
        else:
            json_str = cleaned_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json.loads(json_str)


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    if n1 <= 0 or n2 <= 0:
        return -1.0
    return dot / (math.sqrt(n1) * math.sqrt(n2))


def _normalize_map_class(map_class: str) -> str:
    if not map_class:
        return ""
    s = map_class.strip().lower().replace('_', '-').replace(' ', '-')
    mapping = {
        "t-junction": "T-junction",
        "t-junctions": "T-junction",
        "straight-urban-road": "Straight-Urban-Road",
        "straight-highway-road": "Straight-Highway-Road",
        "curve-urban-road": "Curve-Urban-Road",
        "curve-highway-road": "Curve-Highway-Road",
        "ramp": "Ramp",
        "roundout": "Roundout",
        "crossroad": "Crossroad",
        "complex-junction": "Complex-Junction",
    }
    return mapping.get(s, map_class)


def _extract_class_from_map_key(map_key: str) -> str:
    # Example: Town01_T-junction_01 -> T-junction
    parts = map_key.split('_')
    if len(parts) >= 3:
        return _normalize_map_class(parts[1])
    return ""


def _build_map_keywords(map_obj: Dict[str, Any]) -> List[str]:
    """
    Directly read `map_description.key_words` for each scene in the map segment JSON.
    Return an empty list if the field is missing or malformed.
    """
    map_description = map_obj.get("map_description", {})
    key_words = map_description.get("key_words", []) if isinstance(map_description, dict) else []

    if isinstance(key_words, list):
        return [str(k).strip() for k in key_words if str(k).strip()]

    return []


def _build_map_semantic_text(map_key: str, map_obj: Dict[str, Any]) -> str:
    """
    Build `map_text` using only the high-level semantic fields in `map_description`,
    which will be used to compute similarity against the guideline's semantic description.
    """
    _ = map_key  # Keep the parameter to preserve backward-compatible call signatures

    map_description = map_obj.get("map_description", {}) if isinstance(map_obj, dict) else {}
    if not isinstance(map_description, dict):
        map_description = {}

    lane_and_direction = str(map_description.get("lane_and_direction", "")).strip()
    environment_and_geometry = str(map_description.get("environment_and_geometry", "")).strip()

    return (
        f"lane_and_direction: {lane_and_direction}. "
        f"environment_and_geometry: {environment_and_geometry}. "
    )


def _build_guideline_semantic_query(guideline: LogicalScenarioGuideline) -> str:
    m = guideline.l1_map_selection
    return (
        f"lane_and_direction: {m.lane_and_direction}. "
        f"environment_and_geometry: {m.environment_and_geometry}. "
    )


def _load_map_segments_once() -> Dict[str, Any]:
    global TOWN01_SEGMENTS_CACHE
    if TOWN01_SEGMENTS_CACHE is not None:
        return TOWN01_SEGMENTS_CACHE

    if not os.path.exists(TOWN_SEGMENTS_PATH):
        raise FileNotFoundError(f"Map segment file does not exist: {TOWN_SEGMENTS_PATH}")

    with open(TOWN_SEGMENTS_PATH, "r", encoding="utf-8") as f:
        TOWN01_SEGMENTS_CACHE = json.load(f)
    return TOWN01_SEGMENTS_CACHE


def _initialize_embedding_model_if_needed() -> bool:
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is not None:
        return True

    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"[MapMatch] Local embedding model not found: {LOCAL_MODEL_PATH}")
        return False

    try:
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
        print(f"[MapMatch] Loaded local embedding model: {LOCAL_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[MapMatch] Failed to load local embedding model: {e}")
        traceback.print_exc()
        EMBEDDING_MODEL = None
        return False


def _initialize_maps_carla_vectors_if_needed() -> bool:
    """
    Load the offline-built `maps_carla` FAISS index and extract all `map_key -> vector` entries at once.
    """
    global MAPS_CARLA_VECTORSTORE, MAPS_CARLA_VECTOR_BY_KEY

    if MAPS_CARLA_VECTORSTORE is not None and MAPS_CARLA_VECTOR_BY_KEY:
        return True

    if not _initialize_embedding_model_if_needed() or EMBEDDING_MODEL is None:
        return False

    if not os.path.exists(MAPS_CARLA_INDEX_PATH):
        print(f"[MapMatch] `maps_carla` index not found: {MAPS_CARLA_INDEX_PATH}")
        return False

    try:
        MAPS_CARLA_VECTORSTORE = FAISS.load_local(
            MAPS_CARLA_INDEX_PATH,
            EMBEDDING_MODEL,
            allow_dangerous_deserialization=True
        )

        # Recover the vector corresponding to `map_key` directly through `docstore` + `index_to_docstore_id`
        index_to_doc_id = MAPS_CARLA_VECTORSTORE.index_to_docstore_id
        docstore = MAPS_CARLA_VECTORSTORE.docstore

        vector_by_key: Dict[str, List[float]] = {}
        for row_idx, doc_id in index_to_doc_id.items():
            doc = docstore.search(doc_id)
            if not doc:
                continue
            map_key = str(doc_id)
            vec = MAPS_CARLA_VECTORSTORE.index.reconstruct(int(row_idx))
            vector_by_key[map_key] = [float(x) for x in vec]

        MAPS_CARLA_VECTOR_BY_KEY = vector_by_key
        print(f"[MapMatch] Loaded `maps_carla` vectors: {len(MAPS_CARLA_VECTOR_BY_KEY)} entries")
        return bool(MAPS_CARLA_VECTOR_BY_KEY)
    except Exception as e:
        print(f"[MapMatch] Failed to load `maps_carla` index/vectors: {e}")
        traceback.print_exc()
        MAPS_CARLA_VECTORSTORE = None
        MAPS_CARLA_VECTOR_BY_KEY = {}
        return False


def _select_best_map_by_hard_filter_and_semantic(guideline: LogicalScenarioGuideline) -> Dict[str, Any]:
    segments = _load_map_segments_once()

    selected_class = _normalize_map_class(guideline.l1_map_selection.selected_map_class)
    selected_keywords = set([k.strip() for k in guideline.l1_map_selection.key_words if k and k.strip()])

    # Step 1: hard filter by `map_class` (key-value match on the map class)
    class_filtered: List[Tuple[str, Dict[str, Any]]] = []
    for map_key, map_obj in segments.items():
        if _extract_class_from_map_key(map_key) == selected_class:
            class_filtered.append((map_key, map_obj))

    if not class_filtered:
        # Fallback to the full set when hard filtering yields no results to avoid completely breaking the flow
        class_filtered = list(segments.items())
        fallback_reason = f"No results from class hard filtering; fell back to the full candidate set. selected_map_class={selected_class}"
    else:
        fallback_reason = ""

    # Step 2: hard filter by keywords
    keyword_filtered: List[Tuple[str, Dict[str, Any], List[str]]] = []
    for map_key, map_obj in class_filtered:
        map_keywords = _build_map_keywords(map_obj)
        if selected_keywords.issubset(set(map_keywords)):
            keyword_filtered.append((map_key, map_obj, map_keywords))

    if not keyword_filtered:
        # Fallback to `class_filtered` when keyword hard filtering yields no results
        keyword_filtered = [(k, v, _build_map_keywords(v)) for k, v in class_filtered]
        fallback_reason = (
            (fallback_reason + " | ") if fallback_reason else ""
        ) + f"No results from keyword hard filtering; fell back to class candidates. selected_map_keywords={list(selected_keywords)}"

    # Step 3: select the highest semantic score (directly use prebuilt vectors for hard-filtered `map_key`s only)
    query_text = _build_guideline_semantic_query(guideline)

    use_embedding = _initialize_embedding_model_if_needed()
    use_prebuilt_vectors = _initialize_maps_carla_vectors_if_needed()

    best_map_key = ""
    best_map_obj: Dict[str, Any] = {}
    best_score = -1.0

    if use_embedding and use_prebuilt_vectors and EMBEDDING_MODEL:
        query_vec = EMBEDDING_MODEL.embed_query(query_text)
        for map_key, map_obj, _ in keyword_filtered:
            map_vec = MAPS_CARLA_VECTOR_BY_KEY.get(map_key)
            if not map_vec:
                continue
            score = _cosine_similarity(query_vec, map_vec)
            if score > best_score:
                best_score = score
                best_map_key = map_key
                best_map_obj = map_obj

    # If prebuilt vectors are unavailable, or all filtered results are missing from the index, fall back to online vector construction to ensure availability
    if best_map_key == "":
        if use_embedding and EMBEDDING_MODEL:
            query_vec = EMBEDDING_MODEL.embed_query(query_text)
            for map_key, map_obj, _ in keyword_filtered:
                map_text = _build_map_semantic_text(map_key, map_obj)
                map_vec = EMBEDDING_MODEL.embed_query(map_text)
                score = _cosine_similarity(query_vec, map_vec)
                if score > best_score:
                    best_score = score
                    best_map_key = map_key
                    best_map_obj = map_obj
        else:
            # Fallback without a model: approximate using string overlap (still keeps the pipeline runnable)
            q_tokens = set(re.findall(r"[a-zA-Z_]+", query_text.lower()))
            for map_key, map_obj, _ in keyword_filtered:
                map_text = _build_map_semantic_text(map_key, map_obj).lower()
                m_tokens = set(re.findall(r"[a-zA-Z_]+", map_text))
                inter = len(q_tokens.intersection(m_tokens))
                union = len(q_tokens.union(m_tokens)) or 1
                score = inter / union
                if score > best_score:
                    best_score = score
                    best_map_key = map_key
                    best_map_obj = map_obj

    return {
        "best_map_key": best_map_key,
        "best_semantic_score": best_score,
        "fallback_reason": fallback_reason,
        "best_map_annotation": best_map_obj,
    }


def initialize_llm_and_chain(llm_config=None):
    global LLM_INSTANCE, GUIDELINE_PROMPT_TEMPLATE, FINAL_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE
    global GUIDELINE_PARSER, FINAL_OUTPUT_PARSER, CURRENT_LLM_CONFIG

    if llm_config is None:
        llm_config = {
            "base_url": os.getenv("VLLM_BASE_URL", "http://172.20.200.91:8000/v1"),
            "model": os.getenv("VLLM_MODEL_NAME_CODE_GEN", "Qwen3.6-35B-A3B"),
            "api_key": "EMPTY"
        }

    is_config_changed = (
        LLM_INSTANCE is None
        or CURRENT_LLM_CONFIG.get('base_url') != llm_config['base_url']
        or CURRENT_LLM_CONFIG.get('model') != llm_config['model']
        or CURRENT_LLM_CONFIG.get('api_key') != llm_config.get('api_key')
    )

    if not is_config_changed and GUIDELINE_PROMPT_TEMPLATE and FINAL_PROMPT_TEMPLATE and FINAL_OUTPUT_PARSER:
        return True

    try:
        print(f"DSG: Initializing/switching model -> URL: {llm_config['base_url']}, Model: {llm_config['model']}")

        safe_api_key = llm_config.get('api_key') or "EMPTY"
        if safe_api_key in {"cqu-123456", "EMPTY", "", None}:
            safe_api_key = "EMPTY"

        LLM_INSTANCE = ChatOpenAI(
            base_url=llm_config['base_url'],
            api_key=safe_api_key,
            model=llm_config['model'],
            temperature=0.8,
            model_kwargs={
                "top_p": 0.95,
                "presence_penalty": 1.5,
                # vLLM-specific parameters
                "extra_body": {
                    "enable_thinking": False,
                    "preserve_thinking": False
                }
            }
        )

        CURRENT_LLM_CONFIG = llm_config.copy()

        GUIDELINE_PARSER = PydanticOutputParser(pydantic_object=LogicalScenarioGuideline)
        FINAL_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=LogicalScenarioOutput)

        GUIDELINE_PROMPT_TEMPLATE = PromptTemplate(
            template=guideline_prompt_template_str,
            input_variables=["test_function_description", "generation_hint"],
            partial_variables={
                "map_inventory_context": MAP_INVENTORY_CONTEXT,
                "format_instructions": GUIDELINE_PARSER.get_format_instructions()
            }
        )

        FINAL_PROMPT_TEMPLATE = PromptTemplate(
            template=final_prompt_template_str,
            input_variables=["test_function_description", "guideline_json", "best_map", "generation_hint"],
            partial_variables={
                "format_instructions": FINAL_OUTPUT_PARSER.get_format_instructions()
            }
        )

        REFINE_PROMPT_TEMPLATE = PromptTemplate(
            template=refinement_prompt_template_str,
            input_variables=["user_draft", "similar_scenarios_context", "scenario_rules"],
            partial_variables={
                "format_instructions": FINAL_OUTPUT_PARSER.get_format_instructions()
            }
        )

        print("DSG: Scene generation service initialized successfully")
        return True
    except Exception as e:
        print(f"DSG initialization failed: {e}")
        traceback.print_exc()
        return False


def _generate_ai_scenario_id() -> str:
    return f"AI-{random.randint(0, 9999):04d}"


def _build_final_scenario_payload(
    data: Dict[str, Any],
    map_key: str = "",
    map_selection: Optional[MapDescription] = None
) -> Dict[str, Any]:
    """
    Normalize the output to `scenarios.json` style:
    - Keep only `id` / `name` / `description` / `annotations` / `layers` at the top level
    - `name` comes from `LogicalScenarioOutput.scenario_name`
    - `layers` keeps and standardizes only `L0` / `L1` / `L2` / `L3` / `L4` / `L5` / `L6`
    - `L2` / `L6` use fixed default values
    """
    # `best_map` key prioritizes the explicit `map_key`, otherwise falls back to the map class
    d = dict(data)

    scenario_name = d.get("scenario_name") or d.get("name") or ""

    layers_in = d.get("layers", {}) if isinstance(d.get("layers"), dict) else {}

    l1_road = map_key
    if not l1_road and map_selection is not None:
        l1_road = map_selection.selected_map_class

    layers_out = {
        "L0_EgoInternal": layers_in.get("L0_EgoInternal", ""),
        "L1_Road": l1_road,
        "L2_Infrastructure": "No special traffic signs or signals.",
        "L3_TemporaryChanges": layers_in.get("L3_TemporaryChanges", "No construction or temporary obstacles."),
        "L4_DynamicObjects": layers_in.get("L4_DynamicObjects", ""),
        "L5_Environment": layers_in.get("L5_Environment", d.get("annotations", {}).get("weather", "Clear")),
        "L6_DigitalCommunication": "No V2X communication interaction."
    }

    return {
        "id": d.get("id"),
        "name": scenario_name,
        "description": d.get("description", ""),
        "annotations": d.get("annotations", {}),
        "layers": layers_out,
    }


def _is_transient_llm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in [" 502", "502", "bad gateway", "503", "504", "gateway timeout", "service unavailable"])


def _invoke_llm_with_retry(prompt: str, max_retries: int = 3, base_sleep: float = 1.0):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return LLM_INSTANCE.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            transient = _is_transient_llm_error(exc)
            wait_s = base_sleep * (2 ** attempt) if transient else base_sleep
            print(f"    > LLM call failed (Attempt {attempt + 1}/{max_retries}){' [transient]' if transient else ''}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(wait_s)
            if not transient:
                break
    raise last_exc


def _generate_single_safe(desc: str, hint: str, idx: int = 1, shared_guideline_obj: Optional[LogicalScenarioGuideline] = None) -> Optional[Dict]:
    if not LLM_INSTANCE:
        return None

    # --- [New code start: dynamic differentiation strategy] ---
    diversity_focus_list = [
        "Environment challenge: prioritize adverse weather (e.g. rain, snow, fog), low visibility (night, fog, snow), or poor lighting conditions (night, overcast).",
        "Hazardous behavior: prioritize dangerous behaviors of dynamic traffic participants relative to the ego vehicle.",
        "Complex interaction: prioritize complex conflict scenes where multiple traffic participants are involved simultaneously, for example by adding background traffic participants in addition to the adversarial ones."
    ]
    # Use `idx` to decide the current focus and ensure each concurrent run receives different rules
    current_focus = diversity_focus_list[idx % len(diversity_focus_list)]
    
    dynamic_hint = (
        hint
        + f"\n\n[Mandatory Differentiation Requirement]: To ensure diversity in the scene library, please focus the current variant on [{current_focus}]; meanwhile, the scene must always pose a clear challenge to the ego vehicle."
    )
    # --- [New code end] ---

    max_retries = 2
    for i in range(max_retries):
        try:
            print(f"  > AI generation in progress (Attempt {i + 1})...")

            # Step 1: generate or reuse the guideline first
            if shared_guideline_obj is None:
                guideline_prompt = GUIDELINE_PROMPT_TEMPLATE.format(
                    test_function_description=desc,
                    generation_hint=hint
                )
                guideline_resp = _invoke_llm_with_retry(guideline_prompt, max_retries=3, base_sleep=1.0)
                guideline_dict = _clean_json_from_response(guideline_resp.content)
                guideline_obj = LogicalScenarioGuideline.model_validate(guideline_dict)
            else:
                guideline_obj = shared_guideline_obj

            # Step 2: retrieve maps and perform semantic matching based on the generated guideline
            map_match_result = _select_best_map_by_hard_filter_and_semantic(guideline_obj)

            # Step 3: generate the final output based on the guideline and the matched map result
            guideline_payload = guideline_obj.model_dump(exclude_none=True, exclude={"l1_map_selection"})
            best_map_key = map_match_result.get("best_map_key", "")
            best_map_payload = dict(map_match_result.get("best_map_annotation", {}))
            best_map_payload.pop("map_description", None)
            best_map = {"map_key": best_map_key, **best_map_payload} if best_map_key else best_map_payload
            final_prompt = FINAL_PROMPT_TEMPLATE.format(
                test_function_description=desc,
                guideline_json=json.dumps(guideline_payload, ensure_ascii=False),
                best_map=json.dumps(best_map, ensure_ascii=False),
                generation_hint=dynamic_hint
                )
            final_resp = _invoke_llm_with_retry(final_prompt, max_retries=3, base_sleep=1.0)
            final_dict = _clean_json_from_response(final_resp.content)
            final_obj = LogicalScenarioOutput.model_validate(final_dict)

            output_data = final_obj.model_dump(exclude_none=True)

            # Auto-fill the id if it is missing
            if not output_data.get("id"):
                output_data["id"] = _generate_ai_scenario_id()

            output_data = _build_final_scenario_payload(
                output_data,
                map_key=map_match_result.get("best_map_key", ""),
                map_selection=guideline_obj.l1_map_selection
            )
            return output_data
        except Exception as e:
            print(f"  > [Attempt {i + 1}] Generation error: {e}")
            time.sleep(1.5 * (i + 1))

    return None


def generate_multiple_dangerous_scenarios(
    test_function_description: str,
    num_to_generate: int = 1,
    user_provided_generation_hint: str = "",
    selected_l1_road_type_key: str = "",
    llm_config: Dict = None,
    progress_callback=None
) -> List[Dict[str, Any]]:
    if not initialize_llm_and_chain(llm_config):
        return []

    final_hint = user_provided_generation_hint
    if selected_l1_road_type_key:
        final_hint += f"\n[Preferred map type]: {selected_l1_road_type_key}"

    print(f"DSG: Requesting generation of {num_to_generate} scenes...")
    generated_scenarios: List[Dict[str, Any]] = []
    completed = 0

    shared_guideline_obj: Optional[LogicalScenarioGuideline] = None
    if num_to_generate > 0:
        try:
            guideline_prompt = GUIDELINE_PROMPT_TEMPLATE.format(
                test_function_description=test_function_description,
                generation_hint=final_hint
            )
            guideline_resp = _invoke_llm_with_retry(guideline_prompt, max_retries=3, base_sleep=1.0)
            guideline_dict = _clean_json_from_response(guideline_resp.content)
            shared_guideline_obj = LogicalScenarioGuideline.model_validate(guideline_dict)
        except Exception as e:
            print(f"DSG: Shared guideline generation failed; batch generation cannot continue: {e}")
            return []

    for i in range(num_to_generate):
        result = _generate_single_safe(
            test_function_description,
            final_hint,
            i + 1,
            shared_guideline_obj=shared_guideline_obj
        )
        completed += 1
        if progress_callback:
            try:
                progress_callback(completed, num_to_generate, result is not None)
            except Exception:
                pass
        if result:
            generated_scenarios.append(result)

    print(f"DSG: Generation complete, {len(generated_scenarios)} valid scenes generated.")
    return generated_scenarios


def refine_scenario_for_library(
    user_draft: str,
    context: str,
    rules: str,
    llm_config: Dict = None
) -> Optional[Dict[str, Any]]:
    if not initialize_llm_and_chain(llm_config):
        return None

    try:
        print("DSG: Completing/refining the scene draft...")
        prompt_str = REFINE_PROMPT_TEMPLATE.format(
            user_draft=user_draft,
            similar_scenarios_context=context,
            scenario_rules=rules
        )
        response = LLM_INSTANCE.invoke(prompt_str)
        final_dict = _clean_json_from_response(response.content)
        final_obj = LogicalScenarioOutput.model_validate(final_dict)

        output_data = final_obj.model_dump(exclude_none=True)
        if not output_data.get("id"):
            output_data["id"] = _generate_ai_scenario_id()

        output_data = _build_final_scenario_payload(output_data)
        return output_data
    except Exception as e:
        print(f"DSG Refinement Error: {e}")
        return None


def create_document_from_scenario(scenario_data: Dict) -> str:
    """
    Helper function: convert scene data into plain text for RAG retrieval.
    Only use `scenario_name`, `description`, and `L4_DynamicObjects`
    """
    try:
        name = scenario_data.get('scenario_name', '')
        desc = scenario_data.get('description', '')

        layers = scenario_data.get('layers', {})
        if not isinstance(layers, dict):
            layers = {}

        l4_text = layers.get('L4_DynamicObjects', '')

        return f"Scenario Name: {name}\nDescription: {desc}\nDynamics: {l4_text}"
    except Exception as e:
        print(f"Error creating doc from scenario: {e}")
        return str(scenario_data)


if __name__ == "__main__":
    test_config = {
        "base_url": "http://172.20.200.91:8000/v1",
        "model": "Qwen3.6-35B-A3B",
        "api_key": "EMPTY"
    }
    initialize_llm_and_chain(test_config)
    print("--- Testing Generation ---")
    desc = "The vehicle is driving straight on an urban road while a side vehicle quickly cuts into the ego lane, creating a collision risk"
    res = generate_multiple_dangerous_scenarios(desc, 1, llm_config=test_config)
    if res:
        print(json.dumps(res[0], ensure_ascii=False, indent=2))