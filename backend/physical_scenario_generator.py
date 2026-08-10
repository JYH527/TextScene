import json
import os
import re
import sys
from typing import List, Dict, Any, Optional, Literal, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

# ================= 0. Global Variables and Configuration =================
CURRENT_LLM_CONFIG = {}
LLM_INSTANCE = None


# ================= 1. Auxiliary File Loading & Parsing =================
def load_markdown_file(filename: str) -> str:
    """Load a reference file from the current directory."""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[Warning] {filename} not found in the current directory.")
        return f"[{filename} missing; please refer to the default rules]"

def parse_references(ref_text: str) -> Dict[str, str]:
    """Split the English reference file into independent rule blocks."""
    patterns = {
        "global": r"(## Global Physical Unit Declaration.*?)(?=## I\.|## II\.|## III\.|## IV\.|## V\.|$)",
        "environment": r"(## I\. Environment.*?)(?=## II\.|## III\.|## IV\.|## V\.|$)",
        "duration": r"(## II\. Simulation Duration.*?)(?=## III\.|## IV\.|## V\.|$)",
        "entities": r"(## III\. Entities.*?)(?=## IV\.|## V\.|$)",
        "actions": r"(## IV\. Actions.*?)(?=## V\.|$)",
        "triggers": r"(## V\. Triggers.*?)(?=$)",
    }
    sections = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, ref_text, re.DOTALL | re.IGNORECASE)
        sections[key] = match.group(1).strip() if match else f"[Warning: No rules found for {key}]"
    return sections

def prepare_logical_scenario_context(logical_scenario: Dict) -> Dict:
    layers = logical_scenario.get("layers", {})
    return {
        "name": logical_scenario.get("name", ""),
        "description": logical_scenario.get("description", ""),
        "L0_EgoInternal": layers.get("L0_EgoInternal", ""),
        "L1_Road": layers.get("L1_Road", ""),
        "L2_Infrastructure": layers.get("L2_Infrastructure", ""),
        "L3_TemporaryChanges": layers.get("L3_TemporaryChanges", ""),
        "L4_DynamicObjects": layers.get("L4_DynamicObjects", ""),
        "L5_Environment": layers.get("L5_Environment", ""),
    }

# ================= 2. Data Structure Definitions (Pydantic) =================
class PhysicalEnvironment(BaseModel):
    cloud_state: str = Field(default="free", description="Cloud condition")
    precipitation_type: str = Field(default="none", description="Precipitation type")
    precipitation_intensity: float = Field(default=0.0, description="Precipitation intensity")
    fog_visual_range: float = Field(default=100000.0, description="Visible distance under fog")
    sun_azimuth: float = Field(default=0.0, description="Sun azimuth angle")
    sun_elevation: float = Field(default=1.5, description="Sun elevation angle")
    sun_intensity: float = Field(default=100000.0, description="Sunlight intensity")

class EnvironmentAndDurationOutput(BaseModel):
    environment: PhysicalEnvironment
    simulation_duration: float = Field(description="Simulation duration")

class PositionTriggerValue(BaseModel):
    road_id: int = Field(description="Road ID of the trigger point")
    lane_id: int = Field(description="Lane ID of the trigger point")
    s: float = Field(description="Longitudinal S coordinate of the trigger point")

class SpeedActionParams(BaseModel):
    target_speed: float = Field(description="Target speed for deceleration or acceleration")
    dynamics_dimension: Literal["distance"] = Field(default="distance")
    dynamics_value: float = Field(description="Distance over which braking or acceleration occurs (shorter is more aggressive)")

class LaneChangeActionParams(BaseModel):
    target_lane_offset: int = Field(description="Use 1 for a left lane change and -1 for a right lane change. Never use an absolute lane number.")
    dynamics_dimension: Literal["distance"] = Field(default="distance")
    dynamics_value: float = Field(description="Distance required to complete the lane change")

class AssignRouteWayPoint(BaseModel):
    road_id: int = Field(description="Waypoint road ID")
    lane_id: int = Field(description="Waypoint lane ID")
    s: float = Field(description="Waypoint longitudinal position")

class AssignRouteActionParams(BaseModel):
    waypoints: List[AssignRouteWayPoint] = Field(description="A continuous topological waypoint sequence defining the route. Routing rules: 1. With connecting roads (Junction Link): [Start on initial road] -> [Start on connecting road] -> [End on connecting road] -> [End on target road]. 2. Without connecting roads: [Start on initial road] -> [End on initial road] -> [End on target road].")

class FollowTrajectoryVertex(BaseModel):
    lane_id: int = Field(description="Vertex lane ID")
    s: float = Field(description="Vertex longitudinal position")
    offset: float = Field(description="Lateral offset from lane center")

class FollowTrajectoryActionParams(BaseModel):
    vertices: List[FollowTrajectoryVertex] = Field(description="A sequence of physical trajectory vertices; when moving laterally, the s coordinate should increase slightly in sequence. Used only for lateral movement of non-motorized agents on the current road.")

class PhysicalAction(BaseModel):
    actor_name: str = Field(description="Acting entity name")
    type: Literal["SpeedAction", "LaneChangeAction", "FollowTrajectoryAction"] = Field(description="Action type")
    params: Union[SpeedActionParams, LaneChangeActionParams, FollowTrajectoryActionParams] = Field(description="Action parameter dictionary")
    trigger_type: Literal["TraveledDistanceCondition", "EntityReachPosition", "RelativeDistanceCondition"] = Field(description="Trigger type")    
    trigger_value: Union[float, PositionTriggerValue] = Field(description="Relative/Traveled triggers must use a float; ReachPosition must use a coordinate object")
    trigger_ref: str = Field(description="The exact name of the reference entity. Warning: RelativeDistanceCondition must never reference the entity itself.")
    trigger_rule: Literal["lessThan", "greaterThan", "None"] = Field(description="Rule for RelativeDistanceCondition; otherwise None.")
    
    @model_validator(mode='after')
    def validate_type_and_params(self):
        type_mapping = {
            "SpeedAction": SpeedActionParams,
            "LaneChangeAction": LaneChangeActionParams,
            # 👇 移除 AssignRouteAction 及其映射
            "FollowTrajectoryAction": FollowTrajectoryActionParams
        }
        expected_class = type_mapping.get(self.type)
        if expected_class and not isinstance(self.params, expected_class):
            raise ValueError(f"Action type '{self.type}' must match parameter type {expected_class.__name__}")
        return self
    # 👇 新增：强制拦截 Ego 的非法动作
    @model_validator(mode="after")
    def enforce_ego_action_restriction(self):
        # 👇 彻底禁止 Ego 在 Stage 中拥有任何动作
        if self.actor_name == "Ego":
            raise ValueError(
                "CRITICAL RED LINE VIOLATION: The entity 'Ego' MUST NOT be assigned ANY action in the Story stages. "
                "Ego's navigation is fully handled by 'init_waypoints' at t=0."
            )
        return self

class BaseEntity(BaseModel):
    name: str = Field(description="Entity name")
    is_ego: bool = Field(description="Whether this entity is the ego vehicle")
    # 👇 新增：强制定义实体的三大角色
    entity_role: Literal["ego", "background", "adversary"] = Field(description="Strict role: 'ego', 'background', or 'adversary'. ALL roles MUST define their routes in init_waypoints.")
    model_type: str = Field(description="Mapped vehicle model name (e.g. car_white, truck, pedestrian_adult)")
    init_road_id: int = Field(description="Initial road ID")
    init_lane_id: int = Field(description="Initial lane ID")
    init_s: float = Field(description="Initial longitudinal position")
    init_speed: float = Field(description="Initial speed")
    init_waypoints: List[AssignRouteWayPoint] = Field(description="A continuous topological waypoint sequence defining the route. Routing rules: 1. With connecting roads (Junction Link): [Start on initial road] -> [Start on connecting road] -> [End on connecting road] -> [End on target road]. 2. Without connecting roads: [Start on initial road] -> [End on initial road] -> [End on target road].")

class PhysicalEntity(BaseModel):
    name: str
    is_ego: bool
    entity_role: Literal["ego", "background", "adversary"] = Field(description="Strict role: 'ego', 'background', or 'adversary'. ALL roles MUST define their routes in init_waypoints.")
    model_type: str
    init_road_id: int
    init_lane_id: int
    init_s: float
    init_speed: float
    init_waypoints: List[AssignRouteWayPoint] = Field(description="A continuous topological waypoint sequence defining the route. Routing rules: 1. With connecting roads (Junction Link): [Start on initial road] -> [Start on connecting road] -> [End on connecting road] -> [End on target road]. 2. Without connecting roads: [Start on initial road] -> [End on initial road] -> [End on target road].")

class FinalPhysicalActionOutput(BaseModel):
    stage_index: int = Field(description="Stage index")
    action: PhysicalAction = Field(
        description="Action executed in this stage; used to update the motion state after each stage.",
    )

class MacroActionIntent(BaseModel):
    actor_name: str = Field(description="Acting entity name")
    # 👇 移除 "AssignRouteAction"
    action_type: Literal["SpeedAction", "LaneChangeAction", "FollowTrajectoryAction"] = Field(description="Planned macro action type")
    target_speed: Optional[float] = Field(default=None, description="Target speed for speed action")
    target_lane_offset: Optional[int] = Field(default=None, description="Lane offset for lane change action")
    # 👇 移除 target_waypoint 属性
    target_vertex: Optional[FollowTrajectoryVertex] = Field(default=None, description="Target vertex for trajectory action")
    
    @model_validator(mode="after")
    def validate_single_target_field(self):
        target_map = {
            "SpeedAction": self.target_speed is not None,
            "LaneChangeAction": self.target_lane_offset is not None,
            # 👇 移除 AssignRouteAction 映射
            "FollowTrajectoryAction": self.target_vertex is not None,
        }
        if sum(target_map.values()) != 1:
            raise ValueError("Exactly one of target fields must be set based on action_type.")
        return self
    # 👇 新增：强制拦截 Ego 的非法动作
    @model_validator(mode="after")
    def enforce_ego_action_restriction(self):
        # 👇 同理，在宏观意图阶段就掐断 Ego 的非法动作
        if self.actor_name == "Ego":
            raise ValueError(
                "CRITICAL RED LINE VIOLATION: The entity 'Ego' MUST NOT appear in the stages. "
                "Define Ego's route in 'init_waypoints' within entities_placement instead."
            )
        return self

class ScenarioStage(BaseModel):
    stage_index: int = Field(description="Stage index")
    description: str = Field(description="Stage description")
    action: MacroActionIntent = Field(description="Action in this stage")
    reason: str = Field(description="Reason for this stage")

class LLMMacroPlan(BaseModel):
    entities_placement: List[BaseEntity] = Field(description="Initial entity placement list")
    stages: List[ScenarioStage] = Field(description="Scenario stages")
    reasoning_summary: str = Field(description="Overall reasoning summary")

    # 👇 新增：在宏观计划层面全局拦截违规动作分配
    @model_validator(mode="after")
    def enforce_role_restrictions(self):
        role_map = {e.name: e.entity_role for e in self.entities_placement}
        for stage in self.stages:
            actor = stage.action.actor_name
            role = role_map.get(actor)
            if role in ["ego", "background"]:
                raise ValueError(
                    f"CRITICAL RED LINE VIOLATION: Entity '{actor}' has role '{role}'. "
                    f"Entities with role 'ego' or 'background' MUST NOT be assigned ANY action in the stages! "
                    f"Only 'adversary' entities can execute actions in the Story."
                )
        return self

class PhysicalScenarioOutput(BaseModel):
    scenario_name: str = Field(description="Scenario name")
    map_key: str = Field(description="Map key")
    environment: PhysicalEnvironment = Field(description="Environment parameters")
    entities: List[PhysicalEntity] = Field(description="Physical entities")
    actions: List[FinalPhysicalActionOutput] = Field(default_factory=list, description="Stage actions")
    simulation_duration: float = Field(description="Simulation duration")

class AgentState(TypedDict, total=False):
    logical_scenario_str: str
    map_data_dict: str
    map_key: str
    ref_rules: Dict[str, str]
    environment: Optional[PhysicalEnvironment]
    duration: Optional[float]
    macro_plan: Optional[LLMMacroPlan]
    physical_entities: Optional[List[PhysicalEntity]]
    current_entities: Optional[List[PhysicalEntity]]
    current_stage_index: int
    actions: List[FinalPhysicalActionOutput]
    errors: List[str]


# ================= 3. Physics Engine (Kinematics Tool) =================
def tool_params_calculate(current_entities: List[PhysicalEntity], action: PhysicalAction, map_data: dict) -> List[PhysicalEntity]:
    """
    极简且健壮的拓扑物理推演引擎（适配全局 Init 路由架构）：
    1. 彻底弃用模糊的路网推测，严格按照实体绑定的 init_waypoints 队列进行道路跳转。
    2. 支持多 Stage 无缝继承，自动对齐当前所处的道路节点。
    3. 严格遵循 OpenDRIVE S 轴正负方向物理法则。
    """
    entities_dict = {e.name: e.model_dump() for e in current_entities}
    
    def get_val(obj, key, default=None):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    # OpenDRIVE 物理规则：右侧车道(负数)沿 S 轴递增，左侧车道(正数)逆 S 轴递减
    def get_dir(lane_id):
        return 1.0 if int(lane_id) < 0 else -1.0

    # === 1. 解析道路基础长度 ===
    roads_info = {}
    if "raw_roads" in map_data:
        for r_name, r_data in map_data["raw_roads"].items():
            r_id = int(r_name.split("_")[1])
            info = r_data.get("info", "")
            match = re.search(r"Len:([\d\.]+)m", info)
            length = float(match.group(1)) if match else 1000.0
            roads_info[r_id] = {"length": length}

    action_times = {}
    max_stage_time = 0.0
    
    # === 2. 计算触发时间和动作执行时间 ===
    # 直接处理传入的单一 action
    act = action
    
    if act:
        actor_name = get_val(act, "actor_name")
        if actor_name in entities_dict:
            actor = entities_dict[actor_name]
            v_a = float(actor.get("init_speed", 0.0))
            s_a = float(actor.get("init_s", 0.0))
            road_a = int(actor.get("init_road_id", 0))
            
            t_trig = 0.0
            trigger_type = get_val(act, "trigger_type")
            trigger_val = get_val(act, "trigger_value")
            
            if trigger_type == "TraveledDistanceCondition":
                t_trig = float(trigger_val) / max(0.1, v_a)
            elif trigger_type == "EntityReachPosition":
                tgt_s = float(get_val(trigger_val, "s", s_a))
                tgt_r = int(get_val(trigger_val, "road_id", road_a))
                tgt_l = int(get_val(trigger_val, "lane_id", -1))
                
                if tgt_r == road_a:
                    t_trig = abs(tgt_s - s_a) / max(0.1, v_a)
                else:
                    wp_queue = actor.get("init_waypoints", [])
                    curr_dir = 1.0 if int(actor.get("init_lane_id", -1)) < 0 else -1.0
                    r_len = roads_info.get(road_a, {"length": 1000.0})["length"]
                    dist_to_end = (r_len - s_a) if curr_dir > 0 else s_a
                    
                    total_dist = dist_to_end
                    found_target = False
                    started_tracking = False
                    
                    for wp in wp_queue:
                        wp_r = int(get_val(wp, "road_id"))
                        if wp_r == road_a:
                            started_tracking = True
                            continue
                        
                        if not started_tracking:
                            continue
                            
                        if wp_r == tgt_r:
                            tgt_dir = 1.0 if tgt_l < 0 else -1.0
                            tgt_r_len = roads_info.get(tgt_r, {"length": 1000.0})["length"]
                            # 根据目标车道方向判断是从0点驶入还是末端驶入
                            if tgt_dir > 0:
                                total_dist += tgt_s 
                            else:
                                total_dist += (tgt_r_len - tgt_s)
                            found_target = True
                            break
                        else:
                            # 中间连接道路完整长度
                            total_dist += roads_info.get(wp_r, {"length": 1000.0})["length"]
                            
                    if found_target:
                        t_trig = total_dist / max(0.1, v_a)
                    else:
                        t_trig = dist_to_end / max(1.0, v_a)  # 兜底
            
            # 只能在相同Road上才能使用RelativeDistanceCondition
            elif trigger_type == "RelativeDistanceCondition":
                ref_name = get_val(act, "trigger_ref")
                if ref_name in entities_dict:
                    ref_ent = entities_dict[ref_name]
                    v_r = float(ref_ent.get("init_speed", 0.0))
                    s_r = float(ref_ent.get("init_s", 0.0))
                    r_r = int(ref_ent.get("init_road_id", 0))
                    v_rel = abs(v_a - v_r)
                    limit = float(trigger_val)
                    if r_r == road_a:
                        dist_init = abs(s_r - s_a)
                    else:
                        dist_init = 50.0 
                        
                    if get_val(act, "trigger_rule") == "lessThan" and dist_init > limit and v_rel < 0.1:
                        t_trig = 999.0 # 死锁兜底
                    elif v_rel > 0.1:
                        t_trig = abs(dist_init - limit) / v_rel
            
            t_act = 0.0
            act_type = get_val(act, "type")
            params = get_val(act, "params")
            pts = get_val(params, "vertices")
            
            if act_type == "SpeedAction":
                v_target = float(get_val(params, "target_speed", v_a))
                d_act = float(get_val(params, "dynamics_value", 0.0))
                t_act = 2 * d_act / max(0.1, v_a + v_target)
            elif act_type == "LaneChangeAction":
                dist_act = float(get_val(params, "dynamics_value", 0.0))
                t_act = dist_act / max(0.1, v_a)
            elif act_type == "FollowTrajectoryAction" and pts:
                last_offset = float(get_val(pts[-1], "offset", 0.0))
                current_offset = 0.0 # 当前实体默认均在车道中心
                t_act = abs(last_offset - current_offset) / max(0.1, v_a)
                    
            action_times[actor_name] = {"act_type": act_type, "params": params, "t_trig": t_trig, "t_act": t_act, "pts": pts}
            max_stage_time = t_trig + t_act
            
    # === 3. 核心机制：严格依据 init_waypoints 实现多阶段无缝道路跳转 ===
    def move_entity(ent, dist):
        remaining = dist
        curr_r = int(ent.get("init_road_id", 0))
        curr_l = int(ent.get("init_lane_id", -1))
        curr_s = float(ent.get("init_s", 0.0))
        
        wp_queue = ent.get("init_waypoints", []).copy()
        
        while wp_queue and int(get_val(wp_queue[0], "road_id")) != curr_r:
            wp_queue.pop(0)

        while remaining > 0.01:
            r_len = roads_info.get(curr_r, {"length": 1000.0})["length"]
            curr_dir = get_dir(curr_l)
            dist_to_end = (r_len - curr_s) if curr_dir > 0 else curr_s

            if remaining <= dist_to_end:
                curr_s += remaining * curr_dir
                remaining = 0
                break

            remaining -= dist_to_end
            curr_s = r_len if curr_dir > 0 else 0.0

            while wp_queue and int(get_val(wp_queue[0], "road_id")) == curr_r:
                wp_queue.pop(0)

            if wp_queue:
                next_wp = wp_queue[0]
                curr_r = int(get_val(next_wp, "road_id"))
                curr_l = int(get_val(next_wp, "lane_id"))
                curr_dir = get_dir(curr_l)
                curr_s = 0.0 if curr_dir > 0 else roads_info.get(curr_r, {"length": 1000.0})["length"]
            else:
                remaining = 0
                break

        ent["init_road_id"] = curr_r
        ent["init_lane_id"] = curr_l
        ent["init_s"] = curr_s

    # === 4. 执行物理位移推演 ===
    for name, ent in entities_dict.items():
        v_init = float(ent.get("init_speed", 0.0))
        
        if name in action_times:
            info = action_times[name]
            act_type, params = info["act_type"], info["params"]
            t_trig, t_act = info["t_trig"], info["t_act"]
            pts = info["pts"]
            
            # 1. 触发前滑行
            move_entity(ent, v_init * t_trig)
            
            # 2. 动作期结算
            if act_type == "SpeedAction":
                dist_act = float(get_val(params, "dynamics_value", 0.0))
                move_entity(ent, dist_act)
                v_current = float(get_val(params, "target_speed", v_init))
                ent["init_speed"] = v_current
                move_entity(ent, max(0, max_stage_time - t_trig - t_act) * v_current)
                
            elif act_type == "LaneChangeAction":
                dist_act = float(get_val(params, "dynamics_value", 0.0))
                move_entity(ent, dist_act)
                current_lane = int(ent.get("init_lane_id"))
                offset = int(get_val(params, "target_lane_offset", 0))
                new_lane = current_lane + offset
                if current_lane < 0 and new_lane >= 0:
                    new_lane += 1 
                elif current_lane > 0 and new_lane <= 0:
                    new_lane -= 1
                ent["init_lane_id"] = new_lane
                move_entity(ent, max(0, max_stage_time - t_trig - t_act) * v_init)
                
            elif act_type == "FollowTrajectoryAction" and pts:
                last_wp = pts[-1]
                ent["init_road_id"] = int(get_val(last_wp, "road_id", ent.get("init_road_id")))
                ent["init_lane_id"] = int(get_val(last_wp, "lane_id", ent.get("init_lane_id")))
                ent["init_s"] = float(get_val(last_wp, "s", ent.get("init_s")))
                ent["offset"] = float(get_val(last_wp, "offset", 0.0))
                move_entity(ent, max(0, max_stage_time - t_trig - t_act) * v_init)
        else:
            move_entity(ent, v_init * max_stage_time)
            
    return [PhysicalEntity.model_validate(v) for v in entities_dict.values()]

def _invoke_with_parser_streaming(llm, prompt_str: str, pydantic_class: type[BaseModel], step_name: str) -> BaseModel:
    """直接调用大模型并严格要求输出特定 Schema 的 JSON，不再提取中间思考流。"""
    schema = pydantic_class.model_json_schema()
    
    system_prompt = (
        "You are a world-class autonomous driving test scenario construction expert.\n"
        "Your task is to transform a logical scenario into a concise parameterized physical scenario, the parameterized scenarios will serve as a guideline blueprint for the subsequent writing of OpenSCENARIO 1.0 code.\n"
        "TARGET JSON SCHEMA:\n"
        f"```json\n{json.dumps(schema, ensure_ascii=False)}\n```\n\n"
        "CRITICAL RULE FOR YOUR RESPONSE:\n"
        "You are highly encouraged to output your step-by-step mathematical reasoning first. However, your final answer MUST contain exactly ONE valid JSON block matching the schema above, wrapped in ```json ... ```."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_str)
    ]
    
    response = llm.invoke(messages)
    full_response = response.content
    
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', full_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            start_idx = full_response.find('{')
            end_idx = full_response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = full_response[start_idx:end_idx + 1].strip()
            else:
                json_str = full_response.strip()

        return pydantic_class.model_validate_json(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse model output. Error: {str(e)}\nRaw model output excerpt: {full_response[-300:]}")


# ================= 5. LangGraph Nodes & Edges =================
def node_generate_environment(state: AgentState):
    print("[Agent] ➡️ Phase 1: Environment & Duration Generation")
    logical_desc = json.loads(state['logical_scenario_str']).get('description', '')
    env_desc = json.loads(state['logical_scenario_str']).get('L5_Environment', '')
    prompt = f"""
    As an expert in autonomous driving test scenario parameterization, please extract and design the physical scenario's environment parameters and simulation duration based on the logical scenario description and physical mapping rules.

    [Input Context]
    - Logical scenario description: {logical_desc}
    - L5 environment description: {env_desc}
    - Environment parameter mapping rules: {state['ref_rules']['environment']}

    [Duration mapping rules] (Must be strictly followed)
    - {state['ref_rules']['duration']}

    ⚠️ [Format Requirements] ⚠️:
    - Before generating the final JSON, you MUST conduct a rigorous step-by-step reasoning and calculation to ensure the environment parameters and duration perfectly match the physical mapping rules.
    - Your output must strictly match the `EnvironmentAndDurationOutput` JSON format requirements.
    - The final JSON must be wrapped in ```json ... ```.
    """
    res = _invoke_with_parser_streaming(LLM_INSTANCE, prompt, EnvironmentAndDurationOutput, "Env Gen")
    
    # [修改]：将环境的所有参数值与物理含义融合为专业描述
    summary = (
        f"Environment & Duration Configuration Complete. "
        f"The simulation duration is strictly set to {res.simulation_duration} seconds to ensure sufficient time for the scenario to unfold. "
        f"Atmospheric conditions are physically modeled with a '{res.environment.cloud_state}' cloud state and '{res.environment.precipitation_type}' precipitation at an intensity level of {res.environment.precipitation_intensity}, "
        f"while the visual range is restricted to {res.environment.fog_visual_range} meters to accurately simulate sensor visibility constraints. "
        f"Lighting dynamics are governed by a sun azimuth angle of {res.environment.sun_azimuth}° and an elevation of {res.environment.sun_elevation}°, "
        f"providing {res.environment.sun_intensity} lux of environmental illumination."
    )
    print(f"[NODE_SUMMARY]::{summary}")
    
    return {"environment": res.environment, "duration": res.simulation_duration}

def node_design_macro_plan(state: AgentState):
    print("[Agent] ➡️ Phase 2: Scenario Macro Script Planning")
    prompt = f"""
    You are an Expert Autonomous Driving Test Scenario Orchestrator specializing in ADVERSARIAL EDGE CASES. Your strict mandate is to generate a macro scenario plan (Stages) by selecting participating entities, calculating their exact initial physical kinematics, and orchestrating their actions chronologically to create DANGEROUS, high-risk situations for the Ego vehicle.

    [INPUT CONTEXT]
    - Logical Scenario: {state['logical_scenario_str']}
    - OpenDRIVE Map Annotations: {state['map_data_dict']}

    🔥 [ADVERSARIAL DANGER MANDATE - MAXIMIZE RISK] 🔥
    - Your primary goal is to force the Ego vehicle into emergency responses (e.g., hard braking, evasive steering, crash avoidance) while maintaining physical plausibility.
    - You MUST minimize Time-To-Collision (TTC) while maintaining physical plausibility.
    - Adversary must be positioned and orchestrated to execute dangerous behaviors: blind-spot cut-ins, sudden panic braking in front of Ego, or running intersections.

    [CRITICAL CONSTRAINTS & RED LINES - FAILURE TO OBEY WILL CAUSE SIMULATION CRASH]

    1. MAP & TOPOLOGY VALIDITY (ZERO HALLUCINATION):
       - You MUST ONLY use `road_id` and `lane_id` values explicitly present in the [OpenDRIVE map annotations].
       - S-Axis Direction Rule: 
         * IF `lane_id < 0`: Driving direction is ALONG the S-axis (Forward means `s` INCREASES).
         * IF `lane_id > 0`: Driving direction is AGAINST the S-axis (Forward means `s` DECREASES).

    2. ENTITY INITIALIZATION RULES (STRICT MATH LOGIC REQUIRED):
       You MUST classify every entity into one of three strict `entity_role` categories:
       - 🔵 "ego" (Ego Vehicle): `name="Ego"`, `is_ego=true`.
       - 🟢 "background" (Background Traffic): Normal vehicles driving safely.
       - 🔴 "adversary" (Adversarial Entity): The threat actor executing dangerous maneuvers.
       
       => BASELINE NAVIGATION (init_waypoints): ALL entities MUST define their route using `init_waypoints`.
       ⚠️ STRICT WAYPOINT GENERATION RULES (CRITICAL):
       - IF there IS a connecting road (Junction Link) between the initial and target road, you MUST generate exactly 4 waypoints:
         [Start on initial road] -> [Start on connecting road] -> [End on connecting road] -> [End on target road].
       - IF there is NO connecting road (direct continuation or same road), you MUST generate exactly 3 waypoints:
         [Start on initial road] -> [End on initial road] -> [End on target road].

    3. EGO ACTION RESTRICTIONS (ABSOLUTE PROHIBITION):
       - Entities labeled as "ego" or "background" MUST NOT appear in the `stages` array under any circumstances.
       - The `stages` array is STRICTLY EXCLUSIVE to "adversary" entities executing DANGEROUS/SUDDEN actions (e.g., LaneChangeAction cut-ins, panic SpeedAction). DO NOT use AssignRouteAction in stages.
       
    4. STAGE ORCHESTRATION RULES (ISOLATION MANDATE):
       - ONE ACTION PER STAGE: A single Stage MUST contain EXACTLY ONE action assigned to EXACTLY ONE entity.
       - NO CONCURRENCY: IF "adversary" entities act simultaneously, or one "adversary" entities does multiple things, you MUST split them into consecutive, separate Stages.
       - DO NOT generate empty Stages (every Stage must change an entity's motion state).

    5. ACTION PARAMETER EXCLUSIVITY (SCHEMA ENFORCEMENT):
       Based on the `action_type`, you MUST populate EXACTLY ONE target parameter. All others MUST be null or omitted:
       - IF `SpeedAction` ➔ EXCLUSIVELY provide `target_speed`.
       - IF `LaneChangeAction` ➔ EXCLUSIVELY provide `target_lane_offset` (Must be `1` for left, `-1` for right. NEVER use absolute lane IDs).
       - IF `FollowTrajectoryAction` ➔ EXCLUSIVELY provide `target_vertex`.
       (⚠️ CRITICAL: DO NOT use AssignRouteAction in stages. All routing is strictly handled via init_waypoints)

    ⚠️ [OUTPUT FORMAT & REASONING PIPELINE] ⚠️
    1. Before generating the JSON, you MUST first analyze the OpenDRIVE topology, identify valid participating entities, and calculate each entity's initial road, lane, longitudinal position, and speed so the placement is physically valid.
    2. You MUST also reason about the macro staging plan by deciding how many sequential stages are required, which single actor is active in each stage, and how each stage advances the scenario without violating the one-action-per-stage rule.
    3. Your final JSON structure MUST perfectly match the `LLMMacroPlan` schema.
    4. The JSON must be valid and wrapped in ```json ... ```.
    """
    res = _invoke_with_parser_streaming(LLM_INSTANCE, prompt, LLMMacroPlan, "Macro Script")
    
    # [修改]：将实体初始化位置、速度、以及宏观 Stage 阶段的意图融合为专业描述
    entity_details = []
    for e in res.entities_placement:
        role = "Ego vehicle" if e.is_ego else "NPC"
        entity_details.append(
            f"'{e.name}' ({role}, mapped to 3D model '{e.model_type}') is anchored on road ID {e.init_road_id}, "
            f"lane ID {e.init_lane_id} at longitudinal coordinate s={e.init_s}m, carrying an initial kinematic speed of {e.init_speed}m/s"
        )
    entities_str = "; ".join(entity_details)

    stage_details = []
    for s in res.stages:
        stage_details.append(
            f"Stage {s.stage_index} dictates that actor '{s.action.actor_name}' will execute a {s.action.action_type} "
            f"designed to {s.description.lower()} (Reason: {s.reason.lower()})"
        )
    stages_str = " then ".join(stage_details)

    summary = (
        f"Macro Script & Kinematic Blueprint Established. "
        f"Initial Entity Placement: {entities_str}. "
        f"Stage Orchestration Logic: {stages_str}. "
        f"Overall Tactical Rationale: {res.reasoning_summary}"
    )
    print(f"[NODE_SUMMARY]::{summary}")
    
    entities = [PhysicalEntity.model_validate(e.model_dump()) for e in res.entities_placement]
    return {
        "macro_plan": res,
        "physical_entities": entities,
        "current_entities": entities,
        "current_stage_index": 0,
        "actions": []
    }

def node_design_action(state: AgentState):
    current_idx = state.get("current_stage_index", 0)
    stages = state["macro_plan"].stages
    current_stage = stages[current_idx]
    current_entities = state.get("current_entities", state.get("physical_entities", []))
    print(f"[Agent] ➡️ Phase 3: Action Design (Stage {current_stage.stage_index}/{len(stages)})")
    prompt = f"""
    You are an Expert Autonomous Driving Action Compiler specializing in ADVERSARIAL BEHAVIOR. Your strict mandate is to convert the macro intent of Stage {current_stage.stage_index} into precise, aggressively timed physical action parameters (Params) and dynamic triggers (Trigger) to maximize danger to the Ego.

    [INPUT CONTEXT]
    - OpenDRIVE Map Annotations: {state['map_data_dict']}
    - CURRENT Physical State of Entities: {json.dumps([e.model_dump() for e in current_entities], ensure_ascii=False)}
    - MACRO INTENT for THIS Stage: {current_stage.model_dump_json(indent=2)}

    [CRITICAL CONSTRAINTS & RED LINES - EXECUTE IN STRICT ORDER]

    1. ENTITY SELECTION (ISOLATION MANDATE):
       - You MUST ONLY generate the `action` for the exact "adversary" entity specified in the [MACRO INTENT]. 
       - DO NOT generate an action for any other entities. They will automatically maintain their current motion state.

    2. ACTION PARAMETER CALCULATION (AGGRESSIVE DYNAMICS STRICTLY ENFORCED):
       Based on the `action_type`, enforce the following physical constraints:
       - IF `SpeedAction`: Provide `target_speed`. You MUST infer a `dynamics_value` that represents a SEVERE/AGGRESSIVE maneuver. ⚠️ CRITICAL: Even for hard braking (deceleration), `dynamics_value` for 'rate' MUST BE A POSITIVE ABSOLUTE FLOAT (e.g., 8.0, never -8.0). The engine deduces deceleration automatically.
       - IF `LaneChangeAction`: 
         * `target_lane_offset` MUST strictly be `1` (for Left) or `-1` (for Right). NEVER use absolute lane numbers!
         * `dynamics_value` MUST be dangerously short (10m - 25m) to simulate an aggressive, sudden cut-in, minimizing the Ego's reaction window.
       - IF `FollowTrajectoryAction`: Provide a `vertices` sequence.

    3. TRIGGER DESIGN (ADVERSARIAL TIMING REQUIRED):
       You MUST select ONE of the following Trigger Cases based on physical timing logic to maximize risk.

       [CASE A: Immediate Execution Trigger]
       - USE WHEN: The action must start instantly at the beginning of this stage.
       - `trigger_type`: "TraveledDistanceCondition"
       - `trigger_value`: 0.0 (Must be exactly 0.0 float)
       - `trigger_ref`: The exact name of the acting entity (`actor_name`).

       [CASE B: Relative Distance Trigger (MANDATORY for Dangerous Cut-ins & Panic Braking)]
       - USE WHEN: Action strictly depends on the dynamic gap to another vehicle.
       - `trigger_type`: "RelativeDistanceCondition"
       - `trigger_ref`: The EXACT name of the target reference entity (e.g., "Ego"). ⚠️ RED LINE: NEVER set the `actor_name` itself as the reference!
       - `trigger_value`: A precise physical distance gap (e.g., 5.0, 10.0, 15.0). ⚠️ RED LINE: NEVER use 0.0 here!
       - `trigger_rule`: "lessThan" or "greaterThan".
       - ⚠️ DEADLOCK AVOIDANCE RULE (CRITICAL PHYSICS CONSTRAINT): 
         The OpenSCENARIO engine uses a "Rising Edge" trigger. The condition MUST evaluate to FALSE when the stage begins, and later become TRUE as the vehicles move. If it is TRUE at the start, it will DEADLOCK!
         * IF using "lessThan" (e.g., waiting for vehicles to get close): The CURRENT GAP between the entities at the start of this stage MUST be strictly GREATER than your `trigger_value`. (e.g., If current gap is 15m, setting `trigger_value` to 10m is valid. If gap is 8m, setting `trigger_value` to 10m will DEADLOCK).
         * IF using "greaterThan" (e.g., waiting for vehicles to separate): The CURRENT GAP MUST be strictly LESS than your `trigger_value`.

       [CASE C: Absolute Position Trigger]
       - USE WHEN: Action triggers at a specific map coordinate (e.g., running a red light just as Ego approaches an intersection).
       - `trigger_type`: "EntityReachPosition"
       - `trigger_value`: A valid `PositionTriggerValue` strictly derived from the map.
       - `trigger_ref`: The exact name of the acting entity (`actor_name`).

    ⚠️ [OUTPUT FORMAT & REASONING PIPELINE] ⚠️
    1. Before generating the JSON, you MUST first analyze the current stage's macro intent, the listed actor, the action type, and the map-based physical context, then compute the trigger condition, action parameters, and resulting motion update so the stage output is physically consistent.
    2. Your final JSON structure MUST perfectly match the `FinalPhysicalActionOutput` schema.
    3. Ensure the outermost `stage_index` strictly equals {current_stage.stage_index}.
    4. The JSON must be valid and wrapped in ```json ... ```.
    """
    res = _invoke_with_parser_streaming(LLM_INSTANCE, prompt, FinalPhysicalActionOutput, f"Action_Stage_{current_idx}")
    
    # [修改]：将触发器类型、条件、阈值，以及动作的加减速度、车道偏移、作用距离全部解构并结合在一起解释
    action_logs = []
    a = res.action
    # 深度解析触发器参数及其含义
    if a.trigger_type == "RelativeDistanceCondition":
        trig_exp = f"when the relative distance to reference entity '{a.trigger_ref}' evaluates to {a.trigger_rule} {a.trigger_value} meters"
    elif a.trigger_type == "TraveledDistanceCondition":
        trig_exp = f"after entity '{a.trigger_ref}' has traveled exactly {a.trigger_value} meters"
    elif a.trigger_type == "EntityReachPosition":
        trig_exp = f"when entity '{a.trigger_ref}' arrives at the specific map coordinate (road {getattr(a.trigger_value, 'road_id', 'N/A')}, lane {getattr(a.trigger_value, 'lane_id', 'N/A')}, s={getattr(a.trigger_value, 's', 'N/A')}m)"
    else:
        trig_exp = f"activated by {a.trigger_type} with a threshold of {a.trigger_value}"

    # 深度解析物理动作参数及其含义
    if a.type == "SpeedAction":
        param_exp = f"aggressively adjusting its target speed to {getattr(a.params, 'target_speed', 'N/A')} m/s completed over a tight {getattr(a.params, 'dynamics_dimension', 'distance')} of {getattr(a.params, 'dynamics_value', 'N/A')} meters"
    elif a.type == "LaneChangeAction":
        offset = getattr(a.params, 'target_lane_offset', 0)
        dir_str = "left" if offset > 0 else "right"
        param_exp = f"executing a {dir_str} lane change (lane offset: {offset}) aggressively mapped over a {getattr(a.params, 'dynamics_dimension', 'distance')} of {getattr(a.params, 'dynamics_value', 'N/A')} meters"
    elif a.type == "FollowTrajectoryAction":
        pts = getattr(a.params, 'vertices', [])
        param_exp = f"following a rigid local coordinate trajectory structured by {len(pts)} physical vertices"
    else:
        param_exp = f"executing basic {a.type}"

    action_logs.append(f"Actor '{a.actor_name}' is compiled to execute a '{a.type}', {param_exp}, which is precisely activated {trig_exp}")

    actions_desc = " Furthermore, ".join(action_logs)
    summary = (
        f"Physical Action Dynamics Compiled for Stage {res.stage_index}. "
        f"{actions_desc}. "
        f"These exact physical parameters ensure adversarial constraint adherence and have been forwarded to the local Kinematics Tool for collision checking and trajectory integration."
    )
    print(f"[NODE_SUMMARY]::{summary}")
    
    print(f"[Tool] 🛠️ Calculating Kinematics via tool_params_calculate...")
    map_dict = json.loads(state['map_data_dict']) 
    updated_entities = tool_params_calculate(current_entities, res.action, map_dict) 
    return {
        "actions": state.get("actions", []) + [res],
        "current_entities": updated_entities,
        "current_stage_index": current_idx + 1
    }

def route_next_stage(state: AgentState) -> Literal["design_action", "__end__"]:
    current_idx = state.get("current_stage_index", 0)
    # Defensive mechanism: if the LLM produces an empty macro_plan or an empty stages list, end immediately
    if not state.get("macro_plan") or not state["macro_plan"].stages:
        return "__end__"
    total_stages = len(state["macro_plan"].stages)
    return "design_action" if current_idx < total_stages else "__end__"


# ================= 6. Graph Construction & Runner =================
def build_graph():
    """Build the unified LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate_environment", node_generate_environment)
    workflow.add_node("design_macro_plan", node_design_macro_plan)
    workflow.add_node("design_action", node_design_action)
    
    workflow.add_edge(START, "generate_environment")
    workflow.add_edge("generate_environment", "design_macro_plan")
    workflow.add_edge("design_macro_plan", "design_action")
    workflow.add_conditional_edges("design_action", route_next_stage, {"design_action": "design_action", "__end__": END})
    
    return workflow.compile()

def init_physical_chain(llm_config=None):
    global LLM_INSTANCE, CURRENT_LLM_CONFIG
    try:
        if llm_config is None: raise ValueError("llm_config is required")
        CURRENT_LLM_CONFIG = llm_config.copy()
        LLM_INSTANCE = ChatOpenAI(
            base_url=llm_config['base_url'],
            api_key=llm_config.get('api_key', "EMPTY"),
            model=llm_config['model'],
            temperature=0.4,
            streaming=False,
            model_kwargs={
                "top_p": 0.95,
                "presence_penalty": 0.0
            }
        )
        return True
    except Exception as e:
        print(f"[Init] LLM Error: {e}")
        return False

def generate_physical_scenario(logical_scenario: Dict, map_raw_data: Dict, llm_config: Dict = None) -> Dict:
    if not init_physical_chain(llm_config):
        raise RuntimeError("LLM Init Failed")

    ref_rules_dict = parse_references(load_markdown_file("Skills/physical_parameters/references_simple.md"))
    filtered_scenario = prepare_logical_scenario_context(logical_scenario)
    
    map_key = logical_scenario.get("layers", {}).get("L1_Road", "")
    if not map_key: raise ValueError("Missing map_key in logical_scenario")

    cleaned_map = dict(map_raw_data or {})
    cleaned_map.pop("map_description", None)

    initial_state: AgentState = {
        "logical_scenario_str": json.dumps(filtered_scenario, ensure_ascii=False),
        "map_data_dict": json.dumps(cleaned_map, ensure_ascii=False),
        "map_key": map_key,
        "ref_rules": ref_rules_dict,
        "environment": None,
        "duration": None,
        "macro_plan": None,
        "physical_entities": [],
        "current_entities": [],
        "actions": [],
        "current_stage_index": 0,
        "errors": [],
    }

    app = build_graph()
    print(f"\n🚀 [Agent Workflow Started] Map: {map_key}")
    final_state = app.invoke(initial_state)

    map_key_short = map_key
    if match := re.search(r"(Town\d{2})", map_key):
        map_key_short = f"{match.group(1)}.xodr"

    final_output = PhysicalScenarioOutput(
        scenario_name=logical_scenario.get("name", "Generated_Scenario"),
        map_key=map_key_short,
        environment=final_state["environment"],
        entities=final_state.get("physical_entities", []),
        actions=final_state.get("actions", []),
        simulation_duration=final_state["duration"] or 10.0,
    )
    return final_output.model_dump(exclude_none=True)


# ================= 7. External API =================
def generate(logical_scenario: Dict, map_annotation: Dict, llm_config: Dict = None) -> str:
    """Unified external entry point."""
    map_key = logical_scenario.get("annotations", {}).get("map_key") or logical_scenario.get("l1_map_selection", {}).get("selected_map_key")
    map_data = map_annotation.get(map_key) if (map_key and isinstance(map_annotation, dict)) else map_annotation
    physical_dict = generate_physical_scenario(logical_scenario, map_data, llm_config)
    return json.dumps(physical_dict, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pass