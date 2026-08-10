# --- START OF FILE coder_generate.py ---

import os
import traceback
import json
from typing import Optional, Dict

# 引入 LangChain 相关库
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models.chat_models import BaseChatModel

# ==================== 核心 Prompt 模板 (针对 Esmini 深度优化) ====================
OPENSCENARIO_GENERATION_PROMPT_TEMPLATE = """
Role:
你是一位精通 OpenSCENARIO 1.0 标准和 Esmini 仿真器的底层代码生成专家。
Task:
根据下方提供的 [基于 JSON 的结构化物理场景文本]，编写一份严谨、合规且可以直接在 Esmini 中运行的 .xosc 文件。

--- ⚠️ 核心语法与映射红线 (违反任意一条会导致仿真崩溃) ⚠️ ---

1. 严禁发明 XML 标签：绝对不允许出现 <Stage>, <Trigger>, <Actor> 等非标准标签。
2. 命名全局唯一：所有 Event, Action, Condition 的 name 属性必须唯一 (如 name="Event_Adversary1_0")。
3. 注意标签的缩进一致和闭合

=== 必须严格遵守的 XML 组装模板 ===

1. 文件头与参考地图 (直接使用)：
<?xml version="1.0" encoding="utf-8"?>
<OpenSCENARIO xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="OpenScenario.xsd">
    <FileHeader description="[scenario_name]" author="System" revMajor="1" revMinor="0" date="2025-01-01T00:00:00"/>
    <ParameterDeclarations/>
    <CatalogLocations>
        <VehicleCatalog>
            <Directory path="../resources/xosc/Catalogs/Vehicles"/>
        </VehicleCatalog>
        <PedestrianCatalog>
            <Directory path="../resources/xosc/Catalogs/Pedestrians"/>
        </PedestrianCatalog>
    </CatalogLocations>
    <RoadNetwork>
        <LogicFile filepath="[map_key]"/>
    </RoadNetwork>

2. Entities 实体定义：
    <Entities>
        <!-- 遍历 entities 列表生成，pedestrian 使用 PedestrianCatalog，车辆使用 VehicleCatalog -->
        <ScenarioObject name="[name]">
            <CatalogReference catalogName="VehicleCatalog" entryName="[model_type]"/>
        </ScenarioObject>
        <若含有障碍物，则使用 MiscObject 标签，并使用 miscObjectCategory="obstacle" 属性>
        <MiscObject name="box_yellow" miscObjectCategory="obstacle" mass="10.0">
            <BoundingBox>
                <Center x="0.0" y="0.0" z="0.3"/>
                <Dimensions width="1.5" length="1.0" height="0.8"/>
            </BoundingBox>
            <Properties>
                <Property name="type" value="cardboard_box"/>
            </Properties>
        </MiscObject>
    </Entities>

3. Storyboard 初始化 (<Init> 必须包含 <Actions>)：
    <Storyboard>
        <Init>
            <Actions> <!-- ⚠️ 绝对不能漏掉此标签 -->
                <GlobalAction>
                    <EnvironmentAction>
                        <Environment name="Environment1">
                            <TimeOfDay animation="false" dateTime="2025-01-01T12:00:00"/>
                            <Weather cloudState="[cloud_state]">
                                <Sun intensity="[sun_intensity]" azimuth="[sun_azimuth]" elevation="[sun_elevation]"/>
                                <Fog visualRange="[fog_visual_range]"/>
                                <Precipitation precipitationType="[precipitation_type]" intensity="[precipitation_intensity]"/>
                            </Weather>
                            <RoadCondition frictionScaleFactor="1.0"/>
                        </Environment>
                    </EnvironmentAction>
                </GlobalAction>
                <!-- 遍历实体初始状态 -->
                <Private entityRef="[name]">
                    <PrivateAction>
                        <LongitudinalAction>
                            <SpeedAction>
                                <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                                <SpeedActionTarget>
                                    <AbsoluteTargetSpeed value="[init_speed]"/>
                                </SpeedActionTarget>
                            </SpeedAction>
                        </LongitudinalAction>
                    </PrivateAction>
                    <PrivateAction>
                        <TeleportAction>
                            <Position>
                                <LanePosition roadId="[init_road_id]" laneId="[init_lane_id]" s="[init_s]" offset="0.0"/>
                            </Position>
                        </TeleportAction>
                    </PrivateAction>
                    <!-- Ego 和 背景实体 /Adversary实体 需要在Story部分定义初始路线 -->
                    <PrivateAction>
                        <RoutingAction>
                            <AssignRouteAction>
                                <Route name="InitRoute_[name]" closed="false">
                                    <!-- 遍历 waypoints -->
                                    <Waypoint routeStrategy="shortest">
                                        <Position>
                                            <LanePosition roadId="[road_id]" laneId="[lane_id]" s="[s]" offset="0.0"/>
                                        </Position>
                                    </Waypoint>
                                </Route>
                            </AssignRouteAction>
                        </RoutingAction>
                    </PrivateAction>
                </Private>
            </Actions>
        </Init>

4. 动作执行部分 (Story 骨架)：
读取 JSON 中的 `actions` 列表，每一个stage的每一个action，每一个action都映射为一个Act，严格按照以下骨架套用！
【⚠️ Act 的 StartTrigger 串行触发规则 ⚠️】：
- 第一个 Act（如 stage_index == 0）：<StartTrigger> 必须使用 `SimulationTimeCondition` (value="0")；
- 后续的 Act（如 stage_index > 0）：<StartTrigger> 必须使用 `StoryboardElementStateCondition`，以上一个 Act 的 `completeState` (完成状态) 作为启动条件。

        <Story name="MainStory">
            <Act name="Act_[actor_name]_[stage_index]">
                <ManeuverGroup maximumExecutionCount="1" name="MG_[actor_name]_[stage_index]">
                    <Actors selectTriggeringEntities="false">
                        <EntityRef entityRef="[actor_name]"/>
                    </Actors>
                    <Maneuver name="Man_[actor_name]_[stage_index]">
                        <Event name="Ev_[actor_name]_[stage_index]" priority="overwrite" maximumExecutionCount="1">
                            <!-- 【这里插入 Action 标签，见下方说明】 -->
                            <!-- 【这里插入 StartTrigger 标签，见下方说明】 -->
                        </Event>
                    </Maneuver>
                </ManeuverGroup>
                <StartTrigger>
                    <ConditionGroup>
                        <Condition name="ActStart_[actor_name]_[stage_index]" delay="0" conditionEdge="rising">
                            <ByValueCondition>
                                <SimulationTimeCondition value="0" rule="greaterThan"/>
                            </ByValueCondition>
                        </Condition>
                        
                        <!-- 【如果 stage_index > 0，则添加以下条件】 -->
                        <Condition name="ActStart_[actor_name]_[stage_index]" delay="0" conditionEdge="rising">
                            <ByValueCondition>
                                <StoryboardElementStateCondition storyboardElementType="act" storyboardElementRef="Act_[上一个动作的actor_name]_[stage_index - 1]" state="completeState"/>
                            </ByValueCondition>
                        </Condition>
                        
                    </ConditionGroup>
                </StartTrigger>
            </Act>
        </Story>

【支持的 Action 标签片段 (填入上述 Event 中)】:
- LaneChangeAction (变道):
    <Action name="Act_LC_[actor_name]_[stage_index]">
        <PrivateAction>
            <LateralAction>
                <LaneChangeAction>
                    <LaneChangeActionDynamics dynamicsShape="sinusoidal" value="[dynamics_value]" dynamicsDimension="distance"/>
                    <LaneChangeTarget>
                        <RelativeTargetLane entityRef="[actor_name]" value="[target_lane_offset]"/>
                    </LaneChangeTarget>
                </LaneChangeAction>
            </LateralAction>
        </PrivateAction>
    </Action>

- SpeedAction (变速):
    <Action name="Act_Spd_[actor_name]_[stage_index]">
        <PrivateAction>
            <LongitudinalAction>
                <SpeedAction>
                    <SpeedActionDynamics dynamicsShape="linear" value="[dynamics_value]" dynamicsDimension="distance"/>
                    <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="[target_speed]"/>
                    </SpeedActionTarget>
                </SpeedAction>
            </LongitudinalAction>
        </PrivateAction>
    </Action>

- FollowTrajectoryAction (轨迹跟随):
    <Action name="Act_Traj_[actor_name]_[stage_index]">
        <PrivateAction>
            <RoutingAction>
                <FollowTrajectoryAction>
                    <Trajectory name="Traj_[actor_name]_[stage_index]" closed="false">
                        <Shape>
                            <Polyline>
                                <!-- 遍历 JSON params.vertices -->
                                <Vertex>
                                    <Position>
                                        <LanePosition roadId="[实体当前的road_id]" laneId="[lane_id]" s="[s]" offset="[offset]"/>
                                    </Position>
                                </Vertex>
                            </Polyline>
                        </Shape>
                        <TimeReference>
                            <None/>
                        </TimeReference>
                        <TrajectoryFollowingMode followingMode="position"/>
                    </Trajectory>
                </FollowTrajectoryAction>
            </RoutingAction>
        </PrivateAction>
    </Action>

【支持的 StartTrigger 标签片段 (填入上述 Event 中)】:
- RelativeDistanceCondition (相对距离):
    <StartTrigger>
        <ConditionGroup>
            <Condition name="Cond_[actor_name]_[stage_index]" delay="0" conditionEdge="rising">
                <ByEntityCondition>
                    <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="[actor_name]"/>
                    </TriggeringEntities>
                    <EntityCondition>
                        <RelativeDistanceCondition entityRef="[trigger_ref]" relativeDistanceType="longitudinal" value="[trigger_value]" freespace="true" rule="[trigger_rule]"/>
                    </EntityCondition>
                </ByEntityCondition>
            </Condition>
        </ConditionGroup>
    </StartTrigger>

- TraveledDistanceCondition (行驶距离):
    <StartTrigger>
        <ConditionGroup>
            <Condition name="Cond_[actor_name]_[stage_index]" delay="0" conditionEdge="none">
                <ByEntityCondition>
                    <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="[actor_name]"/>
                    </TriggeringEntities>
                    <EntityCondition>
                        <TraveledDistanceCondition value="[trigger_value]"/>
                    </EntityCondition>
                </ByEntityCondition>
            </Condition>
        </ConditionGroup>
    </StartTrigger>

- EntityReachPosition (到达特定位置):
    <StartTrigger>
        <ConditionGroup>
            <Condition name="Cond_[actor_name]_[stage_index]" delay="0" conditionEdge="none">
                <ByEntityCondition>
                    <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="[actor_name]"/>
                    </TriggeringEntities>
                    <EntityCondition>
                        <ReachPositionCondition tolerance="2.0">
                            <Position>
                                <LanePosition roadId="[road_id]" laneId="[lane_id]" s="[s]"/>
                            </Position>
                        </ReachPositionCondition>
                    </EntityCondition>
                </ByEntityCondition>
            </Condition>
        </ConditionGroup>
    </StartTrigger>

5. 停止触发器 (StopTrigger)：
        <StopTrigger>
            <ConditionGroup>
                <Condition name="StopCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition>
                        <SimulationTimeCondition value="[simulation_duration]" rule="greaterThan"/>
                    </ByValueCondition>
                </Condition>
            </ConditionGroup>
        </StopTrigger>
    </Storyboard>
</OpenSCENARIO>

--- 输入的结构化物理场景文本 (JSON) ---
{structured_scenario_text}

--- 请根据输入 JSON 替换上方模板中的 [变量]，生成严谨的 OpenSCENARIO XML 代码 (仅输出 XML 内容) ---
"""

# 全局变量
CURRENT_LLM_CONFIG: Dict = {}
LLM_INSTANCE: Optional[BaseChatModel] = None
SCENARIO_GENERATION_CHAIN: Optional[RunnableSequence] = None


def _clean_llm_output(raw_output: str) -> str:
    """清洗 LLM 返回的文本，提取纯 XML"""
    cleaned = raw_output.strip()
    
    # 提取 <OpenSCENARIO>...</OpenSCENARIO> 区块
    start_tag = "<OpenSCENARIO"
    end_tag = "</OpenSCENARIO>"
    
    start_idx = cleaned.find(start_tag)
    end_idx = cleaned.rfind(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        return cleaned[start_idx : end_idx + len(end_tag)]
    elif start_idx != -1:
        return cleaned[start_idx:]
    
    return cleaned


def initialize_chain(llm_config: Dict = None) -> bool:
    """初始化 LangChain 链路"""
    global LLM_INSTANCE, SCENARIO_GENERATION_CHAIN, CURRENT_LLM_CONFIG
    
    if llm_config is None:
        llm_config = {
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), 
            "model": os.getenv("LLM_MODEL", "gpt-4o"), 
            "api_key": os.getenv("LLM_API_KEY", "EMPTY")
        }

    if (LLM_INSTANCE is not None
            and CURRENT_LLM_CONFIG.get("base_url") == llm_config["base_url"]
            and CURRENT_LLM_CONFIG.get("model") == llm_config["model"]
            and CURRENT_LLM_CONFIG.get("api_key") == llm_config.get("api_key")):
        return True

    try:
        print(f"[CODE_GEN] 正在初始化模型: {llm_config['model']} ...")
        
        LLM_INSTANCE = ChatOpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config.get("api_key", "EMPTY"),
            model=llm_config["model"],
            temperature=0.0  # 代码生成强制设为0，保证 XML 结构稳定性
        )
        
        prompt = PromptTemplate.from_template(OPENSCENARIO_GENERATION_PROMPT_TEMPLATE)
        output_parser = StrOutputParser()
        SCENARIO_GENERATION_CHAIN = prompt | LLM_INSTANCE | output_parser
        
        CURRENT_LLM_CONFIG = llm_config.copy()
        return True
    except Exception as e:
        print(f"[CODE_GEN] 初始化失败: {e}")
        traceback.print_exc()
        return False


def generate_openscenario(structured_text: str, llm_config: Dict = None) -> str:
    """主入口：将 physical_scenario_generator 输出的 JSON 转换为 .xosc"""
    if not initialize_chain(llm_config):
        return "# ERROR: LLM Initialization Failed"

    try:
        print("[CODE_GEN] 正在将物理场景参数编译为 OpenSCENARIO 代码...")
        
        raw_result = SCENARIO_GENERATION_CHAIN.invoke({
            "structured_scenario_text": structured_text
        })
        
        final_xml = _clean_llm_output(raw_result)
        
        # 基础校验
        validation_errors = []
        if "<EnvironmentAction>" not in final_xml: validation_errors.append("缺少环境初始化")
        if "<CatalogReference" not in final_xml: validation_errors.append("实体模型映射错误")
        if "<StopTrigger>" not in final_xml: validation_errors.append("缺少仿真时长停止触发器")
            
        if validation_errors:
            print(f"[WARNING] 生成的代码存在隐患: {', '.join(validation_errors)}")
            
        return final_xml

    except Exception as e:
        print(f"[CODE_GEN] 生成过程发生异常: {e}")
        traceback.print_exc()
        return f"# ERROR: Generation failed - {str(e)}"


# ==================== 本地联合测试用例 ====================
if __name__ == "__main__":
    # 配置 LLM
    my_llm_config = {
        "base_url": "http://172.20.200.91:8000/v1", 
        "model": "Qwen3.6-27B",  # 填入实际模型名
        "api_key": "EMPTY"
    }

    # 模拟从 physical_scenario_generator 产出的严格 JSON 格式
    test_json_from_generator = """
    {
      "scenario_name": "Dangerous_CutIn",
      "map_key": "Town04.xodr",
      "environment": {
        "cloud_state": "free",
        "precipitation_type": "none",
        "precipitation_intensity": 0.0,
        "fog_visual_range": 100000.0,
        "sun_azimuth": 0.0,
        "sun_elevation": 1.5,
        "sun_intensity": 80000.0
      },
      "entities": [
        {
          "name": "Ego",
          "is_ego": true,
          "model_type": "car_red",
          "init_road_id": 4,
          "init_lane_id": -2,
          "init_s": 20.0,
          "init_speed": 25.0
        },
        {
          "name": "Adversary1",
          "is_ego": false,
          "model_type": "car_blue",
          "init_road_id": 4,
          "init_lane_id": -3,
          "init_s": 35.0,
          "init_speed": 22.0
        }
      ],
      "actions": [
        {
          "stage_index": 0,
          "actions": [
            {
              "actor_name": "Adversary1",
              "type": "LaneChangeAction",
              "params": {
                "target_lane_offset": 1,
                "dynamics_dimension": "distance",
                "dynamics_value": 15.0
              },
              "trigger_type": "RelativeDistanceCondition",
              "trigger_value": 12.0,
              "trigger_ref": "Ego",
              "trigger_rule": "lessThan"
            }
          ]
        }
      ],
      "simulation_duration": 15.0
    }
    """

    print("=== 开始联合代码生成测试 ===")
    xml_output = generate_openscenario(test_json_from_generator, llm_config=my_llm_config)
    
    if xml_output and not xml_output.startswith("# ERROR"):
        output_file = "test_gen.xosc"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_output)
        print(f"\n[SUCCESS] OpenSCENARIO 代码已保存至: {output_file}")
        print("\n=== 代码预览 (前 1000 字符) ===\n")
        print(xml_output[:1000])
    else:
        print("\n[FAILED] 生成失败")

# --- END OF FILE coder_generate.py ---