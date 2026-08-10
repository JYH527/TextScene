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

# ==================== 核心 Prompt 模板 (针对 Carla深度优化) ====================
OPENSCENARIO_GENERATION_PROMPT_TEMPLATE = """
Role:
你是一位精通 OpenSCENARIO 1.0 标准的 OpenSCENARIO 代码修改专家。
Task:
请根据【OpenSCENARIO 1.0 代码转化规则】，将提供的【可在 esmini 上运行的 OpenSCENARIO代码】转化为【可在 Carla 上运行的 OpenSCENARIO 1.0 代码】。

【OpenSCENARIO 1.0 代码转化规则】：
1.资源导入路径修改：
    原有的 <CatalogLocations> 标签：
    <CatalogLocations>
        <VehicleCatalog>
            <Directory path="../resources/xosc/Catalogs/Vehicles"/>
        </VehicleCatalog>
        <PedestrianCatalog>
            <Directory path="../resources/xosc/Catalogs/Pedestrians"/>
        </PedestrianCatalog>
    </CatalogLocations>

    替换后的 <CatalogLocations> 标签：
    <CatalogLocations>
        <VehicleCatalog>
            <Directory path="../scenario_runner/srunner/examples/catalogs/"/>
        </VehicleCatalog>
        <PedestrianCatalog>
            <Directory path="../scenario_runner/srunner/examples/catalogs/"/>
        </PedestrianCatalog>
    </CatalogLocations>

[2. 实体模型替换与标准化]
- CARLA 支持的车型列表严格限定为: vehicle.tesla.model3, vehicle.kawasaki.ninja, vehicle.toyota.prius, vehicle.audi.a2, vehicle.bmw.grandtourer, vehicle.tesla.cybertruck, walker.pedestrian.00012。
- 请根据原有实体名称或特性（如 car_red, car_blue）替换为上述 Carla 模型（默认可统一使用 vehicle.tesla.model3）。
- 若原实体为行人 (Pedestrian), 必须使用: walker.pedestrian.00012。
- 静态障碍物必须使用 MiscObject, 且 name 必须为 "static.prop.box01", miscObjectCategory 必须为 "obstacle"。
- ⚠️ 车辆和行人必须使用 <CatalogReference> 标签来定义！

转化示例：
    <Entities>
        <ScenarioObject name="Ego">
            <CatalogReference catalogName="VehicleCatalog" entryName="vehicle.tesla.model3"/>
        </ScenarioObject>
        <ScenarioObject name="Adv_1">
            <CatalogReference catalogName="VehicleCatalog" entryName="vehicle.toyota.prius"/>
        </ScenarioObject>
        <ScenarioObject name="Adv_2">
            <MiscObject mass="5" name="static.prop.box01" miscObjectCategory="obstacle">
                <BoundingBox>
                    <Center x="0" y="0" z="0.75"/>
                    <Dimensions length="1.5" width="1.5" height="1.5"/>
                </BoundingBox>
                <Properties/>
            </MiscObject>
        </ScenarioObject>
    </Entities>

[3. 轨迹顶点 (Vertex) 的时间属性强制补充 (致命报错防护)]
- Carla 对 XSD 校验极度严格！在 <Trajectory> -> <Shape> -> <Polyline> 中，所有的 <Vertex> 标签【必须】包含 time 属性！
- 规则：如果原始 Esmini 代码的 <Vertex> 缺少 time 属性，你必须为它们补充逻辑递增的 time 值（例如第一个顶点补充 time="0.0", 第二个补充 time="5.0" 等）。
  错误示例: <Vertex><Position>...</Position></Vertex>
  正确输出: <Vertex time="0.0"><Position>...</Position></Vertex>

[4. 天气与环境枚举值修正 (致命报错防护)]
- Carla 对 XML Schema 校验极度严格。检查 <Precipitation> 标签，如果存在 precipitationType="none"，必须将其强制修改为 precipitationType="dry"！
  错误示例: <Precipitation precipitationType="none" intensity="0.0"/>
  正确输出: <Precipitation precipitationType="dry" intensity="0.0"/>

[5. 动作与触发器兼容性修正]
- 触发器边缘：确保所有 <Condition> 标签必须包含 conditionEdge="rising" 属性，否则 Carla 无法触发事件。
- 动态形状：确保所有的 <LaneChangeActionDynamics> 和 <SpeedActionDynamics> 的 dynamicsShape 优先修改为 "linear" 或 "step"，严禁使用 "sinusoidal"。

[6. 杂项与边界保护]
- 严格保持 XML 标签的闭合、嵌套层级和缩进。
- 绝不臆造 CARLA 不支持的非标准标签。

=== 原始 Esmini 代码输入 ===
{structured_scenario_text}

=== 输出要求 ===
直接输出重构后的纯 XML 代码，不要包含任何额外的解释文本、Markdown 标记或 ```xml 代码块修饰符。
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
        "model": "Qwen3.6-35B-A3B",  # 填入实际模型名
        "api_key": "EMPTY"
    }

    print("=== 开始 Esmini -> Carla OpenSCENARIO 代码转化测试 ===")

    # 1. 尝试读取上一个脚本生成的 Esmini 格式 XML 文件
    input_file_path = "test_gen.xosc"
    esmini_xml_input = ""

    if os.path.exists(input_file_path):
        print(f"[INFO] 找到本地文件 {input_file_path}，正在读取作为输入...")
        with open(input_file_path, "r", encoding="utf-8") as f:
            esmini_xml_input = f.read()
    else:
        print(f"[INFO] 未找到 {input_file_path}，使用默认的 Esmini 测试 XML 字符串...")

    # 3. 调用生成函数
    xml_output = generate_openscenario(esmini_xml_input, llm_config=my_llm_config)

    if xml_output and not xml_output.startswith("# ERROR"):
        # 将转换后的 Carla 版本单独保存
        output_file = "test_carla_gen.xosc"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_output)
        print(f"\n[SUCCESS] Carla 版本的 OpenSCENARIO 代码已保存至: {output_file}")

        # 简单验证一下转换效果
        if "../srunner/examples/catalogs/" in xml_output:
            print("[CHECK] 目录转换成功: 包含了 Carla ScenarioRunner 的目录。")
        if "car_red" not in xml_output and "car_blue" not in xml_output:
            print("[CHECK] 车型转换成功: Esmini 的旧车型已被替换。")

        print("\n=== 代码预览 (前 1000 字符) ===\n")
        print(xml_output[:1000])
    else:
        print("\n[FAILED] 转换失败，LLM 返回异常。")

# --- END OF FILE coder_generate.py ---