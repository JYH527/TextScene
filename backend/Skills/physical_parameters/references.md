---
name: parameter-filling-rules
description: Detailed rules and mathematical formulas for filling in physical scenario parameters
---

# Parameter Filling Rules & Mathematical Formulas
## 全局物理单位声明 (CRITICAL)
- **距离/长度**: 米 (m)
- **速度**: 米/秒 (m/s)。(注：36 km/h = 10 m/s; 72 km/h = 20 m/s)
- **时间**: 秒 (s)
- **车道正负方向法则**:
  - `lane_id < 0` (负数车道): 车辆行驶方向与 S 轴增加方向**相同**。前方坐标 = `当前S + 距离`。
  - `lane_id > 0` (正数车道): 车辆行驶方向与 S 轴增加方向**相反**。前方坐标 = `当前S - 距离`。
---

## 一、Environment (环境参数)
根据逻辑描述中的天气词汇进行映射：
1. **environment.cloud_state**: 
   - 晴天/默认: `"free"`
   - 阴天: `"overcast"`
   - 雨/雪天: `"rainy"`
2. **environment.precipitation_type**: 
   - 无雨雪: `"none"`
   - 雨天: `"rain"`
   - 雪天: `"snow"`
3. **environment.precipitation_intensity**: 
   - 必须在 `[0.0, 1.0]` 之间。小雨: `0.3`，中雨: `0.5`，暴雨/暴雪: `0.8`。无雨雪为 `0.0`。
4. **environment.fog_visual_range**: 
   - 晴朗/无雾: `100000.0`
   - 轻度雾/视线受阻: `100.0`
   - 大雾/严重遮挡: `30.0`
5. **environment.sun_azimuth**: 默认填 `0.0`。
6. **environment.sun_elevation**: 
   - 白天: `1.5`
   - 黄昏: `0.2`
   - 夜晚: `-1.5` (太阳在地平线以下)
7. **environment.sun_intensity**: 
   - 白天: `100000.0`
   - 阴天/黄昏: `30000.0`
   - 夜晚: `0.0`
---

## 二、仿真时间 (Simulation Time)
- **simulation_duration**:
  - 一般场景 (单次变道/急刹): `20.0`
  - 复杂场景 (过复杂路口/长距离交互): `30.0` 或 `40.0`
---

## 三、Entities (实体初始状态参数)

### 3.1 当 Entity 为 Ego 时 (必须最先填充)
1. **name**: 必须为 `"Ego"`
2. **is_ego**: `True`
3. **vehicle_model**: 轿车填 `"car_white"`
4. **init_road_id**: 选择地图拓扑中长度足够、符合逻辑场景描述的起始道路。
5. **init_lane_id**: 只能从所选道路的 `driving` 列表中选取。
6. **init_s**: (必须留足预跑距离)
   - 若所在的 `lane_id < 0`: `15.0 < init_s < 30.0`
   - 若所在的 `lane_id > 0`: `(Road_Len - 30.0) < init_s < (Road_Len - 15.0)`
7. **init_speed**: 
   - 城市普通道路: `8.0` 到 `12.0` (约30-40km/h)
   - 城市快速路/高速: `15.0` 到 `22.0` (约50-80km/h)

### 3.2 当 Entity 不为 Ego 时 (NPC/Adversary)
所有 NPC 的初始位置和初始状态必须**基于 Ego 的位置和设定的时空公式**进行推算！
1. **name**: 填充为 `"Adversary1"`, `"Adversary2"`, `"Pedestrian1"` 等。
2. **is_ego**: `False`
3. **vehicle_model**: 
   - 轿车: `"car_red"` 或 `"car_black"`
   - 货车: `"truck"`
   - 行人: `"pedestrian_adult"`
   - 自行车: `"bicycle"`
   - 摩托车: `"motorcycle"`
4. **init_road_id & init_lane_id & init_s & init_speed**: 
   #### 场景 A: 超车场景 (Overtaking)
   - **逻辑**: NPC 在 Ego 后方，且速度大于 Ego，从Ego同向的相邻车道超越。
   - `init_road_id`: 与Ego相同。
   - `init_lane_id`: 与 Ego 同向的左侧或右侧相邻`shoulder`或`driving`车道,必须同向。
   - `init_speed`: $V_{npc} = V_{ego} + 5.0$
   - `init_s` 计算 (追赶时间设为 $t = 3.0$ 秒):
     - 若 `lane_id < 0`: $init\_s = Ego\_s - 15.0$
     - 若 `lane_id > 0`: $init\_s = Ego\_s + 15.0$

   #### 场景 B: 快速切入场景 (Cut-in)
   - **逻辑**: NPC 在 Ego 侧前方，速度略慢或持平，然后突然变道至 Ego 前方。
   - `init_road_id`: 与Ego相同。
   - `init_lane_id`: Ego 的左侧或右侧同向相邻`shoulder`或`driving`车道，必须同向。
   - `init_speed`: $V_{npc} \approx V_{ego} - 2.0$ (制造速度差逼近)
   - `init_s` (NPC设置在Ego前方 15 米处):
     - 若 `lane_id < 0`: $init\_s = Ego\_s + 15.0$
     - 若 `lane_id > 0`: $init\_s = Ego\_s - 15.0$

   #### 场景 C: 跟车与前车急刹 (Lead Vehicle Braking)
   - **逻辑**: NPC 在 Ego 正前方同车道行驶，随后急刹。
   - `init_road_id`: 与Ego相同。
   - `init_lane_id`: 必须与 Ego 的 `init_lane_id` 完全相同。
   - `init_speed`: $V_{npc} = V_{ego}$ (保持匀速)
   - `init_s`: (设置安全跟车距离约 20 米)
     - 若 `lane_id < 0`: $init\_s = Ego\_s + 20.0$
     - 若 `lane_id > 0`: $init\_s = Ego\_s - 20.0$

   #### 场景 D: 鬼探头 (Ghost Probe / Blind Crossing) - 需要 2 个 NPC
   - **遮挡车辆 (Adversary1_Blocker)**:
     - `init_road_id`: 与Ego相同。
     - `init_lane_id`: Ego 所在车道的侧前方`shoulder`或`driving`车道上。
     - `init_speed = 0.0` (静止)
     - `init_s`: 置于 Ego 前方 30 米处。
        - 若 `lane_id < 0`: $init\_s = Ego\_s + 30.0$
        - 若 `lane_id > 0`: $init\_s = Ego\_s - 30.0$
   - **横穿行人/电瓶车 (Pedestrian1)**:
     - `init_road_id`: 与Ego相同。
     - `init_lane_id`: 与Adversary在同一车道上。
     - `init_speed = 1.5` (步行) 或 `2.0` (跑步) 或 `5.0` (电瓶车)
     - `init_s`: 置于 Adversary 前方 3.5 米处。
        - 若 `lane_id < 0`: $init\_s = Adversary\_s + 3.5$
        - 若 `lane_id > 0`: $init\_s = Adversary\_s - 3.5$

   #### 场景 E: 路口通行冲突 (Intersection Crossing / T-bone)
    - **逻辑**: NPC 与 Ego 从不同道路驶入同一路口，并在路口内产生轨迹交汇（如侧向直行、对向左转等）。
    - `init_road_id`: 与 Ego 不同。查阅地图 `junctions_semantic`，找到与 Ego 即将进入的同一 Junction 相连的其他道路（侧向臂或对向臂）。
    - `init_lane_id`: 所选初始道路的合法 `driving` 车道。
    - `init_speed`: 根据路口限速设定，如 $8.0$ 到 $12.0$。
    - `init_s` 计算 (时空同步 TTC 法则):
    - 假设 Ego 到达路口的距离为 $D_{ego}$，则 Ego 到达路口耗时 $T = \frac{D_{ego}}{V_{ego}}$。
    - 为了在路口完美相遇，NPC 初始位置距离路口的距离必须是 $D_{npc} = V_{npc} \times T$。
    - 结合 NPC 车道方向：
        - 若 NPC 所在道路连接路口的一端是 `end` (即顺着 S 驶入路口，且 `lane_id < 0`)：$init\_s = Road\_Len - D_{npc}$
        - 若 NPC 所在道路连接路口的一端是 `start` (即逆着 S 驶入路口，且 `lane_id > 0`)：$init\_s = 0.0 + D_{npc}$
---

## 四、Actions 行为参数法则
1. **变道动作 (LaneChangeAction)**
   - `type`: `"LaneChangeAction"`
   - `params` (包含下面三个值): 
     - `"target_lane_offset"`: 左变道填 `1`，右变道填 `-1`。（严禁使用绝对车道号）
     - `"dynamics_dimension"`: `"distance"`
     - `"dynamics_value"`: `完成变道所需的距离
2. **变速/急刹动作 (SpeedAction)**
   - `type`: `"SpeedAction"`
   - `params`(包含下面三个值): 
     - `"target_speed"`: 减速或加速的目标速度。
     - `"dynamics_dimension"`: `"distance"`
     - `"dynamics_value"`: 刹车距离(距离越短刹车越猛)。
3. **跨路口/导航动作 (AssignRouteAction)**
   - `type`: `"AssignRouteAction"`
   - `params`: 包含 `"waypoints"` 列表，必须查阅地图的 `junctions_semantic` 获取连接道路ID。
   **推导步骤 (Algorithm):**
    1. **确定起点和终点**：假设从 `Road_A` 行驶到 `Road_B`。
    2. **查阅地图语义 (Junctions Semantic)**：
    - 搜索地图数据中的 `junctions_semantic`。
    - 找到 `From Road_A to Road_B` 经过所有的 Junction。
    - 提取对应的内部连接道 (`via Road_C`)。
    3. **构建 Waypoints 序列**：
    - 路径必须保证连续性：起点道路Waypoint -> 内部连接道Waypoint -> 终点道路Waypoint。
    - 每个 Waypoint 包含：`{"road_id": int, "lane_id": int, "s": float}`。
   **示例推导 (以驶过十字路口为例)**：
    - *起点*: Road_15 (向北直行)
    - *终点*: Road_10
    - *查阅地图*: Junction_32 中写明 `"From Road_15 to Road_10 via Road_36"`。
    - *计算 `s` 和 `lane_id`*: 
    - 第一点(起点): `Road_15`, `lane_id: -1` (查阅Road_15的driving), `s: Road_15_Len - 10.0` (接近路口尽头)。
    - 第二点(内部): `Road_36`, `lane_id: -1`, `s: 2.0` (刚进入路口)。
    - 第三点(终点): `Road_10`, `lane_id: -3`, `s: 10.0` (完成跨越进入新道路)。
4. **横穿马路动作 (FollowTrajectoryAction)**
   - `type`: `"FollowTrajectoryAction"`
   - `params`: 包含 `"vertices"`，通过改变 `offset` 正负值实现横向移动。
    **推导步骤 (Algorithm):**
    1. **确定横穿纵向基准点**：确定实体（行人/非机动车）开始横穿马路时所在的 `init_s`。
    2. **解析道路横向拓扑 (计算 Offset 极值)**：
       - OpenDRIVE 坐标系中，中心线左侧为正 (+)，右侧为负 (-)。
       - 查阅地图中该道路的 `lanes` 属性。得到各车道宽度为，设路外侧人行道缓冲带为 $2.0m$。
       - 计算得到正数车道`driving`和`shoulder`车道宽度总和为 $width_{left}$，计算左侧路缘最大偏移量：$Offset_{left} = +(width_{left} + 2.0)$
       - 计算得到负数车道`driving`和`shoulder`车道宽度总和为 $width_{right}$，计算右侧路缘最大偏移量：$Offset_{right} = -(width_{right} + 2.0)$
    3. **构建 Vertices 序列 (物理轨迹点)**：
       - 根据横穿方向（从右到左）依次排列 `offset`。
       - **极其重要**：由于实体横穿马路需要时间，在这段时间内必然会产生微小的纵向位移（顺延漂移）。因此，每个顶点（Vertex）的 $s$ 坐标必须依次微增（每次增加 $1.5m$）。
    **示例推导 (以行人从道路右侧横穿到左侧为例)**：
    - *前置条件*: 行人准备在 `Road_5` 横穿，纵坐标位置为 `init_s`: 20。
    - *查阅地图*: `Road_5` 是一条双向四车道，包含正数driving车道 `[1, 2]` 和负数driving车道 `[-1, -2]`。
    - *计算边界 Offset*: 
      - 右侧路缘（起点）: $Offset_{right} = -(2 \times width + 2.0) = -9.0$
      - 左侧路缘（终点）: $Offset_{left} = +(2 \times width + 2.0) = +9.0$
    - *构建最终轨迹 (Vertices)*: 
      - 第一点(起点 - 右路缘隐藏): `{"s": 20.0, "offset": -9.0}`
      - 第二点(中点 - 越过道路中心线): `{"s": 21.5, "offset": 0.0}` (伴随步行耗时，纵向 $s$ 顺延漂移了 $1.5m$)
      - 第三点(终点 - 到达左侧路缘): `{"s": 23.0, "offset": 9.0}` (纵向 $s$ 再次顺延 $1.5m$)

## 五、触发器 (Triggers)
动作必须绑定合法的触发器。严禁逻辑自相矛盾。
### 1. 立即执行 (Initialization)
- **适用场景**：起步加速、保持匀速、初始路径分配（AssignRoute）等不需要动态前提条件，在仿真一开始就应该执行的行为。
- **参数设置与计算逻辑**：
  - `trigger_type`: `"TraveledDistanceCondition"`
  - `trigger_ref`: 实体自身的名字（如 `"Adversary1"` 或 `"Ego"`）。
  - `trigger_value`: `0.1`
    - **计算逻辑**：代表实体自生成后只要移动了 0.01 米（极小值，近似于仿真开始的瞬间），就立刻触发并持续执行该动作。
### 2. 基于相对距离触发 (Relative Distance)
- **适用场景**：前车急刹、旁车切入、鬼探头行人冲出等基于两者空间博弈的强交互场景。
- **参数设置与计算逻辑**：
  - `trigger_type`: `"RelativeDistanceCondition"`
  - `trigger_ref`: 参考实体名称（必须填 `"Ego"`，表示当 Ego 靠近该 NPC 时触发 NPC 的动作）。
  - `trigger_value`: 距离阈值浮点数，设置场景的 TTC 为 1.5s 进行反推计算：
    - **场景 A：前车急刹 (Lead Vehicle Braking)**
      - **逻辑**：当 Ego 逼近前方同车道 NPC 到一定距离时，NPC 突然触发急刹动作。
      - **计算逻辑**：制造紧急避险场景，公式为 $trigger\_value = (V_{ego} - V_{npc}/2) \times 1.5$。
    - **场景 B：旁车切入 (Cut-in)**
      - **逻辑**：旁侧车道 NPC 速度较慢，在 Ego 即将超越时突然变道至 Ego 前方。
      - **计算逻辑**：制造危险切入场景，公式为 $trigger\_value = (V_{ego} - V_{npc}) \times 1.5 + 3.0$（预留 3.0 米的最小车距缓冲）。
    - **场景 C：鬼探头/行人冲出 (Ghost Probe / Blind Crossing)**
      - **逻辑**：行人/非机动车隐藏在静止遮挡物后，当 Ego 靠近到极限反应距离时起步横穿。
      - **计算逻辑**：解析地图标注得到和npc的`offset`得到npc和Ego的横向距离`width`。制造危险行人冲出场景，公式为 $trigger\_value = (`width` / V_{npc} + 1.0) \times V_{ego}。
### 3. 到达绝对坐标点触发 (Reach Position)
- **适用场景**：车辆行驶到特定的路口准备转弯/跨越等强依赖地图拓扑位置的动作。
- **参数设置与计算逻辑**：
  - `trigger_type`: `"EntityReachPosition"`
  - `trigger_ref`: 实体自身的名字。
  - `trigger_value`: 目标坐标字典 `{"road_id": X, "lane_id": Y, "s": Z}`，其 `s` 的具体计算需严格遵守车道正负方向法则：
    - **场景 A：路口转弯/路径分配 (AssignRouteAction)**
      - **逻辑**：车辆在当前道路行驶，快要驶入交叉路口（Junction）时触发连续的 Waypoints 导航动作。
      - **计算逻辑**：为了保证动作连贯，触发点应设置在当前道路尽头前方约 5.0 到 10.0 米处。
        - `road_id` 与 `lane_id`: 实体当前所在的道路和车道。
        - 计算 `s` ($Z$):
          - 若当前 `lane_id < 0`（顺着 S 轴方向行驶）：$Z = Road\_Len - 5.0$
          - 若当前 `lane_id > 0`（逆着 S 轴方向行驶）：$Z = 5.0$