---
name: parameter-filling-rules
description: Strict formulas for physical scenario parameters
---

# Parameter Filling Rules & Mathematical Formulas

## Global Physical Unit Declaration
- Length: m
- Speed: m/s (36km/h=10m/s)
- Time: s
- Direction Rules:
  - `lane_id < 0`: Vehicle travels in the same direction as the S-axis. Forward coordinate = `current S + distance`.
  - `lane_id > 0`: Vehicle travels in the opposite direction of the S-axis. Forward coordinate = `current S - distance`.

## I. Environment
Mapping based on weather vocabulary (default: free/none/0.0/100000.0/0.0/1.5/100000.0):
- `cloud_state`: Sunny `free` | Overcast `overcast` | Rainy/Snowy `rainy`
- `precipitation_type`: None `none` | Rain `rain` | Snow `snow`
- `precipitation_intensity`: None `0.0` | Light `0.3` | Moderate `0.5` | Heavy `0.8`
- `fog_visual_range`: Clear `100000.0` | Mist `100.0` | Fog `30.0`
- `sun_elevation`: Daytime `1.5` | Dusk `0.2` | Night `-1.5`
- `sun_intensity`: Daytime `100000.0` | Cloudy/Dusk `30000.0` | Night `0.0`

## II. Simulation Duration
- `simulation_duration`: Standard scenario `10.0` | Complex junctions / Long-range interaction `15.0`

## III. Entities
### Global Physical Unit Declaration
- Length: m
- Speed: m/s (36km/h=10m/s)
- Time: s
- Direction Rules:
  - `lane_id < 0`: Vehicle travels in the same direction as the S-axis. Forward coordinate = `current S + distance`.
  - `lane_id > 0`: Vehicle travels in the opposite direction of the S-axis. Forward coordinate = `current S - distance`.

### 1. Ego (Mandatory first entry)
- name: "Ego", is_ego: True, vehicle_model: "car_white"
- init_road_id: Select a sufficiently long starting road
- init_lane_id: Select a `driving` lane
- init_s: Leave enough run-up distance. If `lane_id < 0`, set to `15.0`; if `lane_id > 0`, set to `(Road_Len - 15.0)`
- init_speed: Urban `15.0` | Highway `25.0`

### 2. NPC/Adversary (Derived from Ego)
- name: "AdversaryX" / "PedestrianX", is_ego: False
- models: car_red, truck, pedestrian_adult, bicycle, motorcycle
- Scenario Formulas (Set `Ego_s` as ego vehicle S):
  - **A. Overtaking**: Same direction, adjacent lane. $V_{npc}=V_{ego}+8.0$. $S_{npc} = Ego\_s \pm 12.0$ (use `-` for negative lane, `+` for positive lane)
  - **B. Cut-in**: Same direction, adjacent lane. $V_{npc}=V_{ego}-8.0$. $S_{npc} = Ego\_s \pm 12.0$ (use `+` for negative lane, `-` for positive lane)
  - **C. Lead Braking**: Same lane. $V_{npc}=V_{ego}$. $S_{npc} = Ego\_s \pm 15.0$ (use `+` for negative, `-` for positive)
  - **D. Ghost Probe**: Requires 2 NPCs.
    - Blocker: Same direction side-front lane. $V=0.0$. $S = Ego\_s \pm 35.0$ (use `+` for negative, `-` for positive)
    - Pedestrian: Same lane as Blocker. $V=1.5$. $S = Blocker\_s \pm 3.5$ (use `+` for negative, `-` for positive)
  - **E. Junction Merge (T-bone)**: $T = D_{ego}/V_{ego}$. $D_{npc} = V_{npc} \times T$. If entering junction along S (`lane<0`), $S=Road\_Len-D_{npc}$; if entering against S (`lane>0`), $S=D_{npc}$.

## IV. Actions
1. **LaneChangeAction**: `target_lane_offset` (Left `1`, Right `-1`), `dynamics_dimension`: "distance", `dynamics_value`: lane change distance.
2. **SpeedAction**: `target_speed` (Target m/s), `dynamics_dimension`: "distance", `dynamics_value`: braking/acceleration distance.
3. **AssignRouteAction**: Check junction semantics `From Road_A to Road_B via Road_C`. `waypoints` sequence = [Start point (Road_A, near end), Internal lane (Road_C, just entered), End point (Road_B, just entered)].
4. **FollowTrajectoryAction** (Crossing):
   - Check `lanes` width. Calculate left/right curb offsets: $Offset_{left} = +(width_{left} + 2.0)$, $Offset_{right} = -(width_{right} + 2.0)$.
   - `vertices` sequence requires small longitudinal increment (to simulate time taken): [{"s": $S$, "offset": side1}, {"s": $S+1.5$, "offset": 0}, {"s": $S+3.0$, "offset": side2}].

## V. Triggers
1. **Immediate Execution**: `trigger_type`: "TraveledDistanceCondition", `trigger_ref`: Self name, `trigger_value`: `0.1`
2. **Relative Distance (Strong Interaction)**: `trigger_type`: "RelativeDistanceCondition", `trigger_ref`: Must fill with target NPC or Ego name!
   - Lead vehicle braking: $value = (V_{ego} - V_{npc}/2) \times 1.5$
   - Side vehicle cut-in: $value = (V_{ego} - V_{npc}) \times 1.5 + 3.0$
   - Ghost probe: $value = (lateral\_dist / V_{npc} + 1.0) \times V_{ego}$
3. **Reach Coordinate (Junction/Path)**: `trigger_type`: "EntityReachPosition", `trigger_ref`: Self name, `trigger_value`: {"road_id": X, "lane_id": Y, "s": Z}.
   - Trigger point Z is set 5.0m before the end of the road. If `lane<0`, take $Road\_Len-5.0$; if `lane>0`, take `5.0`.