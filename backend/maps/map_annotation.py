import xml.etree.ElementTree as ET
import json
import os
import re
import math
from collections import defaultdict

# ======================【辅助函数：计算几何方向】======================
def get_cardinal_direction(hdg_radians):
    """
    OpenDRIVE 航向角转罗盘方位
    0=East, 1.57=North, 3.14=West, -1.57=South
    """
    if hdg_radians is None: return "Unknown"
    
    degrees = math.degrees(float(hdg_radians)) % 360
    
    # 45度区间划分
    if 45 <= degrees < 135:
        return "North"
    elif 135 <= degrees < 225:
        return "West"
    elif 225 <= degrees < 315:
        return "South"
    else:
        return "East"

def get_opposite_direction(direction):
    """获取反向方位"""
    pairs = {
        "North": "South", "South": "North",
        "East": "West", "West": "East",
        "Unknown": "Unknown"
    }
    return pairs.get(direction, "Unknown")

# ======================【第一步：原始数据解析】======================
def parse_xodr_tunnel(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    map_annotation = {}

    for road in root.findall('road'):
        r_id = road.get('id')
        length = float(road.get('length'))
        j_id = road.get('junction')
        
        # --- 1. 获取几何航向 (Heading) ---
        start_hdg = 0.0
        plan_view = road.find('planView')
        geo_types = set()
        
        if plan_view:
            geometries = plan_view.findall('geometry')
            if geometries:
                start_hdg = float(geometries[0].get('hdg'))
            
            for geo in geometries:
                if geo.find('line') is not None: geo_types.add("Line")
                elif geo.find('arc') is not None: geo_types.add("Arc")
                elif geo.find('spiral') is not None: geo_types.add("Spiral")
                elif geo.find('poly3') is not None or geo.find('paramPoly3') is not None: geo_types.add("Poly")

        heading_label = get_cardinal_direction(start_hdg)

        # --- 2. 场景与形状 ---
        loc = "Highway" if j_id == "-1" else f"Junction_Link(ID:{j_id})"
        if not geo_types: shape = "Unknown"
        elif geo_types == {"Line"}: shape = "Straight"
        else: shape = f"Curve({'+'.join(sorted(list(geo_types)))})"

        # --- 3. 拓扑连接 (Link) ---
        link = road.find('link')
        link_data = {"predecessor": None, "successor": None}
        raw_link_strs = {} # 暂存 pred 和 succ 的字符串

        for tag in ['predecessor', 'successor']:
            item = link.find(tag) if link is not None else None
            if item is not None:
                eid = item.get('elementId')
                etype = item.get('elementType')
                # 保存结构化数据用于分析
                link_data[tag] = {"type": etype, "id": eid}
                
                cp = item.get('contactPoint', 'end')
                raw_link_strs[tag] = f"{tag[:4]}:{etype}_{eid}({cp})"
            else:
                raw_link_strs[tag] = f"{tag[:4]}:None"
        
        # 原代码直接拼接，这里注释掉，移到后面
        # conn_str = " -> ".join(conn_parts)

        # --- 4. 车道提取 (核心修复：区分正负车道) ---
        lane_groups = defaultdict(list)
        unique_lanes = set()
        has_pos_driving = False # 是否有左侧车道 (反向流: Succ -> Pred)
        has_neg_driving = False # 是否有右侧车道 (正向流: Pred -> Succ)

        lanes = road.find('lanes')
        if lanes:
            for section in lanes.findall('laneSection'):
                # 检查 Left (Positive ID)
                left = section.find('left')
                if left:
                    for lane in left.findall('lane'):
                        if lane.get('type') == 'driving':
                            has_pos_driving = True
                            lane_groups['driving'].append(int(lane.get('id')))

                # 检查 Right (Negative ID)
                right = section.find('right')
                if right:
                    for lane in right.findall('lane'):
                        if lane.get('type') == 'driving':
                            has_neg_driving = True
                            lane_groups['driving'].append(int(lane.get('id')))
                
                # 提取 sidewalks 等其他信息
                for side in ['left', 'right', 'center']:
                    side_elem = section.find(side)
                    if side_elem:
                        for lane in side_elem.findall('lane'):
                            l_id = int(lane.get('id'))
                            l_type = lane.get('type')
                            if l_id != 0 and l_type != 'driving': # driving 已在上面处理
                                lane_groups[l_type].append(l_id)

        # 【新增修复】：根据车道流向决定连接字符串的显示顺序
        # 如果只有左侧车道（Road 4），流向是 Succ -> Pred
        if has_pos_driving and not has_neg_driving:
            conn_str = f"{raw_link_strs['successor']} -> {raw_link_strs['predecessor']}"
        else:
            # 默认情况 Pred -> Succ
            conn_str = f"{raw_link_strs['predecessor']} -> {raw_link_strs['successor']}"

        # 判断流向类型
        if has_pos_driving and has_neg_driving: flow = "Two-Way"
        elif has_pos_driving or has_neg_driving: flow = "One-Way"
        else: flow = "No_Driving"

        # 整理车道字符串
        lanes_str_list = []
        for l_type in sorted(lane_groups.keys()):
            ids = sorted(lane_groups[l_type])
            lanes_str_list.append(f"{l_type}:{ids}")
        final_lanes = " | ".join(lanes_str_list) if lanes_str_list else "None"

        # --- 5. 额外对象 ---
        cw_list = []
        tunnel_list = []
        objs = road.find('objects')
        if objs:
            for o in objs.findall('object'):
                if 'crosswalk' in o.get('type', '').lower() or 'crosswalk' in o.get('name', '').lower():
                    cw_list.append({"id": o.get('id'), "s": round(float(o.get('s')), 2)})
            for t in objs.findall('tunnel'):
                tunnel_list.append({
                    "id": t.get('id'), "type": t.get('type', 'standard')
                })

        extra_tags = []
        if tunnel_list: extra_tags.append("Contains_Tunnel")
        if cw_list: extra_tags.append("Has_Crosswalk")

        info_str = f"Len:{length:.1f}m | {flow} {loc} Shape:{shape}"
        if extra_tags: info_str += f" [{' | '.join(extra_tags)}]"

        map_annotation[f"Road_{r_id}"] = {
            "info": info_str,
            "heading_val": heading_label,   # 暂存：用于判断方位
            "link_data": link_data,         # 暂存：连接关系
            "flow_flags": {                 # 暂存：流向标志
                "has_pos": has_pos_driving, 
                "has_neg": has_neg_driving
            },
            "link": conn_str,
            "lanes": final_lanes,
            "tunnels": tunnel_list,
            "crosswalks": cw_list
        }

    return map_annotation

# ======================【第二步：路口语义分析（最终修复版）】======================
class JunctionAnalyzer:
    def __init__(self, raw_map_data):
        self.raw_data = raw_map_data
        self.junctions = defaultdict(lambda: {
            "type": "Unknown",
            "internal_roads": [],
            "arm_map": {},          # Road_ID -> 方位字符串
            "connected_arms_ids": set(),
            "driving_paths": []
        })

    def analyze(self):
        # 1. 扫描路口内部连接路，生成路径
        for r_id, data in self.raw_data.items():
            j_match = re.search(r"Junction_Link\(ID:(-?\d+)\)", data['info'])
            if j_match:
                j_id = j_match.group(1)
                if j_id != "-1":
                    self._process_internal_road(j_id, r_id, data)
        
        # 2. 扫描外部道路，确定方位 (Arms)
        for r_id, data in self.raw_data.items():
            if "Junction_Link" in data['info']: continue 
            
            hdg = data['heading_val']
            links = data['link_data']
            
            # 情况A: 道路进入路口 (Successor is Junction)
            if links['successor'] and links['successor']['type'] == 'junction':
                target_j_id = links['successor']['id']
                # 逻辑：车朝 HDG 开进路口 -> 路在相反方向
                # 例：车朝 South 开进路口，说明路在 North
                arm_pos = get_opposite_direction(hdg)
                self.junctions[target_j_id]["arm_map"][r_id] = arm_pos
                self.junctions[target_j_id]["connected_arms_ids"].add(r_id)

            # 情况B: 道路离开路口 (Predecessor is Junction)
            if links['predecessor'] and links['predecessor']['type'] == 'junction':
                target_j_id = links['predecessor']['id']
                # 逻辑：车朝 HDG 离开路口 -> 路在相同方向
                # 例：车朝 South 离开路口，说明路在 South
                arm_pos = hdg
                self.junctions[target_j_id]["arm_map"][r_id] = arm_pos
                self.junctions[target_j_id]["connected_arms_ids"].add(r_id)

        # 3. 生成报告
        report = {}
        for j_id, j_data in self.junctions.items():
            arms_ids = sorted(list(j_data["connected_arms_ids"]))
            
            # --- 仅在这里加上方向标注 ---
            formatted_arms = []
            for arm_id in arms_ids:
                direction = j_data["arm_map"].get(arm_id, "Unknown")
                formatted_arms.append(f"{arm_id} ({direction})")

            # 推断类型
            count = len(arms_ids)
            j_type = "Crossroad" if count == 4 else "T-Junction" if count == 3 else "Complex"

            report[f"Junction_{j_id}"] = {
                "type": j_type,
                "arms_count": count,
                "connected_arms": formatted_arms, # 结果如: ["Road_0 (North)", "Road_1 (South)"]
                "driving_paths": sorted(j_data["driving_paths"]),
                "internal_parts": j_data["internal_roads"]
            }
        return report

    def _process_internal_road(self, j_id, r_id, data):
        self.junctions[j_id]["internal_roads"].append(r_id)
        
        if "No_Driving" in data['info']: return

        link_data = data['link_data']
        flags = data['flow_flags']
        
        # 提取连接的外部道路ID
        pred_id = f"Road_{link_data['predecessor']['id']}" if link_data['predecessor'] else None
        succ_id = f"Road_{link_data['successor']['id']}" if link_data['successor'] else None
        
        if pred_id and succ_id:
            # 过滤掉内部路互连的情况
            if not self._is_internal(pred_id) and not self._is_internal(succ_id):
                
                # --- 核心修复：根据车道正负判断流向 ---
                
                # 1. 负向车道 (Right Lanes) -> 意味着正向行驶 (Pred -> Succ)
                if flags['has_neg']:
                    self.junctions[j_id]["driving_paths"].append(f"From {pred_id} to {succ_id} via {r_id}")
                
                # 2. 正向车道 (Left Lanes) -> 意味着反向行驶 (Succ -> Pred)
                # 这就是 Road 27 (连接 Road 1 -> Road 0) 的情况
                if flags['has_pos']:
                    self.junctions[j_id]["driving_paths"].append(f"From {succ_id} to {pred_id} via {r_id}")

    def _is_internal(self, r_key):
        if r_key not in self.raw_data: return False
        return "Junction_Link" in self.raw_data[r_key]['info']

# ======================【处理单个地图文件】======================
def process_single_map(file_path):
    """
    处理单个xodr地图文件
    """
    # 1. 解析
    raw_result = parse_xodr_tunnel(file_path)

    # 2. 分析
    analyzer = JunctionAnalyzer(raw_result)
    junction_summary = analyzer.analyze()

    # 3. 清理 raw_roads (移除临时字段，保持输出纯净)
    for r_data in raw_result.values():
        r_data.pop('heading_val', None)
        r_data.pop('link_data', None)
        r_data.pop('flow_flags', None)

    # 4. 构造最终输出
    final_output = {
        "summary": {
            "total_roads": len(raw_result),
            "total_junctions": len(junction_summary)
        },
        "junctions_semantic": junction_summary,
        "raw_roads": raw_result
    }
    
    return final_output

# ======================【批量处理所有地图文件】======================
def process_all_maps_in_folder(folder_path, output_file="all_maps_annotations.json"):
    """
    处理文件夹下所有xodr地图文件
    """
    # 获取所有xodr文件
    xodr_files = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.xodr'):
            xodr_files.append(os.path.join(folder_path, file_name))
    
    if not xodr_files:
        print(f"在文件夹 {folder_path} 中未找到.xodr文件")
        return
    
    print(f"找到 {len(xodr_files)} 个xodr文件:")
    for f in xodr_files:
        print(f"  - {os.path.basename(f)}")
    
    # 处理每个文件
    all_maps_annotations = {}
    
    for file_path in xodr_files:
        map_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n正在处理: {map_name}...")
        
        try:
            annotation = process_single_map(file_path)
            all_maps_annotations[map_name] = annotation
            print(f"  ✓ 完成处理: {map_name}")
        except Exception as e:
            print(f"  ✗ 处理失败: {map_name} - 错误: {str(e)}")
            all_maps_annotations[map_name] = {
                "error": str(e),
                "summary": {"total_roads": 0, "total_junctions": 0},
                "junctions_semantic": {},
                "raw_roads": {}
            }
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_maps_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 处理完成！所有地图标注已保存至: {output_file}")
    print(f"  共处理了 {len(all_maps_annotations)} 个地图文件")
    
    # 打印统计信息
    total_roads = 0
    total_junctions = 0
    for map_name, data in all_maps_annotations.items():
        if "error" not in data:
            total_roads += data["summary"]["total_roads"]
            total_junctions += data["summary"]["total_junctions"]
    
    print(f"  总计: {total_roads} 条道路, {total_junctions} 个路口")
    
    return all_maps_annotations

# ======================【运行与输出】======================
if __name__ == "__main__":
    # 设置文件夹路径（修改为你自己的文件夹路径）
    input_folder = "."  # 当前目录，可以修改为其他路径，如 "./maps"
    
    # 或者从命令行参数获取文件夹路径
    import sys
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    
    # 处理所有地图
    process_all_maps_in_folder(input_folder)