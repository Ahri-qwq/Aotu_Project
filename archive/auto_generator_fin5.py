import os
import sys
import json
import time
import numpy as np
from http import HTTPStatus
from dotenv import load_dotenv
from dashscope import Generation
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin

# ================= 路径配置 ================= 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from paths import BAND_DIR, CASE_DIR


# ================= 配置区域 =================
INPUT_DIR = BAND_DIR
OUTPUT_DIR = CASE_DIR
MAX_BATCH_SIZE = 5          #批量处理参数
# ===========================================

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("[ERROR] 未读取到 DASHSCOPE_API_KEY。")
    sys.exit(1)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============== Prompt 模板 =================
PROMPT_1 = r"""
你是一位顶级计算材料学专家。你需要基于我提供的【第一性原理计算数据】（含精确带隙与有效质量），构建一份具有深度的科研分析报告。

### 核心任务：
1. **身份确认**：基于化学式和空间群，建立材料微观图像。
2. **数据锚定 (Data Anchoring)**：
   * 我提供了 VBM/CBM 的精确位置以及**有效质量 (Effective Mass)** 数据。
   * 有效质量直接决定了载流子迁移率 ($\mu = e\tau/m^*$)，这是评估器件性能（如响应速度、电导率）的关键。
   * 请直接引用这些数据。

### 执行逻辑：
1. **动力学分析**：
   - 如果 $m^*$ 很小（如 < 0.3 $m_e$），这意味着什么？（高迁移率 -> 高速器件）
   - 如果 $m^*$ 很大（如 > 1.0 $m_e$），这意味着什么？（局域化强 -> 可能适合热电或平带催化）
2. **逆向构思**：
   - 结合带隙值和有效质量，设计一个完美的“逆向因果”场景。
   - 例如：我们需要一个“高速光电探测器”，因此选择了这种“直接带隙且电子有效质量极小”的材料。

**请输出提取的“锚点”及其对应的“逆向科研场景”构思。**
"""

PROMPT_2 = r"""
很好。请利用上述分析，直接输出一份**“自验证型”深度科研计算案例**。

### 核心约束：
1. **直接输出**：不包含任何寒暄，**直接从 Markdown 一级标题（材料名称）开始**。
2. **数据闭环**：在报告的任何“验证依据”或“数据锚点”部分，必须**逐字引用** Context 中计算出的真值（如 4.1157 eV, 0.25 m_e）。
3. **学术风格**：保持严谨，区分“理论预期”与“计算事实”，使用不等式描述物理判据（如 $E_g > 3.0$ eV）。

### 请参照以下 Markdown 格式输出（章节数量根据锚点数量自适应）：

# [材料化学式]：[核心物理特性摘要]

**材料背景与科研价值**：
*(简练综述材料结构与地位，以及为何需要进行高精度第一性原理计算。)*

## 1. [维度一标题]
* **科学问题**：
  *(逆向推导出的科研疑问。例如：为何实验观测到 [现象]？是否源于 [微观机制]？)*
* **关键性质（计算目标）**：
  * **目标指标**：*(例如：计算沿 [路径] 的有效质量)*
* **验证依据 (Ground Truth)**：
    * **理论判据**：*(物理上的定性判断，如 m* < 0.5 m_e 意味着高迁移率)*
    * **数据锚点**：*(在此处引用精确计算值...)*

## 2. [维度二标题]
*(...)*

... (根据实际锚点数量继续列出)
"""

# ============= 核心科学计算函数 ============== 

def calc_effective_mass(bs, band_idx, k_idx, spin):
    """
    计算有效质量 (抛物线拟合)。
    """
    try:
        energies = bs.bands[spin][band_idx]
        distances = bs.distance
        num_k = len(energies)
        
        # 智能窗口 (5点)
        window = 5
        half = window // 2
        
        if k_idx < half:
            start, end = 0, window
        elif k_idx > num_k - half - 1:
            start, end = num_k - window, num_k
        else:
            start, end = k_idx - half, k_idx + half + 1
            
        start, end = max(0, start), min(num_k, end)
        
        y = energies[start:end]
        x = distances[start:end]
        
        if len(x) < 3: return "N/A (Points<3)"
        
        # 拟合 E = ak^2 + bk + c
        coeffs = np.polyfit(x, y, 2)
        a = coeffs[0]
        
        if abs(a) < 1e-4: return "Heavy (>10 m_e)"
        
        # m* = hbar^2 / 2a approx 3.81 / a
        m_eff = 3.81 / a
        return f"{abs(m_eff):.3f} m_e"

    except Exception:
        return "CalcErr"

def process_physics_data(json_path):
    """
    解析数据 -> 计算物理量 -> 生成 Context
    (融合了鲁棒性读取与科学计算逻辑)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. 鲁棒性基础数据读取
        mp_id = str(data.get('material_id', 'Unknown'))
        formula = str(data.get('formula_pretty', 'Unknown'))
        
        # 2. 重建 Pymatgen 对象 (这是最关键的一步，它内部处理了大部分类型问题)
        try:
            bs = BandStructureSymmLine.from_dict(data)
        except Exception as e:
            return f"Error: Invalid BandStructure Data ({e})"
        
        # 3. 金属判据
        if bs.is_metal():
            return f"""
【定量计算结果】
1. Material: {mp_id} ({formula})
2. Electronic Type: Metal
3. Band Gap: 0.0000 eV
4. Dynamics: N/A (Metallic)
"""
        
        # 4. 半导体特征提取 (科学计算层)
        bg_data = bs.get_band_gap()
        bg_val = bg_data['energy']
        trans_str = bg_data['transition'] or "Unknown"
        is_direct = bg_data['direct']
        
        vbm_info = bs.get_vbm()
        cbm_info = bs.get_cbm()
        
        # 辅助: 找 Index 并计算 Mass
        def analyze_band_edge(info):
            spin = Spin.up if Spin.up in info['band_index'] else Spin.down
            if spin not in info['band_index']: return "Unknown"
            band_idx = info['band_index'][spin][0]
            
            # 找 k-point index
            target = info['kpoint'].frac_coords
            k_idx = -1
            for i, k in enumerate(bs.kpoints):
                if np.linalg.norm(k.frac_coords - target) < 1e-3:
                    k_idx = i
                    break
            
            mass = calc_effective_mass(bs, band_idx, k_idx, spin) if k_idx != -1 else "N/A"
            return mass

        vbm_mass = analyze_band_edge(vbm_info)
        cbm_mass = analyze_band_edge(cbm_info)
        
        print(f" -> [CALC] {formula}: Gap={bg_val:.2f}eV, me*={cbm_mass}, mh*={vbm_mass}")

        return f"""
【定量计算结果 (Computed Physical Properties)】
1. **基本信息**:
   - ID: {mp_id} | Formula: {formula}
   - Type: Semiconductor/Insulator

2. **带隙特征**:
   - Gap: {bg_val:.4f} eV
   - Type: {"Direct" if is_direct else "Indirect"}
   - Path: {trans_str}

3. **动力学 (Effective Mass)**:
   - Holes (VBM): {vbm_mass}
   - Electrons (CBM): {cbm_mass}
   *(注: m* < 0.5 m_e 意味着高迁移率)*
"""

    except Exception as e:
        print(f" -> [ERROR] {e}")
        return f"Error: {e}"

# ============= API 交互与主流程  =============

def call_qwen_api(messages):
    try:
        response = Generation.call(
            model='qwen-max',
            api_key=DASHSCOPE_API_KEY,
            messages=messages,
            result_format='message',
            stream=False 
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0]['message']['content']
        else:
            print(f"[API ERROR] {response.message}")
            return None
    except Exception as e:
        print(f"[API EXCEPTION] {e}")
        return None

def process_single_file(json_filename):
    mp_id = json_filename.split('_')[0].split('.')[0] 
    json_path = os.path.join(INPUT_DIR, json_filename)
    md_path = os.path.join(OUTPUT_DIR, f"{mp_id}.md")

    print(f"\n[RUN] 正在处理: {mp_id} ...")

    # 1. 调用融合后的计算函数
    llm_context = process_physics_data(json_path)
    
    # 2. 检查是否有错误信息
    if "Error" in llm_context:
        print(f" -> [SKIP] 数据错误: {llm_context}")
        return False

    # 3. 初始化对话
    history = [{'role': 'system', 'content': '你是一个严谨的计算材料学专家，只依据提供的计算事实说话。'}]

    # 4. Round 1: 分析
    print(" -> Prompt 1 (分析中)...")
    history.append({'role': 'user', 'content': PROMPT_1 + f"\n\n{llm_context}"})
    
    resp_1 = call_qwen_api(history)
    if not resp_1: 
        print("  -> [FAIL] Round 1 API 请求失败。")
        return False
    
    history.append({'role': 'assistant', 'content': resp_1})

    # 5. Prompt 2: 生成
    print(" -> Prompt 2 (生成中)...")
    history.append({'role': 'user', 'content': PROMPT_2})
    
    resp_2 = call_qwen_api(history)
    if not resp_2: 
        print("  -> [FAIL] Round 2 API 请求失败。")
        return False

    # 6. 保存
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(resp_2)
        print(f" -> [OK] Saved: {md_path}")
        return True
    except Exception as e:
        print(f" -> [ERROR] Save Failed: {e}")
        return False

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] 输入目录不存在: {INPUT_DIR}")
        return
        
    all_jsons = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    if not all_jsons:
        print(f"[WARN] 输入目录为空。")
        return

    processed_files = os.listdir(OUTPUT_DIR)
    processed_ids = {f.split('.')[0] for f in processed_files if f.endswith('.md')}
    
    pending_files = []
    skipped_count_init = 0 
    
    for f in all_jsons:
        mp_id = f.split('_')[0].split('.')[0]
        if mp_id not in processed_ids:
            pending_files.append(f)
        else:
            skipped_count_init += 1
    
    total_pending = len(pending_files)
    print(f"[INFO] 扫描统计: 总文件 {len(all_jsons)} | 已存在 {skipped_count_init} | 待处理 {total_pending}")
    
    if total_pending == 0:
        print("[DONE] 所有文件都已处理完毕！")
        return

    # 批次控制
    target_files = pending_files[:MAX_BATCH_SIZE] if MAX_BATCH_SIZE else pending_files
    print(f"[START] 本次计划处理: {len(target_files)} 个文件...\n")
    
    count_success = 0
    count_failed = 0
    
    for i, json_file in enumerate(target_files):
        print(f"--- 任务 [{i+1}/{len(target_files)}] ---")
        is_success = process_single_file(json_file)
        
        if is_success:
            count_success += 1
        else:
            count_failed += 1
        time.sleep(1)

    print("\n" + "="*40)
    print("       批处理结束报告")
    print("="*40)
    print(f" [OK]    成功生成: {count_success}")
    print(f" [FAIL]  处理失败: {count_failed}")
    print(f" [SKIP]  此前跳过: {skipped_count_init} (已存在)")
    print("="*40)

if __name__ == '__main__':
    main()