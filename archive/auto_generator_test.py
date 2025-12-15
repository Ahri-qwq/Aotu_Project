import os
import sys
import json
import time
from http import HTTPStatus
from dotenv import load_dotenv
from dashscope import Generation


# ================= 配置区域 =================
INPUT_DIR = "./bandstructures"  
OUTPUT_DIR = "./case_output"    
MAX_BATCH_SIZE = 3            #最大执行数，可以修改为更大数值，以批量执行。
# ===========================================

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    print("[ERROR] 未读取到 DASHSCOPE_API_KEY，请检查环境变量。")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# Prompt 模板 (保持不变)
# -----------------------------------------------------------------

PROMPT_1 = r"""
你是一位基于“逆向因果链”理论的计算材料学专家。请深度解析用户上传的材料数据文件。

### 核心任务：数据锚定 (Data Anchoring)
深入数据细节，锁定 1-3 个具有极高科研价值的**定量锚点 (Quantitative Anchors)**。
*   **锚点定义**：数据文件中确凿存在的、且能反映材料核心物理特性的具体数值或特征（例如：特定的带隙值、态密度峰值、有效质量数值、各向异性比率等）。

### 执行逻辑：
1.  **提取锚点**：找出这些“死数据”。
2.  **逆向重构**：对于每个锚点，反推一个能够计算出该数据的**科研计算场景**。
    *   *逻辑*：既然答案是 X，那么问题（计算任务）应该怎么设计，才能引导用户算出的结果恰好是 X？

**请输出提取的“锚点”及其对应的“逆向科研场景”构思。（暂时不要生成最终案例，仅提供锚定逻辑）**
"""

PROMPT_2 = r"""
很好，请利用你提取的锚点，直接输出一份**“自验证型”深度科研计算案例**。

### 核心约束：
1.  **直接输出**：不要废话，**直接从一级标题（材料名称）开始**。
2.  **闭环验证**：在每个维度的“验证依据”中，必须引用你刚才提取的**定量锚点**（即引用上传文件中的真值）。
3.  **学术风格**：保持高度严谨，使用不等式或具体数值范围。

### 请参照以下 Markdown 格式输出（章节数量根据锚点数量自适应）：

# [材料名称]：[核心物理特性摘要]

**材料背景与科研价值**：
*(材料名称、别名、简练综述材料结构与地位，以及进行第一性原理计算的必要性。)*

## 1. [维度一标题]
*   **科学问题**：
    *(逆向推导出的科研疑问。例如：为何实验观测到 [现象]？是否源于 [微观机制]？)*
*   **关键性质（计算目标）**：
    *   **目标指标**：*(例如：计算沿 [路径] 的有效质量)*
    *   **验证依据 (Ground Truth)**：
        *   **理论判据**：*(物理上的定性判断，如 $m^*_{\parallel} \ll m^*_{\perp}$)*
        *   **数据锚点**：*(引用上传数据中的真值。例如：根据上传数据，你的计算结果应在 [具体数值] 左右。)*

## 2. [维度二标题]
*(...)*

... (根据实际锚点数量继续列出)

"""

# -----------------------------------------------------------------
# 核心功能函数
# -----------------------------------------------------------------

def call_qwen_api(messages):
    """封装 API 调用"""
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

    llm_context = ""

    # ================= 核心修改区：手动物理提取逻辑 =================
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 提取化学式
        # MP 数据通常包含 formula_pretty (如 "SrTiO3") 或 formula (如 "Sr1 Ti1 O3")
        # 如果都没有，最终输出会只显示 ID 
        formula = data.get('formula_pretty') or data.get('formula') or "Unknown Formula"

        # 2. 提取其他物理参数
        efermi = data.get('efermi', 0.0)
        labels = list(data.get('labels_dict', {}).keys())
        nb_bands = data.get('nb_bands', 0)
        
        # 3. 手动计算 VBM, CBM 和 Band Gap
        all_energies = []
        if 'spin_up' in data:
            for band in data['spin_up']:
                all_energies.extend(band)
        if 'spin_down' in data:
            for band in data['spin_down']:
                all_energies.extend(band)

        if all_energies:
            TOL = 0.001
            valence_states = [e for e in all_energies if e <= efermi + TOL]
            vbm = max(valence_states) if valence_states else efermi
            conduction_states = [e for e in all_energies if e > efermi + TOL]
            cbm = min(conduction_states) if conduction_states else efermi
            band_gap = cbm - vbm
            is_metal = band_gap < 0.01
            
            # 构建精准的物理上下文
            llm_context = f"""
            【材料物理特征摘要 (Calculated from Raw Data)】
            - Material ID: {mp_id}
            - Chemical Formula: {formula}  <-- 告诉 AI 它的名字！
            - Fermi Energy: {efermi:.4f} eV
            - Is Metal: {is_metal}
            - Band Gap: {band_gap:.4f} eV
            - VBM (Valence Band Max): {vbm:.4f} eV
            - CBM (Conduction Band Min): {cbm:.4f} eV
            - Number of Bands: {nb_bands}
            - High Symmetry Points: {labels}
            - Data Source: Local Computed
            """
            print(f"  -> [OK] 物理特征提取成功。")
        else:
             print("  -> [WARN] 未找到能带数据，无法计算特征。")

    except Exception as e:
        print(f"  -> [WARN] 物理提取失败 ({e})，将回退到智能截断模式。")
        llm_context = "" 

    # ================= 回退逻辑：智能结构化截断 =================
    if not llm_context:
        try:
            print("  -> [INFO] 启用智能截断模式...")
            # 重新读取 (或者复用上面的 data 变量)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取化学式
            formula = data.get('formula_pretty') or data.get('formula') or "Unknown"

            core_data = {
                "material_id": data.get("material_id"),
                "formula": formula,  # 【关键】加入化学式
                "efermi": data.get("efermi"),
                "nb_bands": data.get("nb_bands"),
                "labels_dict": data.get("labels_dict"),
                "branches": data.get("branches"),
            }
            
            # 制作样本
            sample_bands = {}
            if "spin_up" in data and isinstance(data["spin_up"], list):
                try:
                    raw_sample = data["spin_up"][:5] 
                    sample_bands["spin_up"] = [row[:5] if isinstance(row, list) else row for row in raw_sample]
                except: pass
            core_data["sample_bands"] = sample_bands

            llm_content = json.dumps(core_data, indent=2, ensure_ascii=False)
            
            # 这里的截断逻辑可以保留，防万一
            if len(llm_content) > 20000:
                 llm_content = llm_content[:20000] + "\n... (Truncated)"
            
            llm_context = f"\n【附带数据文件内容 (Smart Parsed JSON)】:\n{llm_content}"
            
        except Exception as e:
            print(f"  -> [ERROR] 智能截断失败: {e}")
            return False

    # 2. 初始化对话
    history = [{'role': 'system', 'content': '你是一个专业的计算材料学助手。'}]

    # 3. Round 1: 分析
    print("  -> 已发送 Prompt 1 (正在分析)...")
    user_input_1 = PROMPT_1 + f"\n\n{llm_context}"
    history.append({'role': 'user', 'content': user_input_1})
    
    resp_1 = call_qwen_api(history)
    if not resp_1:
        print("  -> [FAIL] Round 1 API 请求失败。")
        return False
    
    history.append({'role': 'assistant', 'content': resp_1})
    
    # 4. Round 2: 生成
    print("  -> 已发送 Prompt 2 (正在生成)...")
    history.append({'role': 'user', 'content': PROMPT_2})
    
    resp_2 = call_qwen_api(history)
    if not resp_2:
        print("  -> [FAIL] Round 2 API 请求失败。")
        return False

    # 5. 保存
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(resp_2)
        print(f"  -> [OK] 保存成功: {md_path}")
        return True
    except Exception as e:
        print(f"  -> [ERROR] 保存失败: {e}")
        return False

# -----------------------------------------------------------------
# 主循环
# -----------------------------------------------------------------

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] 输入目录不存在: {INPUT_DIR}")
        return
        
    all_jsons = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    if not all_jsons:
        print(f"[WARN] 输入目录为空。")
        return

    # 扫描增量
    processed_files = os.listdir(OUTPUT_DIR)
    processed_ids = {f.split('.')[0] for f in processed_files if f.endswith('.md')}
    
    pending_files = []
    skipped_count_init = 0 # 初始跳过的数量（已存在的）
    
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
    
    # === 计数器 ===
    count_success = 0
    count_failed = 0
    
    for i, json_file in enumerate(target_files):
        print(f"--- 任务 [{i+1}/{len(target_files)}] ---")
        
        # 执行处理，根据返回值统计
        is_success = process_single_file(json_file)
        
        if is_success:
            count_success += 1
        else:
            count_failed += 1
            
        time.sleep(1)

    # === 最终统计输出 ===
    print("\n" + "="*40)
    print("       批处理结束报告")
    print("="*40)
    print(f" [OK]    成功生成: {count_success}")
    print(f" [FAIL]  处理失败: {count_failed}")
    print(f" [SKIP]  此前跳过: {skipped_count_init} (已存在)")
    print("="*40)

if __name__ == '__main__':
    main()
