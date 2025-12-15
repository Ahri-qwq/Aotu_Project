import os
import sys
import json
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import Spin
from tqdm import tqdm
#下面两行必须在from paths import BAND_DIR, MPID_FILE前
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from paths import BAND_DIR, MPID_FILE

from dotenv import load_dotenv  # 导入加载器
load_dotenv(override=True) 

# 日志配置
logging.getLogger("mp_api").setLevel(logging.ERROR)

# 获取 API KEY
mp_api_key = os.getenv('MP_API_KEY')
if not mp_api_key:
    print("错误：未读取到 MP_API_KEY，请检查环境变量。")
    exit(1)

write_lock = Lock()

def download_single_material_data(mp_id: str, output_dir: str = BAND_DIR) -> dict:
    """
    【优化版】同时下载能带数据 + 材料元数据 (Summary)
    逻辑：必须同时拥有元数据和能带数据才保存，否则过滤。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{mp_id}_bandstructure.json")
    
    # 断点续传检查
    if os.path.exists(output_file):
        return {"mp_id": mp_id, "status": "skipped", "reason": "exists"}

    try:
        with MPRester(mp_api_key) as mpr:
            # -------------------------------------------------------
            # 步骤 1: 获取元数据 (Summary Doc)
            # -------------------------------------------------------
            # 注意：先查 Summary 是为了拿化学式。如果这里都查不到，说明 ID 可能有问题。
            summary_docs = mpr.materials.summary.search(
                material_ids=[mp_id], 
                fields=["material_id", "formula_pretty", "symmetry", "structure"]
            )
            
            if not summary_docs:
                return {"mp_id": mp_id, "status": "failed", "reason": "material not found in summary"}
            
            meta = summary_docs[0] # 拿到元数据对象

            # -------------------------------------------------------
            # 步骤 2: 获取能带数据 (BandStructure) - 核心过滤关卡
            # -------------------------------------------------------
            bs_sc = mpr.get_bandstructure_by_material_id(mp_id)
            
            # 【关键修改】如果 API 返回 None，说明该材料在 MP 数据库中没有计算好的能带结构
            # 此时我们将其标记为 "filtered"（已过滤），而不是 "failed"（下载失败）
            if bs_sc is None:
                return {"mp_id": mp_id, "status": "filtered", "reason": "no bandstructure data available"}

            # -------------------------------------------------------
            # 步骤 3: 只有两者都具备时，才融合数据并保存
            # -------------------------------------------------------
            bs_dump = {
                # --- 元数据字段 ---
                'material_id': mp_id,
                'formula_pretty': meta.formula_pretty, # 化学式
                'symmetry': {
                    'symbol': meta.symmetry.symbol,    # 空间群符号
                    'number': meta.symmetry.number     # 空间群编号
                },
                # --- 能带数据 ---
                'efermi': bs_sc.efermi,
                'nb_bands': bs_sc.nb_bands,
                'branches': bs_sc.branches,
                'distance': bs_sc.distance,
                'is_metal': bs_sc.is_metal(),
                'band_gap': bs_sc.get_band_gap(), # 预计算带隙
                'spin_up': bs_sc.bands[Spin.up].tolist()
            }
            
            if bs_sc.is_spin_polarized:
                bs_dump['spin_down'] = bs_sc.bands[Spin.down].tolist()
            
            # 处理 Labels
            labels_dict = {}
            for key, value in bs_sc.labels_dict.items():
                labels_dict[key] = {
                    'frac_coords': value.frac_coords.tolist(),
                    'label': value.label
                }
            bs_dump['labels_dict'] = labels_dict

            # 写入文件
            with write_lock:
                with open(output_file, 'w') as f:
                    json.dump(bs_dump, f, indent=4)
            
            return {"mp_id": mp_id, "status": "success"}

    except Exception as e:
        err_msg = str(e)
        # 某些特定的 API 报错也可以归类为“无数据”而非“脚本错误”
        if "No setyawan_curtarolo" in err_msg:
             return {"mp_id": mp_id, "status": "filtered", "reason": "no k-path data"}
             
        with write_lock:
            with open("download_errors.log", "a") as log:
                log.write(f"{mp_id}: {err_msg}\n")
        return {"mp_id": mp_id, "status": "error", "reason": err_msg}

def batch_download_bandstructures(material_ids: List[str], max_workers: int = 10):
    """批量下载入口"""
    # 初始化统计字典
    results = {
        "success": 0,   # 成功下载并保存
        "skipped": 0,   # 本地已存在
        "filtered": 0,  # 成功连接但无能带数据（正常过滤）
        "failed": 0,    # 找不到 ID 或逻辑错误
        "error": 0      # 网络或 API 异常
    }
    
    print(f"开始下载 {len(material_ids)} 个材料数据 (含化学式元数据)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_mpid = {
            executor.submit(download_single_material_data, mpid): mpid 
            for mpid in material_ids
        }
        
        with tqdm(total=len(material_ids), desc="下载进度", unit="file") as pbar:
            for future in as_completed(future_to_mpid):
                res = future.result()
                status = res["status"]
                
                # 更新统计
                if status in results:
                    results[status] += 1
                else:
                    results["error"] += 1
                
                pbar.update(1)
                # 进度条后缀只显示关键信息
                pbar.set_postfix({
                    "OK": results["success"], 
                    "NoBand": results["filtered"] # 显示被过滤的数量
                })
    
    # 最终报告
    print("\n" + "="*50)
    print(" 下载任务完成报告")
    print("="*50)
    print(f" 成功 : {results['success']}")
    print(f" 跳过 : {results['skipped']}")
    print(f" 过滤 : {results['filtered']}  <-- 材料无能带数据，未保存")
    print(f" 失败 : {results['failed'] + results['error']}")
    print("="*50)

# ============ 执行部分 ============
if __name__ == "__main__":
    if os.path.exists(MPID_FILE):
        with open(MPID_FILE, "r") as f:
            mpids = [line.strip() for line in f.readlines() if line.strip()]
        # 这里的 batch_download_bandstructures 内部调用了 download_single_material_data
        # 而 download_single_material_data 默认使用了 BAND_DIR，所以不需要额外传参

        # 默认使用所有 ID，也可以切片测试 mpids[:100]
        batch_download_bandstructures(mpids[:111], max_workers=10)
    else:
        print(f"请准备 {MPID_FILE} 文件")