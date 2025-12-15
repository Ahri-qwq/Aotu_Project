import os
import json
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import Spin
from tqdm import tqdm

# ============ 优化配置：屏蔽刷屏日志 ============
# 将 mp-api 的日志级别提高到 ERROR，不再显示 Retrieving...
logging.getLogger("mp_api").setLevel(logging.ERROR)

mp_api_key = os.getenv('MP_API_KEY')
# mp_api_key = "your_api_key_here"

# 全局锁，用于线程安全的文件写入
write_lock = Lock()

def download_single_bandstructure(mp_id: str, output_dir: str = "./bandstructures") -> dict:
    """
    下载单个材料的能带数据并保存为JSON文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否已存在（断点续传）
    output_file = os.path.join(output_dir, f"{mp_id}_bandstructure.json")
    if os.path.exists(output_file):
        return {"mp_id": mp_id, "status": "skipped", "reason": "already exists"}
    
    try:
        with MPRester(mp_api_key) as mpr:
            bs_sc = mpr.get_bandstructure_by_material_id(mp_id)
            
            if bs_sc is None:
                return {"mp_id": mp_id, "status": "failed", "reason": "no bandstructure data"}
            
            # 构建数据结构（与文档中的格式一致）
            bs_dump = {
                'material_id': mp_id,
                'efermi': bs_sc.efermi,
                'nb_bands': bs_sc.nb_bands,
                'branches': bs_sc.branches,
                'distance': bs_sc.distance,
                'spin_up': bs_sc.bands[Spin.up].tolist()
            }
            
            # 处理自旋极化情况
            if bs_sc.is_spin_polarized:
                bs_dump['spin_down'] = bs_sc.bands[Spin.down].tolist()
            
            # 转换 labels_dict（高对称点信息）
            labels_dict = {}
            for key, value in bs_sc.labels_dict.items():
                labels_dict[key] = {
                    'cart_coords': value.cart_coords.tolist(),
                    'frac_coords': value.frac_coords.tolist(),
                    'label': value.label
                }
            bs_dump['labels_dict'] = labels_dict
            
            # 线程安全的文件写入
            with write_lock:
                with open(output_file, 'w') as f:
                    json.dump(bs_dump, f, indent=4)
            
            return {"mp_id": mp_id, "status": "success"}
            
    except Exception as e:
        # 记录错误信息
        # 忽略 "No setyawan_curtarolo" 这种常见警告
        err_msg = str(e)
        if "No setyawan_curtarolo" not in err_msg:
             with write_lock:
                with open("download_errors.log", "a") as log:
                    log.write(f"{mp_id}: {err_msg}\n")
        
        return {"mp_id": mp_id, "status": "error", "reason": err_msg}

def batch_download_bandstructures(material_ids: List[str], max_workers: int = 10):
    """
    并行批量下载能带数据
    """
    # 修复 KeyError：添加 "error" 键
    results = {"success": 0, "failed": 0, "skipped": 0, "error": 0}
    
    print(f"开始下载 {len(material_ids)} 个材料的能带数据...")
    print(f"使用 {max_workers} 个并行线程")
    
    # 使用线程池并行下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_mpid = {
            executor.submit(download_single_bandstructure, mpid): mpid 
            for mpid in material_ids
        }
        
        # 使用 tqdm 显示进度条
        # 优化显示格式：不刷屏，只显示总体进度和统计
        with tqdm(total=len(material_ids), 
                  desc="下载进度", 
                  unit="file",
                  bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for future in as_completed(future_to_mpid):
                result = future.result()
                status = result["status"]
                
                # 累加统计
                results[status] += 1
                pbar.update(1)
                
                # 实时更新进度信息 (只显示统计，不显示具体ID)
                pbar.set_postfix({
                    "OK": results["success"],
                    "Skip": results["skipped"],
                    "Fail": results["failed"] + results["error"]
                })
    
    # 输出最终统计
    print("\n下载完成！统计信息：")
    print(f"  跳过: {results['skipped']}")
    print(f"  成功: {results['success']}")
    print(f"  失败: {results['failed'] + results['error']}")
    print(f"详细错误日志请查看 download_errors.log")

# ============ 执行下载 ============
if __name__ == "__main__":
    try:
        with open("mpids_with_bandstructure.txt", "r") as f:
            mpids = [line.strip() for line in f.readlines()]
        # 下载前100个作为测试（已下载的会跳过）
        batch_download_bandstructures(mpids[:100], max_workers=10)
        
        # 下载全部数据（取消下面的注释）
        # batch_download_bandstructures(mpids, max_workers=10)
        
    except FileNotFoundError:
        print("错误：未找到 mpids_with_bandstructure.txt 文件，请先运行查询脚本。")
