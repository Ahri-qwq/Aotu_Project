import os
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import Spin
from tqdm import tqdm

mp_api_key = os.getenv('MP_API_KEY')
# mp_api_key = "your_api_key_here"

# 全局锁，用于线程安全的文件写入
write_lock = Lock()

def download_single_bandstructure(mp_id: str, output_dir: str = "./bandstructures") -> dict:
    """
    下载单个材料的能带数据并保存为JSON文件
    
    Args:
        mp_id: Materials Project ID
        output_dir: 输出目录
    
    Returns:
        包含状态信息的字典
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
        with write_lock:
            with open("download_errors.log", "a") as log:
                log.write(f"{mp_id}: {str(e)}\n")
        return {"mp_id": mp_id, "status": "error", "reason": str(e)}


def batch_download_bandstructures(material_ids: List[str], max_workers: int = 10):
    """
    并行批量下载能带数据
    
    Args:
        material_ids: Materials Project ID 列表
        max_workers: 最大并行线程数（建议 5-20）
    """
    results = {"success": 0, "failed": 0, "skipped": 0}
    
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
        with tqdm(total=len(material_ids), desc="下载进度") as pbar:
            for future in as_completed(future_to_mpid):
                result = future.result()
                results[result["status"]] += 1
                pbar.update(1)
                
                # 实时更新进度信息
                pbar.set_postfix({
                    "成功": results["success"],
                    "失败": results["failed"],
                    "跳过": results["skipped"]
                })
    
    # 输出最终统计
    print("\n下载完成！统计信息：")
    print(f"  成功: {results['success']}")
    print(f"  失败: {results['failed']}")
    print(f"  跳过: {results['skipped']}")
    print(f"详细错误日志请查看 download_errors.log")


# ============ 执行下载 ============
if __name__ == "__main__":
    # 方式1：从保存的文件读取材料ID
    with open("mpids_with_bandstructure.txt", "r") as f:
        mpids = [line.strip() for line in f.readlines()]
    
    # 方式2：或者直接使用API查询（如果未保存文件）
    # with MPRester(mp_api_key) as mpr:
    #     docs = mpr.materials.summary.search(
    #         has_props=[HasProps.bandstructure], 
    #         fields=["material_id"]
    #     )
    #     mpids = [doc.material_id for doc in docs]
    
    # 下载前100个作为测试（移除此限制可下载全部）
    batch_download_bandstructures(mpids[:100], max_workers=10)
    
    # 下载全部数据（取消下面的注释）
    # batch_download_bandstructure(mpids, max_workers=15)
