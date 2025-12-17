import json
import os
import sys
# ================= 路径配置 ================= 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from paths import BAND_DIR

# 检查下载的文件数量
band_files = [f for f in os.listdir(BAND_DIR) if f.endswith(".json")]
print(f"已下载 {len(band_files)} 个能带数据文件")

# 查看示例数据
if band_files:
    file_path = os.path.join(BAND_DIR, band_files[0])

    with open(file_path, "r") as f:
        sample = json.load(f)
    print(f"\n示例数据结构（{band_files[0]}）：")
    print(f"  费米能级: {sample['efermi']} eV")
    print(f"  能带数: {sample['nb_bands']}")
    print(f"  高对称点: {list(sample['labels_dict'].keys())}")
