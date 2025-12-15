import json
import os

# 检查下载的文件数量
band_files = [f for f in os.listdir("./bandstructures") if f.endswith(".json")]
print(f"已下载 {len(band_files)} 个能带数据文件")

# 查看示例数据
if band_files:
    with open(f"./bandstructures/{band_files[0]}", "r") as f:
        sample = json.load(f)
    print(f"\n示例数据结构（{band_files[0]}）：")
    print(f"  费米能级: {sample['efermi']} eV")
    print(f"  能带数: {sample['nb_bands']}")
    print(f"  高对称点: {list(sample['labels_dict'].keys())}")
