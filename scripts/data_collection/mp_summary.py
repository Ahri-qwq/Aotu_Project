import os
from mp_api.client import MPRester
from emmet.core.summary import HasProps

print(f"DEBUG: API Key is '{os.getenv('MP_API_KEY')}'") 

mp_api_key = os.getenv('MP_API_KEY')
# 或者直接在代码中设置（不推荐提交到版本控制）
# mp_api_key = "your_api_key_here"

with MPRester(mp_api_key) as mpr:
    print("正在查询具有能带数据的材料...")
    docs = mpr.materials.summary.search(
        has_props=[HasProps.bandstructure], 
        fields=["material_id"]
    )
    mpids = [doc.material_id for doc in docs]

print(f"共找到 {len(mpids)} 个材料具有能带数据")

# 保存材料ID列表以便后续使用
with open("mpids_with_bandstructure.txt", "w") as f:
    for mpid in mpids:
        f.write(f"{mpid}\n")

print("材料ID已保存至 mpids_with_bandstructure.txt")


