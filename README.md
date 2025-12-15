# 逆向因果链：材料计算案例自动化生成
本项目旨在构建一套自动化的工作流，驱动 LLM 根据 Materials Project 数据，生成学术级、自验证的科研计算案例。

##  项目结构
- `scripts/data_collection`: 负责从 Materials Project 下载能带数据与元数据。
- `scripts/generation`: 负责调用 Qwen 模型进行逆向因果链推演与案例生成。
- `data`: 存放输入数据与输出结果（未上传至 Repo）。

##  快速开始

1. **安装依赖**
pip install -r requirements.txt

2. **配置环境变量**
在根目录创建 `.env` 文件，填入：

MP_API_KEY=

DASHSCOPE_API_KEY=你的阿里云百炼密钥

3. **运行下载**
python scripts/data_collection/mp_download_fin.py

4. **生成案例**
python scripts/generation/auto_generator_fin.py