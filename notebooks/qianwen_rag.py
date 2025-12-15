import os
import dashscope
from dashscope import Generation, TextEmbedding
import chromadb
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 1. 准备工作：模拟一个简单的知识库 ---
# 在真实场景中，这里是你读取 PDF/TXT 的代码
documents = [
    "Qwen3是阿里云在2025年发布的最新一代超大规模语言模型。",
    "Qwen3相比前代，在逻辑推理和代码能力上提升了40%。",
    "用户有一只宠物猫",
    "大象其实很害怕老鼠，因为老鼠会钻进它们的鼻子里（这是个谣言）。"
]

# 初始化向量数据库（这里用内存模式，重启就没了，方便测试）
print("正在构建知识库...")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_knowledge_base")

# 把文本变成向量存进去 (这里为了演示简单，我们假设用 dashscope 提供的 embedding)
# 真实 RAG 这一步通常是离线做好的
def get_embedding(text):
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text,
        api_key=api_key
    )
    return resp.output.embeddings[0].embedding

# 存入数据库
for i, doc in enumerate(documents):
    # 这里偷懒了，真实场景建议用 batch 批量处理
    emb = get_embedding(doc)
    collection.add(
        ids=[str(i)],
        embeddings=[emb],
        documents=[doc]
    )
print("✅ 知识库构建完成！(存了4条数据)")


# --- 2. 检索 + 生成 ---
def chat_with_rag():
    print("--- RAG 助手 (问问我关于 Qwen3 的事) ---")
    while True:
        query = input("\n你: ")
        if query in ['quit', 'exit']: break
        
        # [步骤A] 检索：先去库里找答案
        print("🔍 正在检索资料...", end="")
        query_emb = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=2 # 只找最相关的2条
        )
        retrieved_docs = results['documents'][0]
        print(f"找到 {len(retrieved_docs)} 条相关资料")
        
        # [步骤B] 增强：把资料塞进 Prompt
        # 这就是 RAG 的灵魂：让模型“看着答案作弊”
        context_str = "\n".join([f"- {doc}" for doc in retrieved_docs])
        
        system_prompt = f"""
        你是一个基于知识库的助手。
        请务必只根据下面的【参考资料】来回答用户的问题。如果资料里没提到，就说不知道。
        
        【参考资料】
        {context_str}
        """
        
        # [步骤C] 生成：带着资料去问 LLM
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]
        
        print("AI正在思考...")
        resp = Generation.call(
            model='qwen-max',
            api_key=api_key,
            messages=messages,
            result_format='message'
        )
        
        print(f"🤖 AI: {resp.output.choices[0]['message']['content']}")

if __name__ == "__main__":
    chat_with_rag()
