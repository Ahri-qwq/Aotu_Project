import os
import sys
from dotenv import load_dotenv
from dashscope import Generation
from http import HTTPStatus

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ 错误：未读取到 DASHSCOPE_API_KEY")
    sys.exit(1)

def call_qwen_stream():
    print("--- 终端对话助手 (输入 quit 退出 / clear 清除记忆) ---")
    
    history = [
        {'role': 'system', 'content': '你是一个有用的智能助手。'}
    ]

    while True:
        user_input = input("\n请输入您的问题: ")
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
            
        if user_input.lower() == 'clear':
            history = [{'role': 'system', 'content': '你是一个有用的智能助手。'}]
            print("🧹 记忆已清除")
            continue
            
        if not user_input.strip():
            continue

        # 把用户问题加入历史
        history.append({'role': 'user', 'content': user_input})
        
        print("AI: ", end="", flush=True) # 先打印个开头

        try:
            # 【关键点1】开启 stream=True
            # 【关键点2】incremental_output=True 
            #  这个参数让API只返回“最新生成的那几个字”，而不是“目前为止的所有字”
            #  这样我们打印时就不会重复了。
            responses = Generation.call(
                model='qwen-max',
                api_key=api_key,
                messages=history,
                result_format='message',
                stream=True, 
                incremental_output=True 
            )

            full_content = "" # 用来收集完整的回答，存入历史

            # 【关键点3】循环接收碎片
            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    # 拿到这一小块文字
                    chunk = response.output.choices[0]['message']['content']
                    
                    # 实时打印（不换行）
                    print(chunk, end="", flush=True)
                    
                    # 拼接到完整回答里
                    full_content += chunk
                else:
                    print(f"\n❌ 出错: {response.message}")

            # 打印完最后换个行
            print() 

            # 【关键点4】把完整的 AI 回答加入历史，保持记忆
            history.append({'role': 'assistant', 'content': full_content})

        except Exception as e:
            print(f"\n💥 发生异常: {e}")
            history.pop() # 出错回滚

if __name__ == '__main__':
    call_qwen_stream()
