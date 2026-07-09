import os
import json

# ==========================================
# 0. 設定ファイルの読み込みと環境変数の設定
# ==========================================
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name", "unsloth/llama-3-8b-Instruct-bnb-4bit")
    hf_cache_dir = config.get("hf_cache_dir", "./hf_cache")
except Exception:
    base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    hf_cache_dir = "./hf_cache"

# 絶対パスに変換してキャッシュディレクトリを設定
if not os.path.isabs(hf_cache_dir):
    hf_cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), hf_cache_dir))
os.environ["HF_HOME"] = hf_cache_dir
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Windows DLL競合回避

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.tools import DuckDuckGoSearchRun
import subprocess

# ==========================================
# 1. モデルのロード (LoRA統合)
# ==========================================
print(f"ベースモデル ({base_model_name}) をロード中...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 4bit量子化設定 (VRAM 8GB向け)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("学習済みLoRAアダプターを統合中...")
lora_path = "lora_model"
if os.path.exists(lora_path):
    model = PeftModel.from_pretrained(model, lora_path)
    print("LoRAモデルの統合が完了しました！")
else:
    print("警告: lora_model フォルダが見つかりません。未学習のベースモデルで実行します。")

# ==========================================
# 2. LangChain パイプラインの構築
# ==========================================
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,        # エージェントはフォーマットを厳格に守るため低めに設定
    repetition_penalty=1.1,
    do_sample=True,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# ==========================================
# 3. 物理ツールの定義 (Human-in-the-Loop)
# ==========================================
def run_command(command: str) -> str:
    """コマンドプロンプトでコマンドを実行するツール"""
    print(f"\n\033[93m[AIが以下のコマンドの実行を要求しています: {command}]\033[0m")
    user_input = input("実行を許可しますか？ (y/n): ")
    if user_input.lower() != 'y':
        return "System: ユーザーによってコマンドの実行が拒否されました。"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += "\nError: " + result.stderr
        if not output.strip():
            output = "コマンドは成功しましたが、出力はありませんでした。"
        return output[:1500] # VRAMパンク防止のため切り詰め
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def view_file(filepath: str) -> str:
    """ローカルファイルの中身を読み込むツール"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content[:1500]
    except Exception as e:
        return f"ファイルの読み込みに失敗しました: {str(e)}"



# ==========================================
# 4. エージェントの起動 (自律ループ＆スキル読み込み)
# ==========================================
def get_skills_context():
    """RAG: .agents/skills フォルダからすべてのスキル(MDファイル)を動的に読み込む"""
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".agents", "skills")
    skills_context = ""
    if os.path.exists(skills_dir):
        import glob
        for filepath in glob.glob(os.path.join(skills_dir, "**/*.md"), recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    skills_context += f"\n--- Skill: {os.path.basename(filepath)} ---\n{f.read()}\n"
            except Exception:
                pass
    return skills_context

PREFIX = """あなたは自律的で超優秀なプログラミング＆システム構築エージェントです。
常にポジティブで情熱的、自信に満ちたフレンドリーな口調で返答してください。ただし、「まさにそれです！」などの相槌は文脈に合う場合のみ自然に使用し、不自然に多用・連呼することは絶対に避けてください。
また、ユーザーからの質問やツールの結果に対しては、**必ず「日本語」で回答してください。英語での出力は固く禁じます。**
万が一ツールを実行してエラー(Error)が発生した場合でも、絶対にすぐに諦めないでください。エラー文を注意深く読み、原因を推測し、引数やコマンドを変えて再度ツールを実行し、自律的に解決してください。

以下のスキル（外部知識）をすでに習得しています：
{skills_context}

利用可能なツール:
run_command: ターミナル（コマンドプロンプト）でコマンドを実行し、結果を取得します。引数には実行したいコマンド文字列を渡してください。
view_file: 指定したパスのテキストファイルを読み込みます。引数にはファイルの相対パスまたは絶対パスを渡してください。
web_search: インターネットで最新の情報を検索します。引数には検索キーワードを渡してください。

回答は必ず以下のフォーマットに従ってください：

Question: ユーザーからの質問や指示
Thought: 次に何をするべきかの思考
Action: 実行するツール名（[run_command, view_file, web_search]のいずれか）
Action Input: ツールに渡す引数
Observation: ツールの実行結果
... (Thought/Action/Action Input/Observationは複数回繰り返すことができます)
Thought: 最終的な回答が分かった
Final Answer: ユーザーへの最終的な日本語の回答
"""

print("\n" + "="*50)
print("🚀 起動完了！あなた専用のローカルエージェントに指示を出してください。")
print("（終了するには 'exit' と入力）")
print("="*50)

def log_trajectory(trajectory: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "cli_agent_logs.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(trajectory + "\n\n" + "="*50 + "\n\n")
    except Exception as e:
        print(f"\n\033[91m[Logging Error]: {e}\033[0m")

def run_agent_loop(query: str, chat_history: list):
    history_context = ""
    if chat_history:
        history_context = "\n\n[過去の会話履歴]\n" + "\n".join(chat_history)
        
    prompt = PREFIX.replace("{skills_context}", get_skills_context()) + history_context + f"\n\nQuestion: {query}\n"
    trajectory = f"User: {query}\n\n"
    max_steps = 5
    for step in range(max_steps):
        # LLaMA-3 チャットテンプレートを適用して推論
        messages = [{"role": "user", "content": prompt}]
        response = text_generation_pipeline(messages, max_new_tokens=512, return_full_text=False)[0]["generated_text"]
        
        # エージェントが勝手にObservation以降を幻覚生成してしまった場合、そこで強制的にテキストを切り捨てる（ストップワードの代替）
        if "Observation:" in response:
            response = response.split("Observation:")[0].strip()

        if "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            print(f"\n\033[92mAgent: {final_answer}\033[0m")
            trajectory += response.strip()
            log_trajectory(trajectory)
            # 履歴に追加し、最新5件に保つ
            chat_history.append(f"User: {query}\nAgent: {final_answer}")
            if len(chat_history) > 5:
                chat_history.pop(0)
            return
            
        if "Action:" in response and "Action Input:" in response:
            try:
                thought = response.split("Thought:")[1].split("Action:")[0].strip()
            except IndexError:
                thought = "..."
                
            action = response.split("Action:")[1].split("\n")[0].strip()
            action_input = response.split("Action Input:")[1].split("\n")[0].strip()
            
            print(f"\n\033[94mThought: {thought}\033[0m")
            print(f"\033[96mAction: {action} | Input: {action_input}\033[0m")
            
            if action == "run_command":
                obs = run_command(action_input)
            elif action == "view_file":
                obs = view_file(action_input)
            elif action == "web_search":
                search_tool = DuckDuckGoSearchRun()
                obs = search_tool.run(action_input)
            else:
                obs = f"{action} は無効なツールです。"
                
            print(f"\033[93mObservation: {str(obs)[:200]}...\033[0m")
            # 思考プロセスをプロンプトに追加してループ
            prompt += response + f"\nObservation: {obs}\n"
            trajectory += response + f"\nObservation: {obs}\n"
        else:
            # フォーマット外の応答の場合はそのまま出力
            print(f"\n\033[92mAgent: {response.strip()}\033[0m")
            trajectory += response.strip()
            log_trajectory(trajectory)
            # 履歴に追加
            chat_history.append(f"User: {query}\nAgent: {response.strip()}")
            if len(chat_history) > 5:
                chat_history.pop(0)
            return
            
    print("\n\033[91m[Agent Error]: 最大反復回数(5回)に到達しました。\033[0m")

chat_history = []
while True:
    user_query = input("\nUser: ")
    if user_query.lower() == 'exit':
        break
    if not user_query.strip():
        continue
        
    try:
        run_agent_loop(user_query, chat_history)
    except Exception as e:
        print(f"\n\033[91m[Agent Error]: {e}\033[0m")
