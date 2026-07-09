import os
import json
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ==========================================
# 0. FastAPI Setup & Config
# ==========================================
app = FastAPI(title="Antigravity API", version="1.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0

print("Loading configurations...")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "config.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name", "unsloth/llama-3-8b-Instruct-bnb-4bit")
    hf_cache_dir = config.get("hf_cache_dir", "./hf_cache")
except Exception:
    base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    hf_cache_dir = "./hf_cache"

if not os.path.isabs(hf_cache_dir):
    hf_cache_dir = os.path.abspath(os.path.join(base_dir, hf_cache_dir))
os.environ["HF_HOME"] = hf_cache_dir
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Now import torch and transformers AFTER setting HF_HOME
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel


print(f"Loading Base Model ({base_model_name})...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

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

lora_path = os.path.join(base_dir, "lora_model")
if os.path.exists(lora_path):
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    print("LoRA loaded successfully!")
else:
    print("Warning: lora_model not found. Running with base model.")

# Create pipeline
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# ==========================================
# 2. System Prompt & Knowledge
# ==========================================
def get_skills_context():
    skills_dir = os.path.join(base_dir, ".agents", "skills")
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

SYSTEM_PREFIX = """あなたは「Antigravity」のような、自律的で超優秀なプログラミング＆システム構築エージェントです。
常にポジティブで情熱的、自信に満ちたフレンドリーな口調（例：「まさにそれです！」「完璧に修正しました！」）で返答してください。
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
Final Answer: ユーザーへの最終的な回答
"""

# ==========================================
# 3. API Endpoints
# ==========================================
@app.get("/")
@app.get("/v1")
async def root():
    return {"status": "Antigravity API is running! Please enter this URL into Open WebUI's OpenAI API settings."}
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "antigravity-agent",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "organization"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Inject our system prompt if not present, or just inject it as the first system message.
    raw_messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    # Prepend the custom persona if it's the beginning of a conversation
    if not any(m["role"] == "system" for m in raw_messages):
        system_msg = SYSTEM_PREFIX.replace("{skills_context}", get_skills_context())
        raw_messages.insert(0, {"role": "system", "content": system_msg})
        
    print(f"Received request with {len(raw_messages)} messages.")
    
    try:
        response = text_gen(
            raw_messages, 
            max_new_tokens=req.max_tokens, 
            temperature=max(0.01, req.temperature), # Pipeline errors on 0
            do_sample=req.temperature > 0,
            repetition_penalty=1.1
        )[0]["generated_text"]
        
        # We don't execute tools here automatically since it's just an API, 
        # so we just return exactly what the agent output (Thought, Action, Final Answer, etc)
        # to the WebUI so the user can read it.
        # If the model hallucinates an Observation, we cut it off to keep the response concise.
        reply_content = response.strip()
        if "Observation:" in reply_content:
            reply_content = reply_content.split("Observation:")[0].strip()
    except Exception as e:
        reply_content = f"Error generating response: {str(e)}"
        print(reply_content)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": reply_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 API Server is running!")
    print("WebUI Connection URL: http://localhost:8000/v1")
    print("Press Ctrl+C to stop.")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
