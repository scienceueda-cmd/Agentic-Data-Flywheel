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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# 8GB VRAM (RTX 2070 Super) 向け最適化設定
# ==========================================
model_name = base_model_name

print("モデルとトークナイザーをロード中...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4bit量子化の設定（8GB VRAM必須）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

print("LoRAアダプターを構成中...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("データセットを読み込み中...")
dataset_path = "lora_dataset.jsonl"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"データセットが見つかりません: {dataset_path}")

dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_chat_template(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_chat_template, batched=True)

print("学習を開始します... (数時間かかる場合があります)")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,      # 8GB VRAMの限界のため1に設定
        gradient_accumulation_steps=8,      # バッチサイズを補うために勾配を蓄積
        warmup_steps=5,
        max_steps=5,                        # テスト用に極端に短く設定
        learning_rate=2e-4,
        fp16=True,                          # RTX 2070 SUPERはfp16を使用
        logging_steps=1,
        optim="adamw_8bit",           # 8bit最適化関数でメモリ節約
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_steps=100,
    ),
)

trainer_stats = trainer.train()

save_path = "lora_model"
print(f"学習完了！モデルを {save_path} に保存します...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 学習完了後に不要になった一時ログを自動クリアする
print("学習済みの一時ログファイル（cli_agent_logs.txt および ready_for_review 内のバッチ）をクリアしています...")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cli_log_path = os.path.join(base_dir, "data", "cli_agent_logs.txt")
archive_dir = os.path.join(base_dir, "ready_for_review")

if os.path.exists(cli_log_path):
    try:
        with open(cli_log_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception as e:
        print(f"CLIログのクリアに失敗しました: {e}")

if os.path.exists(archive_dir):
    for filename in os.listdir(archive_dir):
        file_path = os.path.join(archive_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"バッチファイルの削除に失敗しました {file_path}: {e}")

print("全プロセスが完了しました！新しいエージェントAIの誕生です！")
