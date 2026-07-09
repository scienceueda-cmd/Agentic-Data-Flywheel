import os
import time
import sqlite3
import json
from datetime import datetime
try:
    from plyer import notification
except ImportError:
    notification = None

# Open-WebUIのデータベースファイルの予想パス
DB_PATHS = [
    os.path.expanduser("~/.open-webui/backend/data/webui.db"),
    os.path.expanduser("~/.open-webui/webui.db"),
    "webui.db",
    r"C:\Users\malte\AppData\Local\Programs\Python\Python312\Lib\site-packages\open_webui\data\webui.db"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../ready_for_review"))
CLI_LOG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/cli_agent_logs.txt"))
THRESHOLD_BYTES = 20000 
CHECK_INTERVAL_SECONDS = 10

def send_toast_notification(batch_file_path, source="Open-WebUI"):
    if notification:
        notification.notify(
            title="Antigravity 連携システム",
            message=f"{source}のログが上限に達しました！\n{os.path.basename(batch_file_path)}\nAntigravityに処理を依頼してください。",
            app_name="Agentic LoRA Finetuner",
            timeout=10
        )

def find_db():
    for path in DB_PATHS:
        if os.path.exists(path):
            return path
    return None

def extract_chats_from_db(db_path):
    """SQLiteデータベースからチャット履歴を抽出して1つの文字列にする"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Open-WebUIの 'chat' テーブルから履歴JSONを取得
        cursor.execute("SELECT chat FROM chat")
        rows = cursor.fetchall()
        
        full_text = ""
        for row in rows:
            try:
                chat_data = json.loads(row[0])
                messages = chat_data.get("messages", [])
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"]:
                        full_text += f"{role.capitalize()}: {content}\n\n"
            except:
                pass # JSONパースエラーなどはスキップ
                
        conn.close()
        return full_text
    except Exception as e:
        print(f"DB読み込みエラー: {e}")
        return ""

def monitor_db():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 監視エージェントを起動しました。")
    print(" - Open-WebUI データベースの監視: ON")
    print(" - ターミナル(CLI) エージェントログの監視: ON")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    last_processed_size_webui = 0

    while True:
        # 1. WebUI Database 監視
        db_path = find_db()
        if db_path:
            chat_text = extract_chats_from_db(db_path)
            current_size_webui = len(chat_text.encode('utf-8'))
            
            if current_size_webui - last_processed_size_webui >= THRESHOLD_BYTES:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_filename = f"review_batch_webui_{timestamp}.txt"
                batch_path = os.path.join(ARCHIVE_DIR, batch_filename)
                
                with open(batch_path, 'w', encoding='utf-8') as f:
                    f.write(chat_text)
                    
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WebUIバッチを作成しました: {batch_filename}")
                last_processed_size_webui = current_size_webui
                send_toast_notification(batch_path, source="Open-WebUI")
        
        # 2. CLI Agent Logs 監視
        if os.path.exists(CLI_LOG_PATH):
            try:
                with open(CLI_LOG_PATH, 'r', encoding='utf-8') as f:
                    cli_text = f.read()
                current_size_cli = len(cli_text.encode('utf-8'))
                
                if current_size_cli >= THRESHOLD_BYTES:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_filename = f"review_batch_cli_{timestamp}.txt"
                    batch_path = os.path.join(ARCHIVE_DIR, batch_filename)
                    
                    with open(batch_path, 'w', encoding='utf-8') as f:
                        f.write(cli_text)
                        
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CLIバッチを作成しました: {batch_filename}")
                    
                    # ログの重複を防ぐため、元のログファイルを空にする
                    with open(CLI_LOG_PATH, 'w', encoding='utf-8') as f:
                        f.write("")
                        
                    send_toast_notification(batch_path, source="CLIエージェント")
            except Exception as e:
                print(f"CLIログ読み込みエラー: {e}")
            
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    monitor_db()
