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
THRESHOLD_BYTES = 50000 
CHECK_INTERVAL_SECONDS = 10

def send_toast_notification(batch_file_path):
    if notification:
        notification.notify(
            title="Antigravity 連携システム",
            message=f"Open-WebUIのログが上限に達しました！\n{os.path.basename(batch_file_path)}\nAntigravityに処理を依頼してください。",
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Open-WebUI データベースの監視を開始します...")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    last_processed_size = 0

    while True:
        db_path = find_db()
        if db_path:
            # DBからテキストを抽出
            chat_text = extract_chats_from_db(db_path)
            current_size = len(chat_text.encode('utf-8'))
            
            # 前回処理したサイズから50KB以上増えていればバッチ化
            if current_size - last_processed_size >= THRESHOLD_BYTES:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_filename = f"review_batch_{timestamp}.txt"
                batch_path = os.path.join(ARCHIVE_DIR, batch_filename)
                
                # バッチファイルに書き出し
                with open(batch_path, 'w', encoding='utf-8') as f:
                    f.write(chat_text)
                    
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] バッチを作成しました: {batch_filename}")
                last_processed_size = current_size
                
                send_toast_notification(batch_path)
        else:
            print("webui.db が見つかりません。Open-WebUIを一度起動してください。")
            
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    monitor_db()
