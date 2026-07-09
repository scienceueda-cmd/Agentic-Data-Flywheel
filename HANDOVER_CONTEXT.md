# ファインチューニング用・引継ぎコンテキスト

このファイルは、次回のプロジェクト（ファインチューニングの実行や別エージェントによる作業）のために、現在の環境や構成をまとめたものです。

## 📁 ディレクトリ構成と役割
- **`D:\antigravity folder\agentic_finetuner\`** (プロジェクトルート)
  - `src\log_monitor.py`: Open-WebUIのSQLite(`webui.db`)を監視し、会話データを10KBごとにJSONL形式で抽出するスクリプト。
  - `src\train_lora.py`: 抽出されたJSONLデータ(`lora_dataset.jsonl`)を元に、UnslothでLlama 3をファインチューニングするスクリプト。
  - `lora_dataset.jsonl`: 抽出された学習用データの出力先（結合済み）。`train_lora.py`はこれを読み込みます。
  - `data\ready_for_review\`: `log_monitor.py`が一時的に出力するJSONLファイルの保存先。

## ⚙️ インフラ環境の修正履歴（重要事項）
1. **PyTorchのCUDA対応化**: `torch==2.5.1+cu121` に固定して再インストールし、RTX 2070 SUPER（VRAM 8GB）がフル活用できるようになっています。（`torch.cuda.is_available()` == True）
2. **Unslothの依存関係調整**: Windows環境での `torchao` の `torch.int1` エラーや、`xformers` のC++拡張エラーを解消済みです（xformersを削除し、torchaoをダウングレード）。これにより `import unsloth` が正常に動きます。
3. **Llama-3のツールエラー回避**: Open-WebUIのバグ（`does not support tools`）を回避するため、内部的に `llama3-fixed:8b` というツール対応（空処理）のダミーテンプレートを適用したモデルを作成しています。ユーザーはチャット画面でこの `llama3-fixed:8b` を使用しています。
4. **自動監視タスク**: `C:\Users\malte\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\AutoStart_LogMonitor.vbs` を配置し、PC起動時に裏側で `log_monitor.py` が自動で動くようにしています。

## 🚀 次のアクション（ファインチューニング実行時）
- ユーザーが `llama3-fixed:8b` とチャットし、十分にデータ（`lora_dataset.jsonl`）が溜まっていることを確認してください。
- **【🎉 環境構築完了】Windowsネイティブ（Python直接実行）で完全に動作するよう、`bitsandbytes` のロードクラッシュ、OpenMP (`libiomp5md.dll`) のDLL競合、および `trl` 0.15.0 のバグをすべて解決・修正済みです！WSLは不要で、そのままWindowsのコマンドプロンプトで学習が回せます。**
- `train_lora.py` を実行する際は、パラメーター（`max_steps` や `num_train_epochs`）を本番用に調整してください。（現在はテスト用に極小モデル＋`max_steps=5`のテスト設定からLlama-3に戻した状態になっています）。
- メモリ制限: RTX 2070 SUPERはVRAMが8GBのため、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8`、4bit量子化(`load_in_4bit=True`)を維持してください。
