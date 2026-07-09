---
name: langchain_react_spec
description: LangChainのReActエージェントの公式フォーマット仕様、パース規則、およびファインチューニング用データセット構築のガイドライン。
---

# LangChain ReAct Agent Specification

このスキルは、LangChainのReAct（Reasoning and Acting）エージェントの厳格な仕様と、その挙動をLLMに学習させる（ファインチューニングする）ためのデータセット構造を定義するものです。

## 1. ReActフォーマットの基本構造
LangChainの標準的なReActエージェント（`AgentType.ZERO_SHOT_REACT_DESCRIPTION` 等）は、LLMの出力を以下の特定のキーワードでパース（解析）します。

- `Thought:` - LLMが現状を分析し、次に何をするべきかを考えるプロセス。
- `Action:` - 実行するツールの名前。
- `Action Input:` - ツールに渡す引数（検索クエリなど）。
- `Observation:` - ツールの実際の実行結果（**※LLMではなく、LangChainシステム側が生成する**）。
- `Final Answer:` - ユーザーに対する最終的な回答。

## 2. 厳格な出力停止規則（最重要）
LLMは、ツールを使用する場合、**絶対に `Action Input:` を出力した直後にテキストの生成をストップしなければなりません。**
もしLLM自身が続けて `Observation:` を出力してしまうと、それは「ツールの結果を捏造（ハルシネーション）した」ことになり、LangChainのパーサーが致命的なエラーを起こします。

## 3. 学習用データセット（JSONL）への落とし込み方
Llama 3などのChatモデルをファインチューニングしてReActエージェント化する場合、データセット（`messages` 配列）は以下の交互のやり取りとして厳密に表現される必要があります。

### 【正しいトラジェクトリ（軌跡）の構造】
1. **User (システム)**: ユーザーの質問
2. **Assistant (LLM)**: 思考とツールの指定
   ```text
   Thought: [考察]
   Action: [ツール名]
   Action Input: [引数]
   ```
   *(※ここでAssistantのターンは終了)*
3. **User (システム)**: ツール結果の返却
   ```text
   Observation: [システムの実行結果]
   ```
4. **Assistant (LLM)**: 結果を受けた思考と回答
   ```text
   Thought: [結果の分析]
   Final Answer: [最終回答]
   ```

## 4. データ生成・キュレーション時の注意点（LLM-as-a-Judge向け）
ログファイルから学習データを抽出・生成する際は、以下の点に注意して無駄なデータを破棄・リライトしてください。
- **フィラーの排除**: `Thought:` の前に「わかりました！」「お答えします」などの不要な相槌を入れてはいけません。
- **知ったかぶりの修正**: ツールが必要な質問に対し、LLMが想像で答えているログを見つけた場合は、上記「正しいトラジェクトリ」の形式になるように、あなたがツールの使用プロセスを想像して完璧にリライトしてください。
- **フォーマットの崩れ**: `ActionInput:` や `Action-Input:` などのわずかなタイポも許されません。必ず `Action Input:` に統一してください。
