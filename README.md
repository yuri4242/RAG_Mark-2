# RAG Mark-2

Haystack 2.24 + ChromaDB を用いた日本語対応 RAG（Retrieval-Augmented Generation）チャットシステムです。
PDF ドキュメントをベクトル化し、質問に対して資料に基づいた回答を生成します。

[rag_mark-1](../rag_mark-1/) の LlamaIndex 版を Haystack 2.24 で再実装したものです。

## 構成

| コンポーネント | 使用技術 |
|---|---|
| LLM | OpenAI GPT-4o |
| Embedding | OpenAI text-embedding-3-large |
| Vector Store | ChromaDB（永続化） |
| PDF 読み込み | PyMuPDF |
| チャンク分割 | 日本語対応カスタムスプリッター（句点・改行区切り） |
| フレームワーク | Haystack 2.24 |

## mark-1 との主な違い

- **フレームワーク**: LlamaIndex → Haystack 2.24
- **チャンク分割**: 日本語テキスト向けのカスタムスプリッター（文字数ベース、句点・改行で区切り）
- **パイプライン可視化**: 各ステップにログコンポーネントを挟み、データの流れを可視化
- **Reranker**: なし（ベクトル検索のみ）

## ディレクトリ構成

```
rag_mark-2/
├── main.py                    # RAG チャットシステム本体
├── evaluate.py                # LLM ベースの評価スクリプト
├── evaluation_results.json    # 評価結果
├── requirements.txt           # Python 依存パッケージ
├── .env.example               # 環境変数テンプレート
├── storage/                   # ChromaDB 永続化データ
├── data -> ../data_for_rag/   # ドキュメントデータ（シンボリックリンク）
└── .gitignore
```

## セットアップ

### 1. Python 仮想環境の作成

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

```bash
cp .env.example .env
```

`.env` ファイルを編集し、OpenAI API キーを設定してください。

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4. ドキュメントの配置

`./data/` ディレクトリ（`../data_for_rag/` へのシンボリックリンク）に PDF ファイルを配置してください。

## 使い方

### チャット

```bash
python main.py
```

対話形式で質問を入力すると、ドキュメントに基づいた回答が返されます。終了するには `exit` と入力します。

#### オプション

| オプション | 説明 |
|---|---|
| `--rebuild` | インデックスを再構築する（既存のストレージを削除） |

```bash
# インデックスの再構築
python main.py --rebuild
```

### 評価

RAG の回答品質を LLM で評価します。テストケースファイル（`test_cases.json`）を用意してください。

```bash
# 一括評価
python evaluate.py --mode simple

# 対話形式で個別に評価
python evaluate.py --mode interactive
```

評価指標:

- **Faithfulness（忠実性）** - 回答がコンテキストに忠実かどうか
- **Answer Relevancy（回答関連性）** - 回答が質問に適切に答えているかどうか

結果は `evaluation_results.json` に保存されます。

## パイプライン構成

### インジェスションパイプライン

```
入力ドキュメント（PDF）
  → DocumentCleaner（空行・余分な空白を除去）
  → JapaneseDocumentSplitter（1000文字、300文字オーバーラップ）
  → OpenAIDocumentEmbedder（ベクトル化）
  → DocumentWriter（ChromaDB へ格納・永続化）
```

### クエリパイプライン

```
ユーザー質問
  → OpenAITextEmbedder（クエリをベクトル化）
  → ChromaEmbeddingRetriever（類似度検索、top_k=10）
  → PromptBuilder（コンテキスト付きプロンプト組み立て）
  → OpenAIGenerator（GPT-4o で回答生成）
```
