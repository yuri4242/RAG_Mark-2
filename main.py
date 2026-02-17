"""
Haystack 2.x を用いた RAG システム

- ChromaDB 永続化 + OpenAI Embedding ベクトル検索
- Pipeline クラスによるインジェスション / クエリフローの明示的記述
- 各ステップのデータ入出力ログによるパイプライン可視化

LlamaIndex 版 (rag_mark-1/main.py) と同等の機能を Haystack 2.x で再実装。
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv

from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


# ─── 環境変数 ──────────────────────────────────────────────────────
load_dotenv()
# フォールバック: rag_mark-1 の .env も参照
if not os.getenv("OPENAI_API_KEY"):
    _fallback_env = Path(__file__).resolve().parent.parent / "rag_mark-1" / ".env"
    if _fallback_env.exists():
        load_dotenv(_fallback_env)

# ─── ログ設定 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # Haystack 内部ログは WARNING 以上のみ
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# パイプライン可視化用ロガー（INFO レベルで出力）
pipeline_logger = logging.getLogger("rag_pipeline")
pipeline_logger.setLevel(logging.INFO)

# ─── 定数 ──────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
# フォールバック: rag_mark-1 の data ディレクトリ
if not DATA_DIR.exists():
    _fallback_data = Path(__file__).resolve().parent.parent / "rag_mark-1" / "data"
    if _fallback_data.exists():
        DATA_DIR = _fallback_data

# ChromaDB 永続化設定
STORAGE_DIR = Path("./storage")
COLLECTION_NAME = "rag_collection"

# チャンク分割設定（文字数ベース、LlamaIndex 版の 1000 トークンに相当）
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

# ベクトル検索設定
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 10

# ─── プロンプト ────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "あなたは提供された資料に基づいて回答する専門家です。\n"
    "資料に答えがない場合は、推測せず「資料には記載がありません」と明確に伝えてください。\n"
    "回答には、参照したファイル名を必ず含めてください。"
)

RAG_PROMPT_TEMPLATE = """\
以下のコンテキスト情報を参考にして、ユーザーの質問に回答してください。

## システム指示
{{ system_prompt }}

## コンテキスト
{% for doc in documents %}
--- ソース: {{ doc.meta.get("file_name", "不明") }} (ページ {{ doc.meta.get("page_number", "N/A") }}) ---
{{ doc.content }}
{% endfor %}

## ユーザーの質問
{{ query }}

## 回答
"""


# ═══════════════════════════════════════════════════════════════════
#  カスタムコンポーネント: パイプライン可視化用ロガー
# ═══════════════════════════════════════════════════════════════════

@component
class DocumentLogger:
    """
    Document リストをログ出力し、そのまま次のコンポーネントへ渡すパススルー。
    パイプライン内にこのコンポーネントを挟むことで、各ステップの入出力を可視化する。
    """

    def __init__(self, step_name: str, max_preview: int = 80):
        self.step_name = step_name
        self.max_preview = max_preview

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        log = pipeline_logger
        border = "=" * 60
        log.info(border)
        log.info(f"📋 [{self.step_name}] ドキュメント数: {len(documents)}")

        for i, doc in enumerate(documents[:5]):
            preview = doc.content[:self.max_preview].replace("\n", "↵")
            meta_keys = {
                k: v for k, v in doc.meta.items()
                if k in ("file_name", "page_number", "chunk_index", "source_doc_id")
            }
            score_str = f", score={doc.score:.4f}" if doc.score is not None else ""
            log.info(f"  [{i + 1}] meta={meta_keys}{score_str}")
            log.info(f"       \"{preview}…\"")
            log.info(f"       (文字数: {len(doc.content)})")
        if len(documents) > 5:
            log.info(f"  … 他 {len(documents) - 5} 件省略")
        log.info(border)

        return {"documents": documents}


@component
class QueryLogger:
    """クエリ文字列をログ出力し、そのまま次へ渡すパススルー。"""

    def __init__(self, step_name: str):
        self.step_name = step_name

    @component.output_types(query=str)
    def run(self, query: str) -> dict:
        pipeline_logger.info("=" * 60)
        pipeline_logger.info(f"🔍 [{self.step_name}] クエリ: \"{query}\"")
        pipeline_logger.info("=" * 60)
        return {"query": query}


@component
class GenerationLogger:
    """LLM 生成結果をログ出力し、そのまま次へ渡すパススルー。"""

    def __init__(self, step_name: str, max_preview: int = 200):
        self.step_name = step_name
        self.max_preview = max_preview

    @component.output_types(replies=List[str])
    def run(self, replies: List[str]) -> dict:
        log = pipeline_logger
        log.info("=" * 60)
        log.info(f"🤖 [{self.step_name}] 生成レスポンス数: {len(replies)}")
        if replies:
            preview = replies[0][:self.max_preview].replace("\n", "↵")
            log.info(f"  プレビュー: \"{preview}…\"")
        log.info("=" * 60)
        return {"replies": replies}


# ═══════════════════════════════════════════════════════════════════
#  カスタムコンポーネント: 日本語ドキュメント分割
# ═══════════════════════════════════════════════════════════════════

@component
class JapaneseDocumentSplitter:
    """
    日本語テキストを文字数ベースで分割するカスタムコンポーネント。
    句点「。」や改行で文を区切りつつ、指定サイズでチャンク化する。

    Haystack 標準の DocumentSplitter は英語の空白区切りを前提としているため、
    日本語テキストには本コンポーネントを使用する。
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        result = []
        for doc in documents:
            chunks = self._split_text(doc.content)
            for idx, chunk in enumerate(chunks):
                new_meta = dict(doc.meta)
                new_meta["chunk_index"] = idx
                new_meta["source_doc_id"] = doc.id
                result.append(Document(content=chunk, meta=new_meta))
        return {"documents": result}

    def _split_text(self, text: str) -> List[str]:
        """句点・改行で文分割し、chunk_size 文字ごとにまとめる"""
        # 日本語の文末記号で分割（区切り文字は前の文に含める）
        sentences = re.split(r"(?<=[。\uFF01\uFF1F\n])", text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > self.chunk_size and current:
                chunks.append(current.strip())
                # オーバーラップ: 現チャンク末尾を次チャンクの先頭に引き継ぐ
                if len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap :] + sentence
                else:
                    current = current + sentence
            else:
                current += sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks


# ═══════════════════════════════════════════════════════════════════
#  PDF 読み込み
# ═══════════════════════════════════════════════════════════════════

def load_pdf_with_pymupdf(file_path: str) -> List[Document]:
    """PyMuPDF で PDF を読み込み、ページごとに Haystack Document を生成"""
    pdf = fitz.open(file_path)
    documents = []
    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    content=text,
                    meta={
                        "file_name": Path(file_path).name,
                        "file_path": file_path,
                        "page_number": page_num + 1,
                    },
                )
            )
    pdf.close()
    return documents


def load_all_documents() -> List[Document]:
    """data/ ディレクトリ配下の全 PDF を読み込む"""
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise FileNotFoundError(
            f"⚠️  {DATA_DIR} ディレクトリにファイルがありません。\n"
            "   PDF を配置してから再実行してください。"
        )

    all_docs = []
    for fp in sorted(DATA_DIR.iterdir()):
        if fp.suffix.lower() == ".pdf":
            try:
                docs = load_pdf_with_pymupdf(str(fp))
                all_docs.extend(docs)
                print(f"  ✅ {fp.name}: {len(docs)} ページ")
            except Exception as e:
                print(f"  ❌ {fp.name}: 読み込み失敗 ({e})")

    print(f"\n  合計 {len(all_docs)} ページ分のドキュメントを読み込みました。")
    return all_docs


# ═══════════════════════════════════════════════════════════════════
#  インジェスションパイプライン
# ═══════════════════════════════════════════════════════════════════

def build_indexing_pipeline(document_store: ChromaDocumentStore) -> Pipeline:
    """
    ドキュメント取り込みパイプラインを構築する。

    フロー:
      入力ドキュメント (PyMuPDF で読み込んだ生ページ)
        → log_input      : ログ出力（生ドキュメントの状態を確認）
        → cleaner        : 空行・余分な空白を除去
        → log_cleaned    : ログ出力（クリーニング後の状態を確認）
        → splitter       : 日本語対応チャンク分割 (1000文字, 300文字オーバーラップ)
        → log_split      : ログ出力（分割後のチャンク数・内容を確認）
        → doc_embedder   : OpenAI Embedding でベクトル化
        → log_embedded   : ログ出力（エンベディング後の状態を確認）
        → writer         : InMemoryDocumentStore へ書き込み
    """
    pipe = Pipeline()

    # ── コンポーネント登録 ─────────────────────────────────────
    pipe.add_component(
        "log_input",
        DocumentLogger("1. 入力ドキュメント（生PDF）"),
    )
    pipe.add_component(
        "cleaner",
        DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
        ),
    )
    pipe.add_component(
        "log_cleaned",
        DocumentLogger("2. クリーニング後"),
    )
    pipe.add_component(
        "splitter",
        JapaneseDocumentSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ),
    )
    pipe.add_component(
        "log_split",
        DocumentLogger("3. チャンク分割後"),
    )
    pipe.add_component(
        "doc_embedder",
        OpenAIDocumentEmbedder(model=EMBEDDING_MODEL),
    )
    pipe.add_component(
        "log_embedded",
        DocumentLogger("4. エンベディング後"),
    )
    pipe.add_component(
        "writer",
        DocumentWriter(document_store=document_store),
    )

    # ── コンポーネント接続 ─────────────────────────────────────
    # 各ステップを直列に接続。ログコンポーネントはパススルーとして間に挟む。
    pipe.connect("log_input.documents", "cleaner.documents")
    pipe.connect("cleaner.documents", "log_cleaned.documents")
    pipe.connect("log_cleaned.documents", "splitter.documents")
    pipe.connect("splitter.documents", "log_split.documents")
    pipe.connect("log_split.documents", "doc_embedder.documents")
    pipe.connect("doc_embedder.documents", "log_embedded.documents")
    pipe.connect("log_embedded.documents", "writer.documents")

    return pipe


# ═══════════════════════════════════════════════════════════════════
#  クエリパイプライン
# ═══════════════════════════════════════════════════════════════════

def build_query_pipeline(document_store: ChromaDocumentStore) -> Pipeline:
    """
    質問応答パイプラインを構築する。

    フロー:
      ユーザー質問 (query)
        → log_query        : ログ出力（受信クエリを確認）
        ├→ text_embedder   : OpenAI Embedding でクエリをベクトル化
        │   → retriever    : ベクトル類似度検索 (top_k=10)
        │       → log_retrieved : ログ出力（検索結果ドキュメントを確認）
        │           → prompt_builder : Jinja テンプレートでプロンプト組み立て
        └→ prompt_builder    ← query も直接渡す
              → llm            : OpenAI gpt-4o で回答生成 (temperature=0.1)
                  → log_response : ログ出力（生成結果を確認）
    """
    pipe = Pipeline()

    # ── コンポーネント登録 ─────────────────────────────────────
    pipe.add_component(
        "log_query",
        QueryLogger("1. ユーザークエリ受信"),
    )
    pipe.add_component(
        "text_embedder",
        OpenAITextEmbedder(model=EMBEDDING_MODEL),
    )
    pipe.add_component(
        "retriever",
        ChromaEmbeddingRetriever(
            document_store=document_store,
            top_k=TOP_K,
        ),
    )
    pipe.add_component(
        "log_retrieved",
        DocumentLogger("2. ベクトル検索結果"),
    )
    pipe.add_component(
        "prompt_builder",
        PromptBuilder(template=RAG_PROMPT_TEMPLATE),
    )
    pipe.add_component(
        "llm",
        OpenAIGenerator(
            model="gpt-4o",
            generation_kwargs={"temperature": 0.1},
        ),
    )
    pipe.add_component(
        "log_response",
        GenerationLogger("3. LLM 生成結果"),
    )

    # ── コンポーネント接続 ─────────────────────────────────────
    # query は log_query から text_embedder と prompt_builder の両方へ分岐
    pipe.connect("log_query.query", "text_embedder.text")
    pipe.connect("log_query.query", "prompt_builder.query")

    # エンベディング → 検索 → ログ → プロンプトビルダー → LLM → ログ
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "log_retrieved.documents")
    pipe.connect("log_retrieved.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    pipe.connect("llm.replies", "log_response.replies")

    return pipe


# ═══════════════════════════════════════════════════════════════════
#  チャットループ
# ═══════════════════════════════════════════════════════════════════

def chat_loop(query_pipeline: Pipeline):
    """CLI チャットループ"""
    print("\n" + "=" * 50)
    print("RAG チャットシステム (Haystack 2.x + ベクトル検索)")
    print("質問を入力してください（終了するには 'exit' と入力）")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("あなた: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("チャットを終了します。")
                break

            # ─── クエリパイプライン実行 ───────────────────────────
            result = query_pipeline.run(
                {
                    "log_query": {"query": user_input},
                    "prompt_builder": {"system_prompt": SYSTEM_PROMPT},
                },
                include_outputs_from={"log_retrieved"},
            )

            # ─── 回答表示 ────────────────────────────────────────
            replies = result.get("log_response", {}).get("replies", [])
            if replies:
                print(f"\nアシスタント: {replies[0]}\n")

            # ─── 参照ソース表示 ──────────────────────────────────
            retrieved = result.get("log_retrieved", {}).get("documents", [])
            if retrieved:
                print("--- 参照ソース ---")
                for i, doc in enumerate(retrieved, 1):
                    fname = doc.meta.get("file_name", "不明")
                    page = doc.meta.get("page_number", "N/A")
                    score = doc.score
                    if isinstance(score, float):
                        print(f"  [{i}] {fname} (p.{page}, 類似度スコア: {score:.4f})")
                    else:
                        print(f"  [{i}] {fname} (p.{page})")
                print("------------------\n")

        except KeyboardInterrupt:
            print("\n\nチャットを終了します。")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}\n")
            pipeline_logger.exception("クエリ実行時エラー")


# ═══════════════════════════════════════════════════════════════════
#  メイン
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG チャット (Haystack 2.x + ベクトル検索)")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="インデックスを再構築する（既存のストレージを削除）",
    )
    args = parser.parse_args()

    # ── API キー確認 ──────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ エラー: OPENAI_API_KEY が設定されていません。")
        print("   .env ファイルを作成し、APIキーを設定してください。")
        return

    # ── --rebuild オプション ──────────────────────────────────
    if args.rebuild and STORAGE_DIR.exists():
        print("🗑️  既存のインデックスを削除しています…")
        shutil.rmtree(STORAGE_DIR)
        print("✅ 削除完了。再構築します。")

    try:
        # ── 1. ChromaDocumentStore 初期化 ────────────────────
        print(f"📦 ChromaDocumentStore を初期化中… (永続化先: {STORAGE_DIR})")
        document_store = ChromaDocumentStore(
            collection_name=COLLECTION_NAME,
            persist_path=str(STORAGE_DIR),
        )

        # ── 2. 既存インデックスの確認 ────────────────────────
        existing_count = document_store.count_documents()
        if existing_count > 0:
            print(f"📂 既存のインデックスを読み込みました。({existing_count} 件のチャンク)")
        else:
            # ── 3. ドキュメント読み込み・インジェスション ──────
            print(f"📄 ドキュメントを読み込み中… (ソース: {DATA_DIR})")
            documents = load_all_documents()

            print("\n🔧 インジェスションパイプラインを構築中…")
            indexing_pipeline = build_indexing_pipeline(document_store)

            print("\n📊 インジェスションパイプライン構造:")
            print(indexing_pipeline)

            print("\n🚀 インジェスションパイプラインを実行中…")
            indexing_result = indexing_pipeline.run(
                {"log_input": {"documents": documents}}
            )
            written = indexing_result.get("writer", {}).get("documents_written", 0)
            print(f"\n✅ DocumentStore に {written} 件のチャンクを格納しました。")
            print(f"   (DocumentStore 内の総ドキュメント数: {document_store.count_documents()})")

        # ── 4. クエリパイプライン ─────────────────────────────
        print("\n🔧 クエリパイプラインを構築中…")
        query_pipeline = build_query_pipeline(document_store)

        print("\n📊 クエリパイプライン構造:")
        print(query_pipeline)

        # ── 5. チャットループ ─────────────────────────────────
        chat_loop(query_pipeline)

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
