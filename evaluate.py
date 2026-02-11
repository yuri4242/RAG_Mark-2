"""
Haystack 2.x RAG 評価スクリプト

LLM を使って Faithfulness（忠実性）と Answer Relevancy（回答関連性）を評価する。
mark-1/evaluate.py と同等の機能を Haystack 2.x パイプライン上で再実装。
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# メインの RAG システムからインポート
from main import (
    DATA_DIR,
    JAPANESE_BM25_REGEX,
    SYSTEM_PROMPT,
    BM25_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    build_indexing_pipeline,
    build_query_pipeline,
    load_all_documents,
    pipeline_logger,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore

# ─── 環境変数 ──────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    _fallback_env = Path(__file__).resolve().parent.parent / "rag_mark-1" / ".env"
    if _fallback_env.exists():
        load_dotenv(_fallback_env)

# ─── テストデータ ──────────────────────────────────────────────────
TEST_DATA_FILE = Path("./test_cases.json")
# フォールバック: rag_mark-1 のテストケース
if not TEST_DATA_FILE.exists():
    _fallback_tests = Path(__file__).resolve().parent.parent / "rag_mark-1" / "test_cases.json"
    if _fallback_tests.exists():
        TEST_DATA_FILE = _fallback_tests


def load_test_cases() -> list[dict]:
    """テストケースの読み込み"""
    if not TEST_DATA_FILE.exists():
        sample_cases = [
            {
                "input": "このドキュメントは何について書かれていますか？",
                "expected_output": None,
            },
        ]
        with open(TEST_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(sample_cases, f, ensure_ascii=False, indent=2)
        print(f"📝 サンプルテストケースを {TEST_DATA_FILE} に作成しました。")
        return sample_cases

    with open(TEST_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  RAG システムの初期化
# ═══════════════════════════════════════════════════════════════════

def initialize_rag_system():
    """
    Haystack パイプラインを構築し、ドキュメントを読み込んでインデックス化する。
    Returns: (query_pipeline, document_store)
    """
    import logging
    # 評価時はパイプラインログを抑制
    pipeline_logger.setLevel(logging.WARNING)

    print("📦 InMemoryDocumentStore を初期化中…")
    document_store = InMemoryDocumentStore(
        bm25_tokenization_regex=JAPANESE_BM25_REGEX,
    )

    print(f"📄 ドキュメントを読み込み中… (ソース: {DATA_DIR})")
    documents = load_all_documents()

    print("🔧 インジェスションパイプラインを実行中…")
    indexing_pipeline = build_indexing_pipeline(document_store)
    indexing_result = indexing_pipeline.run({"log_input": {"documents": documents}})
    written = indexing_result.get("writer", {}).get("documents_written", 0)
    print(f"✅ {written} 件のチャンクを格納しました。")

    print("🔧 クエリパイプラインを構築中…")
    query_pipeline = build_query_pipeline(document_store)
    print("✅ 初期化完了\n")

    return query_pipeline


# ═══════════════════════════════════════════════════════════════════
#  RAG クエリ実行
# ═══════════════════════════════════════════════════════════════════

def run_rag_query(query_pipeline, question: str) -> tuple[str, list[str]]:
    """
    Haystack クエリパイプラインを実行し、回答と参照コンテキストを返す。
    """
    result = query_pipeline.run(
        {
            "log_query": {"query": question},
            "prompt_builder": {"system_prompt": SYSTEM_PROMPT},
        },
        include_outputs_from={"log_retrieved"},
    )

    # 回答の取得
    replies = result.get("log_response", {}).get("replies", [])
    answer = replies[0] if replies else ""

    # 参照コンテキストの取得
    contexts = []
    retrieved_docs = result.get("log_retrieved", {}).get("documents", [])
    for doc in retrieved_docs:
        contexts.append(doc.content)

    return answer, contexts


# ═══════════════════════════════════════════════════════════════════
#  LLM 評価
# ═══════════════════════════════════════════════════════════════════

def evaluate_with_llm(
    question: str,
    answer: str,
    contexts: list[str],
    expected: Optional[str],
) -> dict:
    """
    OpenAI API で Faithfulness / Answer Relevancy を評価する。
    """
    client = OpenAI()
    context_text = "\n---\n".join(contexts)

    # ── Faithfulness 評価プロンプト ──
    faithfulness_prompt = f"""以下の回答が、提供されたコンテキストに忠実かどうかを評価してください。
回答がコンテキストに含まれる情報のみに基づいているか、幻覚（コンテキストにない情報の追加）がないかを確認してください。

コンテキスト:
{context_text}

質問: {question}
回答: {answer}

評価結果をJSON形式で出力してください:
{{"score": 0.0から1.0の数値, "reason": "評価理由"}}

スコアの基準:
- 1.0: 回答は完全にコンテキストに基づいている
- 0.7-0.9: 回答はほぼコンテキストに基づいているが、軽微な推論を含む
- 0.4-0.6: 回答の一部がコンテキストに基づいていない
- 0.0-0.3: 回答の大部分がコンテキストに基づいていない（幻覚が多い）
"""

    # ── Answer Relevancy 評価プロンプト ──
    relevancy_prompt = f"""以下の回答が、質問に対して適切かどうかを評価してください。
回答が質問に直接答えているか、関連性があるかを確認してください。

質問: {question}
回答: {answer}
{f"期待される回答: {expected}" if expected else ""}

評価結果をJSON形式で出力してください:
{{"score": 0.0から1.0の数値, "reason": "評価理由"}}

スコアの基準:
- 1.0: 回答は質問に完全に答えている
- 0.7-0.9: 回答は質問にほぼ答えているが、一部不足がある
- 0.4-0.6: 回答は部分的にしか質問に答えていない
- 0.0-0.3: 回答は質問に答えていない
"""

    results = {
        "faithfulness": {"score": None, "reason": None, "error": None},
        "relevancy": {"score": None, "reason": None, "error": None},
    }

    def _call_llm(prompt: str) -> dict:
        """OpenAI API を呼び出し、JSON レスポンスを解析"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}") + 1]
            return json.loads(json_str)
        return {}

    # Faithfulness 評価
    try:
        data = _call_llm(faithfulness_prompt)
        results["faithfulness"]["score"] = float(data.get("score", 0))
        results["faithfulness"]["reason"] = data.get("reason", "")
    except Exception as e:
        results["faithfulness"]["error"] = str(e)

    time.sleep(1)

    # Answer Relevancy 評価
    try:
        data = _call_llm(relevancy_prompt)
        results["relevancy"]["score"] = float(data.get("score", 0))
        results["relevancy"]["reason"] = data.get("reason", "")
    except Exception as e:
        results["relevancy"]["error"] = str(e)

    return results


# ═══════════════════════════════════════════════════════════════════
#  一括評価モード
# ═══════════════════════════════════════════════════════════════════

def run_simple_evaluation(verbose: bool = True):
    """テストケースを一括で評価"""
    print("=" * 60)
    print("RAG 評価システム（Haystack 2.x + BM25 / LLM 直接評価）")
    print("=" * 60)

    # RAG 初期化
    query_pipeline = initialize_rag_system()

    # テストケース読み込み
    print(f"📋 テストケースを読み込み中… ({TEST_DATA_FILE})")
    test_data = load_test_cases()
    print(f"   {len(test_data)} 件のテストケースを読み込みました。")

    results_summary = []
    total_faithfulness = 0
    total_relevancy = 0
    valid_count = 0

    for i, data in enumerate(test_data, 1):
        question = data["input"]
        expected = data.get("expected_output")

        print(f"\n{'='*60}")
        print(f"[{i}/{len(test_data)}] 質問: {question}")
        print("=" * 60)

        # RAG クエリ実行
        answer, contexts = run_rag_query(query_pipeline, question)
        print(f"\n📝 回答:\n{answer}\n")

        # LLM 評価
        print("📊 評価中…")
        eval_results = evaluate_with_llm(question, answer, contexts, expected)

        # Faithfulness 結果
        f_score = eval_results["faithfulness"]["score"]
        f_reason = eval_results["faithfulness"]["reason"]
        f_error = eval_results["faithfulness"]["error"]

        print(f"\n【Faithfulness（忠実性）】")
        if f_error:
            print(f"   ⚠️ エラー: {f_error}")
        elif f_score is not None:
            status = "✅ PASS" if f_score >= 0.7 else "❌ FAIL"
            print(f"   スコア: {f_score:.2f} {status}")
            print(f"   理由: {f_reason}")
            total_faithfulness += f_score
        else:
            print("   ⚠️ スコア取得失敗")

        # Answer Relevancy 結果
        r_score = eval_results["relevancy"]["score"]
        r_reason = eval_results["relevancy"]["reason"]
        r_error = eval_results["relevancy"]["error"]

        print(f"\n【Answer Relevancy（回答関連性）】")
        if r_error:
            print(f"   ⚠️ エラー: {r_error}")
        elif r_score is not None:
            status = "✅ PASS" if r_score >= 0.7 else "❌ FAIL"
            print(f"   スコア: {r_score:.2f} {status}")
            print(f"   理由: {r_reason}")
            total_relevancy += r_score
        else:
            print("   ⚠️ スコア取得失敗")

        if f_score is not None and r_score is not None:
            valid_count += 1

        results_summary.append({
            "question": question,
            "answer": answer,
            "expected": expected,
            "faithfulness": eval_results["faithfulness"],
            "relevancy": eval_results["relevancy"],
        })

        time.sleep(1)

    # ── サマリー ──
    print("\n" + "=" * 60)
    print("📈 評価結果サマリー")
    print("=" * 60)

    if valid_count > 0:
        avg_faithfulness = total_faithfulness / valid_count
        avg_relevancy = total_relevancy / valid_count
        print(f"\n評価完了: {valid_count}/{len(test_data)} 件")
        print(f"\n平均スコア:")
        print(f"  Faithfulness:     {avg_faithfulness:.2f} {'✅' if avg_faithfulness >= 0.7 else '❌'}")
        print(f"  Answer Relevancy: {avg_relevancy:.2f} {'✅' if avg_relevancy >= 0.7 else '❌'}")
    else:
        print("\n⚠️ 有効な評価結果がありません。")

    # 結果保存
    output_file = Path("./evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n📁 詳細結果を {output_file} に保存しました。")

    return results_summary


# ═══════════════════════════════════════════════════════════════════
#  対話評価モード
# ═══════════════════════════════════════════════════════════════════

def interactive_evaluation():
    """対話形式で単一の質問を評価"""
    print("=" * 60)
    print("RAG 対話評価モード (Haystack 2.x + BM25)")
    print("=" * 60)

    query_pipeline = initialize_rag_system()

    print("質問を入力してください（終了するには 'exit'）")
    print("-" * 60)

    while True:
        try:
            question = input("\n質問: ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("評価を終了します。")
                break

            # RAG クエリ実行
            print("\n🔍 回答を生成中…")
            answer, contexts = run_rag_query(query_pipeline, question)

            print(f"\n📝 回答:\n{answer}")
            print(f"\n📚 参照コンテキスト数: {len(contexts)}")

            # 評価
            print("\n📊 評価中…")
            eval_results = evaluate_with_llm(question, answer, contexts, None)

            # 結果表示
            f_score = eval_results["faithfulness"]["score"]
            r_score = eval_results["relevancy"]["score"]

            print(f"\n【Faithfulness】 スコア: {f_score:.2f}" if f_score else "\n【Faithfulness】 スコア: N/A")
            if eval_results["faithfulness"]["reason"]:
                print(f"   理由: {eval_results['faithfulness']['reason']}")

            print(f"\n【Answer Relevancy】 スコア: {r_score:.2f}" if r_score else "\n【Answer Relevancy】 スコア: N/A")
            if eval_results["relevancy"]["reason"]:
                print(f"   理由: {eval_results['relevancy']['reason']}")

            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n\n評価を終了します。")
            break
        except Exception as e:
            print(f"\n❌ エラー: {e}")


# ═══════════════════════════════════════════════════════════════════
#  メイン
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 評価スクリプト (Haystack 2.x)")
    parser.add_argument(
        "--mode",
        choices=["simple", "interactive"],
        default="simple",
        help="評価モード: simple（一括評価）、interactive（対話評価）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力を有効にする",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY が設定されていません。")
    elif args.mode == "simple":
        run_simple_evaluation(verbose=args.verbose)
    else:
        interactive_evaluation()
