import ast
import json
import os
import asyncio
from datetime import datetime

import litellm
import pandas as pd

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
)

import config

# --- CONFIG ---
# INPUT_CSV_PATH = os.getenv(
#     "RAGAS_INPUT_CSV",
#     "outputs/test/testset_with_deepeval_metrics.csv",
# )
# OUTPUT_CSV_PATH = os.getenv(
#     "RAGAS_OUTPUT_CSV",
#     "outputs/test/eval_set_deep_ragas.csv",
# )
INPUT_CSV_PATH = "outputs/test1/5_evalset.csv"
OUTPUT_CSV_PATH = "outputs/test1/6_evalset.csv"

RAGAS_MODEL_ID = os.getenv("RAGAS_MODEL_ID", "openai.gpt-oss-120b-1:0")
RAGAS_REGION = os.getenv("RAGAS_REGION", config.AWS_REGION)
RAGAS_TEMPERATURE = float(os.getenv("RAGAS_TEMPERATURE", "0.4"))
RUN_SUMMARY_PATH = os.getenv(
    "RAGAS_RUN_SUMMARY_PATH",
    "outputs/test/ragas_run_summary.json",
)

# Optional: set AWS profile for this run
# os.environ["AWS_PROFILE"] = "default"

# Required by litellm Bedrock provider
os.environ["AWS_REGION_NAME"] = RAGAS_REGION


llm = llm_factory(
    f"bedrock/{RAGAS_MODEL_ID}",
    provider="litellm",
    client=litellm.acompletion,
    temperature=RAGAS_TEMPERATURE,
    max_tokens=15000
)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_list_column(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            return parsed if isinstance(parsed, list) else [stripped]
        except Exception:
            return [stripped]
    return []


def build_metrics():
    return (
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        ContextEntityRecall(llm=llm),
    )


async def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Input file not found: {INPUT_CSV_PATH}")
        return

    # Resume behavior: if output exists, continue from it; otherwise start from input.
    source_path = OUTPUT_CSV_PATH if os.path.exists(OUTPUT_CSV_PATH) else INPUT_CSV_PATH
    df = pd.read_csv(source_path)

    if "retrieved_contexts" not in df.columns:
        print("Missing column 'retrieved_contexts'. Run retriever first.")
        return
    if "expected_output" not in df.columns:
        print("Missing column 'expected_output'. Run expected output generation first.")
        return
    if "user_input" not in df.columns:
        print("Missing column 'user_input'.")
        return

    precision_metric, recall_metric, entity_recall_metric = build_metrics()

    def is_filled(value):
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    precision_scores = []
    recall_scores = []
    entity_recall_scores = []
    error_log = []

    for idx, row in df.iterrows():
        user_input = (row.get("user_input") or "").strip()
        reference = (row.get("expected_output") or "").strip()
        retrieved_contexts = parse_list_column(row.get("retrieved_contexts", []))
        # print(retrieved_contexts)

        existing_precision = row.get("ragas_context_precision")
        existing_recall = row.get("ragas_context_recall")
        existing_entity_recall = row.get("ragas_context_entity_recall")

        if is_filled(existing_precision) and is_filled(existing_recall) and is_filled(existing_entity_recall):
            precision_scores.append(existing_precision)
            recall_scores.append(existing_recall)
            entity_recall_scores.append(existing_entity_recall)
            print(f"[{idx + 1}/{len(df)}] RAGAS metrics already filled, skipped")
            continue

        try:
            precision_result = await precision_metric.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            recall_result = await recall_metric.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            entity_recall_result = await entity_recall_metric.ascore(
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )

            precision_scores.append(precision_result.value)
            recall_scores.append(recall_result.value)
            entity_recall_scores.append(entity_recall_result.value)
        except Exception as e:
            error_log.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "row": int(idx),
                "error": str(e),
            })
            precision_scores.append(existing_precision if is_filled(existing_precision) else None)
            recall_scores.append(existing_recall if is_filled(existing_recall) else None)
            entity_recall_scores.append(existing_entity_recall if is_filled(existing_entity_recall) else None)

        print(f"[{idx + 1}/{len(df)}] RAGAS metrics computed")

    df["ragas_context_precision"] = precision_scores
    df["ragas_context_recall"] = recall_scores
    df["ragas_context_entity_recall"] = entity_recall_scores

    ensure_parent_dir(OUTPUT_CSV_PATH)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved to {OUTPUT_CSV_PATH}")

    if error_log:
        ensure_parent_dir(RUN_SUMMARY_PATH)
        with open(RUN_SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
            json.dump({
                "rows": len(df),
                "errors": error_log,
            }, summary_file, ensure_ascii=False, indent=2)
        print(f"Run summary with errors saved to: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
