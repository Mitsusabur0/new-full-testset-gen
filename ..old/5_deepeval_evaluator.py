import ast
import json
import os
from datetime import datetime

import pandas as pd

from deepeval.models import AmazonBedrockModel
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

import config

# --- CONFIG ---
# INPUT_CSV_PATH = os.getenv("DEEPEVAL_INPUT_CSV", config.OUTPUT_EVALSET_CSV)
# OUTPUT_CSV_PATH = os.getenv(
#     "DEEPEVAL_OUTPUT_CSV",
#     "outputs/test/testset_with_deepeval_metrics.csv"
# )


INPUT_CSV_PATH = "outputs/test1/4_evalset.csv"
OUTPUT_CSV_PATH = "outputs/test1/5_evalset.csv"



DEEPEVAL_MODEL_ID = os.getenv("DEEPEVAL_MODEL_ID", config.MODEL_ID)
DEEPEVAL_REGION = os.getenv("DEEPEVAL_REGION", config.AWS_REGION)
THRESHOLD = float(os.getenv("DEEPEVAL_THRESHOLD", "0.7"))
TEMPERATURE = float(os.getenv("DEEPEVAL_TEMPERATURE", "0.6"))
RUN_SUMMARY_PATH = os.getenv(
    "DEEPEVAL_RUN_SUMMARY_PATH",
    "outputs/test/deepeval_run_summary.json"
)


model = AmazonBedrockModel(
    model=DEEPEVAL_MODEL_ID,
    region=DEEPEVAL_REGION,
    generation_kwargs={"temperature": TEMPERATURE},
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
        ContextualPrecisionMetric(threshold=THRESHOLD, model=model, include_reason=True),
        ContextualRecallMetric(threshold=THRESHOLD, model=model, include_reason=True),
        ContextualRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True),
    )


def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Input file not found: {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)

    if "retrieved_contexts" not in df.columns:
        print("Missing column 'retrieved_contexts'. Run retriever first.")
        return
    if "actual_output" not in df.columns:
        print("Missing column 'actual_output'. Run actual output generation first.")
        return
    if "expected_output" not in df.columns:
        print("Missing column 'expected_output'. Run expected output generation first.")
        return

    precision_metric, recall_metric, relevancy_metric = build_metrics()

    precision_scores = []
    recall_scores = []
    relevancy_scores = []
    precision_reasons = []
    recall_reasons = []
    relevancy_reasons = []
    error_log = []

    for idx, row in df.iterrows():
        actual_output = (row.get("actual_output") or "").strip()
        expected_output = (row.get("expected_output") or "").strip()
        retrieval_context = parse_list_column(row.get("retrieved_contexts", []))
        user_input = (row.get("user_input") or "").strip()

        try:
            test_case_full = LLMTestCase(
                input=user_input,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
            )
            test_case_relevancy = LLMTestCase(
                input=user_input,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )

            precision_metric.measure(test_case_full)
            recall_metric.measure(test_case_full)
            relevancy_metric.measure(test_case_relevancy)

            precision_scores.append(precision_metric.score)
            recall_scores.append(recall_metric.score)
            relevancy_scores.append(relevancy_metric.score)
            precision_reasons.append(precision_metric.reason)
            recall_reasons.append(recall_metric.reason)
            relevancy_reasons.append(relevancy_metric.reason)
        except Exception as e:
            error_log.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "row": int(idx),
                "error": str(e),
            })
            precision_scores.append(None)
            recall_scores.append(None)
            relevancy_scores.append(None)
            precision_reasons.append(None)
            recall_reasons.append(None)
            relevancy_reasons.append(None)

        print(f"[{idx + 1}/{len(df)}] Deepeval metrics computed")

    df["deepeval_contextual_precision"] = precision_scores
    df["deepeval_contextual_precision_reason"] = precision_reasons
    df["deepeval_contextual_recall"] = recall_scores
    df["deepeval_contextual_recall_reason"] = recall_reasons
    df["deepeval_contextual_relevancy"] = relevancy_scores
    df["deepeval_contextual_relevancy_reason"] = relevancy_reasons

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
    main()
