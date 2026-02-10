import csv
import json
import os
import random
import time
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

import config

# --- CONFIG ---
INPUT_CSV_PATH = os.getenv(
    "ACTUAL_INPUT_CSV_PATH",
    "outputs/test1/2_testset_with_expected_outputs.csv"
)
OUTPUT_CSV_PATH = os.getenv(
    "ACTUAL_OUTPUT_CSV_PATH",
    "outputs/test1/3_testset_with_actual_outputs.csv"
)
# INPUT_CSV_PATH = "outputs/subset/2_testset_with_expected_outputs.csv"
# OUTPUT_CSV_PATH = "outputs/subset/3_testset_with_actual_outputs.csv"

AWS_REGION = os.getenv("ACTUAL_OUTPUT_AWS_REGION", config.AWS_REGION)
AWS_PROFILE_SANDBOX = os.getenv("ACTUAL_OUTPUT_AWS_PROFILE", config.AWS_PROFILE_SANDBOX)
AGENT_ID = os.getenv("ACTUAL_OUTPUT_AGENT_ID", "UKQEMRZQUS")
AGENT_ALIAS_ID = os.getenv("ACTUAL_OUTPUT_AGENT_ALIAS_ID", "JPSUY1DN1P")
MAX_RETRIES = int(os.getenv("ACTUAL_OUTPUT_MAX_RETRIES", str(config.MAX_RETRIES)))
BACKOFF_BASE_SECONDS = float(
    os.getenv("ACTUAL_OUTPUT_BACKOFF_BASE_SECONDS", str(config.BACKOFF_BASE_SECONDS))
)
BACKOFF_MAX_SECONDS = float(
    os.getenv("ACTUAL_OUTPUT_BACKOFF_MAX_SECONDS", str(config.BACKOFF_MAX_SECONDS))
)
BACKOFF_JITTER_SECONDS = float(
    os.getenv("ACTUAL_OUTPUT_BACKOFF_JITTER_SECONDS", str(config.BACKOFF_JITTER_SECONDS))
)
RUN_SUMMARY_PATH = os.getenv(
    "ACTUAL_OUTPUT_RUN_SUMMARY_PATH",
    "outputs/test/actual_outputs_run_summary.json"
)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_agent_client():
    session = boto3.Session(profile_name=AWS_PROFILE_SANDBOX)
    return session.client(service_name="bedrock-agent-runtime", region_name=AWS_REGION)


def backoff_sleep(attempt):
    base = BACKOFF_BASE_SECONDS * (2 ** attempt)
    sleep_for = min(base, BACKOFF_MAX_SECONDS)
    sleep_for += random.uniform(0, BACKOFF_JITTER_SECONDS)
    time.sleep(sleep_for)


def call_with_retry(fn, operation_name, error_log):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except ClientError as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < MAX_RETRIES:
            backoff_sleep(attempt)
        else:
            if last_error is not None:
                error_log.append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "operation": operation_name,
                    "error": str(last_error),
                })
            return None


def extract_agent_text(response):
    if not response:
        return ""

    completion = response.get("completion")
    if completion is None:
        return ""

    text_parts = []
    for event in completion:
        if not isinstance(event, dict):
            continue
        if "chunk" in event and isinstance(event["chunk"], dict):
            chunk = event["chunk"].get("bytes")
            if isinstance(chunk, (bytes, bytearray)):
                text_parts.append(chunk.decode("utf-8", errors="ignore"))
        if "trace" in event:
            continue
    return "".join(text_parts).strip()


def invoke_agent(user_input, client, error_log, session_id):
    def _call():
        return client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=user_input
        )

    response = call_with_retry(_call, "invoke_agent", error_log)
    return extract_agent_text(response) if response else ""


def build_output_columns(input_columns):
    actual_column = "actual_output"
    if actual_column in input_columns:
        return input_columns

    columns = list(input_columns)
    if "expected_output" in columns:
        idx = columns.index("expected_output")
        columns.insert(idx + 1, actual_column)
    else:
        columns.append(actual_column)
    return columns


def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Input file not found: {INPUT_CSV_PATH}")
        return

    ensure_parent_dir(OUTPUT_CSV_PATH)
    client = get_agent_client()
    error_log = []

    with open(INPUT_CSV_PATH, "r", encoding="utf-8", newline="") as in_file, open(
        OUTPUT_CSV_PATH, "w", encoding="utf-8", newline=""
    ) as out_file:
        reader = csv.DictReader(in_file)
        if not reader.fieldnames:
            print(f"No header found in input file: {INPUT_CSV_PATH}")
            return

        output_columns = build_output_columns(reader.fieldnames)
        writer = csv.DictWriter(out_file, fieldnames=output_columns)
        writer.writeheader()

        processed_rows = 0
        for idx, row in enumerate(reader, start=1):
            user_input = (row.get("user_input") or "").strip()
            actual_output = ""

            if user_input:
                session_id = f"row-{idx}-{int(time.time() * 1000)}"
                actual_output = invoke_agent(user_input, client, error_log, session_id)

            output_row = {}
            for col in output_columns:
                if col == "actual_output":
                    output_row[col] = actual_output
                else:
                    output_row[col] = row.get(col, "")

            writer.writerow(output_row)
            processed_rows += 1
            print(f"[{idx}] Actual output generated")

    print(f"Done. Processed {processed_rows} rows.")
    print(f"Saved file: {OUTPUT_CSV_PATH}")

    if error_log:
        ensure_parent_dir(RUN_SUMMARY_PATH)
        with open(RUN_SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
            json.dump({
                "processed_rows": processed_rows,
                "errors": error_log,
            }, summary_file, ensure_ascii=False, indent=2)
        print(f"Run summary with errors saved to: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
