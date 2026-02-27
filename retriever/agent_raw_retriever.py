import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# -----------------------------
# REQUIRED CONFIG (fill these)
# -----------------------------
AWS_PROFILE = "sandbox"
AWS_REGION = "us-east-1"
AGENT_ID = "UKQEMRZQUS"
AGENT_ALIAS_ID = "JPSUY1DN1P"
INPUT_TEXT = "dime en código python cómo compro una casa"

# Optional. If empty, a timestamp-based ID is used.
SESSION_ID = ""

# Set to True if you want Bedrock Agent trace events returned.
ENABLE_TRACE = True

# JSON output written in the same folder as this script.
OUTPUT_JSON_PATH = Path(__file__).resolve().parent / "agent_raw_response.json"


def ensure_session_id(value: str) -> str:
    if value.strip():
        return value.strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"retriever-agent-{timestamp}"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {
            "__type": "bytes",
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    return str(value)


def invoke_bedrock_agent() -> Dict[str, Any]:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    client = session.client("bedrock-agent-runtime")

    request_payload = {
        "agentId": AGENT_ID,
        "agentAliasId": AGENT_ALIAS_ID,
        "sessionId": ensure_session_id(SESSION_ID),
        "inputText": INPUT_TEXT,
        "enableTrace": ENABLE_TRACE,
    }

    response = client.invoke_agent(**request_payload)

    # Materialize the streaming completion events so the full SDK response
    # can be written to JSON.
    raw_response = {k: v for k, v in response.items()}
    raw_response["completion"] = [event for event in response.get("completion", [])]
    return make_json_safe(raw_response)


def main() -> None:
    try:
        result = invoke_bedrock_agent()
    except (ClientError, BotoCoreError, ValueError) as exc:
        error_payload = {
            "error": str(exc),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        OUTPUT_JSON_PATH.write_text(
            json.dumps(error_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise

    OUTPUT_JSON_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved full Bedrock Agent reply to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
