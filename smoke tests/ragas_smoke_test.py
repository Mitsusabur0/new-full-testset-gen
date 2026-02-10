"""
RAGAS smoke test using AWS Bedrock via llm_factory + litellm.
Runs a single metric (Context Precision) on a tiny dataset.
"""

import os
import litellm
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import ContextPrecision

# --- HARD-CODED CONFIG (for quick AWS smoke test) ---
AWS_REGION = "us-east-1"
MODEL_ID = "openai.gpt-oss-120b-1:0"
TEMPERATURE = 0.4

# Optional: set AWS profile for this run
# os.environ["AWS_PROFILE"] = "default"

# Required by litellm Bedrock provider
os.environ["AWS_REGION_NAME"] = AWS_REGION

llm = llm_factory(
    f"bedrock/{MODEL_ID}",
    provider="litellm",
    client=litellm.completion,
    temperature=TEMPERATURE,
)

metric = ContextPrecision(llm=llm)

# Minimal dataset for a single context precision check
sample = Dataset.from_dict(
    {
        "question": ["Where is the Eiffel Tower located?"],
        "ground_truth": ["The Eiffel Tower is located in Paris."],
        "contexts": [[
            "The Eiffel Tower is located in Paris.",
            "The Brandenburg Gate is located in Berlin.",
        ]],
    }
)

result = evaluate(
    dataset=sample,
    metrics=[ContextPrecision(llm=llm)],
)

# Evaluat
print("Context Precision score:", result["context_precision"])


