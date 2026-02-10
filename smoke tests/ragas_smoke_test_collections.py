"""
RAGAS smoke test using collections ContextPrecision (async ascore()).
Uses AWS Bedrock via llm_factory + litellm.
"""

import asyncio
import os
import litellm
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

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
    client=litellm.acompletion,
    temperature=TEMPERATURE,
)

scorer = ContextPrecision(llm=llm)


async def main() -> None:
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Eiffel Tower is located in Paris.",
            "The Brandenburg Gate is located in Berlin.",
        ],
    )

    print("Context Precision score:", result.value)


if __name__ == "__main__":
    asyncio.run(main())
