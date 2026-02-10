import os

from deepeval.models import AmazonBedrockModel
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# --- HARD-CODED CONFIG (for quick AWS smoke test) ---
AWS_REGION = "us-east-1"
MODEL_ID = "openai.gpt-oss-120b-1:0"
TEMPERATURE = 0.6
THRESHOLD = 0.7

# Optional: set AWS profile for this run
# Uncomment if you want to force a profile, e.g. "default" or "sandbox"
# os.environ["AWS_PROFILE"] = "default"


model = AmazonBedrockModel(
    model=MODEL_ID,
    region=AWS_REGION,
    generation_kwargs={"temperature": TEMPERATURE},
)

metric = ContextualPrecisionMetric(
    threshold=THRESHOLD,
    model=model,
    include_reason=True,
)

test_case = LLMTestCase(
    input="¿Qué pasa si quiero devolver un producto?",
    actual_output="Ofrecemos reembolso total dentro de 30 días.",
    expected_output="Puedes solicitar un reembolso completo dentro de 30 días.",
    retrieval_context=[
        "Todos los clientes son elegibles para un reembolso total dentro de 30 días."
    ],
)

metric.measure(test_case)
print("Contextual Precision score:", metric.score)
print("Reason:", metric.reason)
