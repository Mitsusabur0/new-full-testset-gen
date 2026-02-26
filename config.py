import os

# --- PATHS ---
KB_FOLDER = os.getenv("KB_FOLDER", "./kb_small_testfolder")
PIPELINE_OUTPUT_DIR = os.getenv("PIPELINE_OUTPUT_DIR", "outputs/small_test")


PIPELINE_CSV = os.getenv(
    "PIPELINE_CSV",
    os.path.join(PIPELINE_OUTPUT_DIR, "pipeline_state.csv")
)
OUTPUT_RESULTS_PARQUET = os.getenv(
    "OUTPUT_RESULTS_PARQUET",
    os.path.join(PIPELINE_OUTPUT_DIR, "pipeline_state.parquet")
)

# --- AWS CONFIG ---
KB_SERVICE = os.getenv("KB_SERVICE", "bedrock-agent-runtime")
KB_ID = os.getenv("KB_ID", "J7JNHSZPJ3")
# KB_ID = os.getenv("KB_ID", "3TPM53DPBN")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE_LLM = os.getenv("AWS_PROFILE_DEFAULT", "default")
AWS_PROFILE_SANDBOX = os.getenv("AWS_PROFILE_SANDBOX", "sandbox")



AWS_PROFILE_DESA_BEDROCK = os.getenv("AWS_PROFILE_DESA_BEDROCK", "943897082379_BECH_ReadOnlyBedrock")
AWS_PROFILE_DESA_ACCESS = os.getenv("AWS_PROFILE_DESA_ACCESS", "943897082379_BECH_ReadOnlyAccess")

# --- LLM CONFIG ---
MODEL_ID = os.getenv("MODEL_ID", "openai.gpt-oss-120b-1:0")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# LLM PRICING PER 1K TOKENS
INPUT_PRICE = float(os.getenv("INPUT_PRICE", "0.00015"))
OUTPUT_PRICE = float(os.getenv("OUTPUT_PRICE", "0.0006"))

# --- RETRIEVAL / EVAL ---
TOP_K = int(os.getenv("TOP_K", "3"))
EVAL_K = int(os.getenv("EVAL_K", "3"))

# --- REPRODUCIBILITY ---
SEED = int(os.getenv("SEED", "42"))

# --- RETRIES ---
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_BASE_SECONDS = float(os.getenv("BACKOFF_BASE_SECONDS", "1.0"))
BACKOFF_MAX_SECONDS = float(os.getenv("BACKOFF_MAX_SECONDS", "8.0"))
BACKOFF_JITTER_SECONDS = float(os.getenv("BACKOFF_JITTER_SECONDS", "0.3"))
