# Repository Overview: RAG Evaluation Pipeline

## High-level structure

This repository is organized around a single retrieval-evaluation workflow for a RAG stack, with:

- Synthetic query generation from KB chunks.
- Retrieval via Bedrock KB/API (direct or Agent path).
- Optional reranker-based relevance scoring.
- Offline metric computation.
- Optional Streamlit dashboard for result inspection.

The logical flow is:

1. Generate test queries from source KB content.
2. Run retrieval for each query.
3. Optionally compute pairwise semantic relevance scores.
4. Compute hit/rank/precision/recall metrics.
5. Export and inspect results in dashboards / artifacts.

---

## Root-level pipeline files

### `1_generate_user_inputs.py`
- Purpose: Builds synthetic query dataset from Markdown KB documents.
- Input:
  - Reads all `.md` files in `config.KB_FOLDER`.
- Main flow:
  - Loads KB files and for each text chunk calls an LLM in Bedrock (via `AWS_PROFILE_LLM`) for each defined `QUERY_STYLE`.
  - Enforces XML output (`<style_name>`, `<user_input>`) and retry/backoff logic.
  - Handles parse failures with fallback repair call and logs raw failures.
  - Appends rows incrementally to `PIPELINE_CSV`.
- Outputs:
  - `PIPELINE_CSV` columns include `user_input`, `reference_contexts`, `query_style`, `source_file`.
  - Progress and summary files under the same output directory.
- Resume behavior: Tracks `(file_path, style_name)` in `generation_progress.jsonl` to continue partially completed runs.

### `2_retriever.py`
- Purpose: Executes direct KB vector retrieval for each generated query.
- Input:
  - Reads `PIPELINE_CSV` created by step 1 and uses `user_input`.
- Main flow:
  - Calls Bedrock runtime `retrieve` with `TOP_K`.
  - Extracts retrieved context text and source URI per result.
  - Writes retrieved lists back to `PIPELINE_CSV`.
- Outputs:
  - Updates `PIPELINE_CSV` in place with `retrieved_contexts` and `retrieved_file`.
  - Optional `retriever_run_summary.json` on errors.

### `2_alt_retriever_agent.py`
- Purpose: Alternate retrieval implementation using Bedrock Agent invocation instead of direct KB retrieval.
- Input:
  - Reads the same `PIPELINE_CSV` from step 1.
- Main flow:
  - Calls `invoke_agent` per query with session IDs and optional tracing.
  - Parses citations from response completion events.
  - De-duplicates references while building `retrieved_contexts` and `retrieved_file`.
- Outputs:
  - Same shape as step 2 (updates `PIPELINE_CSV`) so downstream stages remain compatible.
  - Optional error summary file.

### `3_relevance_eval.py`
- Purpose: Computes cross-encoder-style relevance scores between query and each retrieved chunk.
- Input:
  - Requires `PIPELINE_CSV` with `user_input` and `retrieved_contexts`.
- Main flow:
  - Loads list-like stringified columns from CSV.
  - Builds query-context pairs.
  - Uses `FlagEmbedding.FlagReranker` (`MODEL_NAME`) to score each pair in adaptive batches.
  - Normalizes scores to `[0,1]` and restores them per row.
- Output:
  - Adds `relevance_scores` to `PIPELINE_CSV`.

### `4_evaluator.py`
- Purpose: Produces final quality metrics from retrieval outputs.
- Input:
  - Requires `PIPELINE_CSV` plus columns from step 2/2_alt and optional `relevance_scores`.
- Main flow:
  - Parses list columns safely.
  - Computes:
    - hit rate (`custom_hit_rate`)
    - mean reciprocal rank (`custom_mrr`)
    - precision@k / recall@k
    - reranker-based precision@k (`precision_at_k_relevance`, threshold-based)
- Output:
  - Saves enriched results to `PIPELINE_OUTPUT_DIR`:
    - `*_results.csv`
    - `*_results.parquet`
  - Also copies parquet to `streamlit/complete_datasets` for dashboard use.

### `config.py`
- Centralizes environment-based configuration used across scripts:
  - KB locations and output directories
  - AWS service/profile/region/IDs
- Also stores retrieval, evaluation, and retry hyperparameters.

### `requirements.txt`
- Declares runtime dependencies, including Bedrock/client libs, pandas/Arrow/parquet, reranker model tooling, and Streamlit/Altair for reporting.

---

## Folders with grouped responsibilities

### `retriever/`
- Contains manual/sanity-check scripts for raw Bedrock calls:
  - `kb_raw_retriever.py`: one-off KB `retrieve` call capture.
  - `agent_raw_retriever.py`: one-off Agent invocation capture with streaming completion materialization.
- Also stores example raw responses (`*.json`) for debugging.
- Role: low-level API exploration and debugging, independent from the main pipeline CSV flow.

### `aws_tokenizer/`
- Small utility scripts for token counting and embedding checks against Bedrock models.
- `token_count_all_md.py` is a batch utility for token counts across an `.md` corpus.
- Useful for corpus sanity checks and preprocessing cost estimation.

### `streamlit/`
- Visualization and review app for evaluation runs.
- `app.py`:
  - Loads parquet datasets from `streamlit/complete_datasets`.
  - Shows global and per-style metrics, dataset compare, and case-level drill-down.
- `metrics.json`:
  - Human-readable metric descriptions used for in-app help/tooltips.

### `outputs/`
- Stores materialized run artifacts by experiment:
  - pipeline states
  - retrieval progress and parse-fail logs
  - final CSV/Parquet result sets
- Naming indicates scenario (`small_test`, `full_test`, `full_generation_no_retrieval`, etc.).

### `kb_small_testfolder/`
- Example KB input dataset (Markdown source documents) used for quick local runs.

## Dataflow diagram (conceptual)

`kb_small_testfolder/*.md`  
→ `1_generate_user_inputs.py` (`PIPELINE_CSV` with synthetic inputs)  
→ (`2_retriever.py` OR `2_alt_retriever_agent.py`) (`retrieved_contexts`, `retrieved_file`)  
→ `3_relevance_eval.py` (optional `relevance_scores`)  
→ `4_evaluator.py` (`*_results.csv`, `*_results.parquet`)  
→ `streamlit/app.py` (optional dashboard)

This order can be repeated with different `CONFIG` targets (different `PIPELINE_OUTPUT_DIR`, KB IDs, top-k, and environment profiles) to support multiple experiments side-by-side in `outputs/`.
