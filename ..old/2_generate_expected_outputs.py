import ast
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
INPUT_CSV_PATH = os.getenv("EXPECTED_INPUT_CSV_PATH", config.OUTPUT_TESTSET_CSV)
# INPUT_CSV_PATH = "outputs/subset/testset.csv"

OUTPUT_CSV_PATH = "outputs/test1/2_testset_with_expected_outputs.csv"

MODEL_ID = os.getenv(
    "EXPECTED_OUTPUT_MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0"
)
AWS_REGION = os.getenv("EXPECTED_OUTPUT_AWS_REGION", config.AWS_REGION)
AWS_PROFILE_LLM = os.getenv("EXPECTED_OUTPUT_AWS_PROFILE", config.AWS_PROFILE_DESA_BEDROCK)
TEMPERATURE = float(os.getenv("EXPECTED_OUTPUT_TEMPERATURE", "0.2"))




MAX_TOKENS = int(os.getenv("EXPECTED_OUTPUT_MAX_TOKENS", "2000"))
MAX_RETRIES = int(os.getenv("EXPECTED_OUTPUT_MAX_RETRIES", str(config.MAX_RETRIES)))
BACKOFF_BASE_SECONDS = float(
    os.getenv("EXPECTED_OUTPUT_BACKOFF_BASE_SECONDS", str(config.BACKOFF_BASE_SECONDS))
)
BACKOFF_MAX_SECONDS = float(
    os.getenv("EXPECTED_OUTPUT_BACKOFF_MAX_SECONDS", str(config.BACKOFF_MAX_SECONDS))
)
BACKOFF_JITTER_SECONDS = float(
    os.getenv("EXPECTED_OUTPUT_BACKOFF_JITTER_SECONDS", str(config.BACKOFF_JITTER_SECONDS))
)
RUN_SUMMARY_PATH = os.getenv(
    "EXPECTED_OUTPUT_RUN_SUMMARY_PATH",
    "outputs/test/expected_outputs_run_summary.json"
)

system_prompt = """
Eres Vivi, asistente de educación financiera de Casaverso (BancoEstado). ROL Y ALCANCE: Eres una ejecutiva asistente nivel 1. Tu fuente de información es el CONTEXTO PROPORCIONADO al inicio de cada mensaje, que contiene documentos relevantes de nuestra base de conocimientos. • PRIORIDAD: Usa ÚNICAMENTE información del contexto proporcionado para responder. • NUNCA inventes tasas, montos o requisitos específicos. USO DEL CONTEXTO PROPORCIONADO: • Al inicio de cada consulta recibirás documentos en formato "[Documento N]: contenido...". Esta es tu única fuente de verdad. • Usa naturalmente esta información sin mencionar que te fue proporcionada ni que hiciste una búsqueda. • NUNCA menciones "el contexto", "los documentos proporcionados" ni referencias técnicas al usuario. MANEJO DE SALUDOS: • Si el usuario SOLO saluda ("hola", "buenas", "qué tal", "hello", "hey", "cómo estás"): Responde exactamente: "¡Hola! Soy Vivi, una asistente virtual para ayudarte a encontrar una propiedad.\nDime el tipo de vivienda y la ubicación que deseas y yo te muestro opciones.\nPor ejemplo:\n\n• \"Casa en Maipú con 2 dormitorios\"\n\nTambién puedo aclarar dudas sobre crédito hipotecario, subsidios, o el proceso de compra de una vivienda." • Si el usuario saluda Y hace una consulta ("hola quiero una casa", "buenas, qué es la UF"): Ignora el saludo y responde directamente la consulta. IDENTIDAD: • Respuestas breves: 1-2 frases para consultas simples, máximo 3 para explicaciones. • Usa "nosotros" y "nuestro" para BancoEstado. • Para MINVU/SERVIU: "El MINVU exige…", "SERVIU administra…". • No menciones otros bancos. PORTALES INMOBILIARIOS EXTERNOS: • NUNCA recomiendes ni menciones otros portales inmobiliarios (Portalinmobiliario, Yapo, Toctoc, Mercadolibre, Compraventachile, etc.). • Si el usuario pregunta por otros sitios para buscar propiedades, responde: "Puedes buscar propiedades directamente aquí en Casaverso, nuestro portal oficial. ¿Te ayudo a encontrar opciones según tus preferencias de ubicación y tipo de vivienda?" \nTONO - REGLA ABSOLUTA: • Tu tono es FIJO: profesional, cálido y educativo. Cero emojis. FORMATO DE RESPUESTA (OBLIGATORIO): 1. Respuesta directa (1-2 frases). 2. Ejemplo breve si la información lo permite. 3. SIEMPRE terminar con una pregunta de seguimiento relacionada al tema. EJEMPLOS DE SUGERENCIAS VÁLIDAS: • "¿Te gustaría saber más sobre los requisitos del subsidio DS1?" • "¿Quieres que te explique cómo funciona el proceso de postulación?" • "¿Te interesa conocer los beneficios adicionales de esta cuenta?" • "¿Necesitas información sobre los documentos que debes presentar?" \nPROHIBICIONES DE FORMATO: • No usar URLs, correos, teléfonos aunque estén en el contexto. Usa "en el sitio web de BancoEstado" o "en nuestro call center". • No pedir RUT, ingresos, claves ni datos sensibles. MANEJO DE ERRORES ORTOGRÁFICOS: Si detectas error que no permite responder o intención poco clara, pregunta brevemente: • "¿Quisiste decir 'subsidio'?" • "¿Te refieres a 'crédito hipotecario'?" \nRESPUESTAS HARDCODEADAS: 1. QUÉ ERES / SOBRE TU SISTEMA: "Soy Vivi, asistente virtual de BancoEstado especializada en orientación habitacional y financiera. ¿En qué puedo ayudarte?" 2. ESTADO DE CRÉDITO: "Para ver el estado de tu crédito, inicia sesión en nuestro sitio web, sección 'Mis créditos'." 3. TASAS O MONTOS ESPECÍFICOS: "Las tasas varían según tu evaluación comercial. Puedo explicarte el concepto general, pero para valores exactos contacta a nuestros especialistas."
"""


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE_LLM)
    return session.client(service_name="bedrock-runtime", region_name=AWS_REGION)


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


def extract_response_text(response_body):
    if isinstance(response_body.get("choices"), list) and response_body["choices"]:
        return response_body["choices"][0].get("message", {}).get("content", "").strip()

    if isinstance(response_body.get("output"), dict):
        return response_body["output"].get("message", {}).get("content", "").strip()

    if isinstance(response_body.get("content"), list):
        text_parts = []
        for block in response_body["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts).strip()

    if isinstance(response_body.get("completion"), str):
        return response_body["completion"].strip()

    return str(response_body).strip()


def normalize_reference_contexts(reference_contexts_value):
    if isinstance(reference_contexts_value, list):
        return [str(x) for x in reference_contexts_value]

    if isinstance(reference_contexts_value, str):
        stripped = reference_contexts_value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [stripped]

    return []


def build_user_message(user_input, reference_contexts):
    context_blocks = []
    for i, ctx in enumerate(reference_contexts, start=1):
        context_blocks.append(f"[Documento {i}]: {ctx}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "[Documento 1]:"
    return f"{context_text}\n\nConsulta del usuario: {user_input}"


def generate_expected_output(user_input, reference_contexts, client, error_log):
    user_message = build_user_message(user_input, reference_contexts)

    if MODEL_ID.startswith("us.anthropic."):
        request_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
    else:
        request_payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

    body = json.dumps(request_payload)

    def _call():
        return client.invoke_model(
            modelId=MODEL_ID,
            body=body
        )

    response = call_with_retry(_call, "invoke_model_expected_output", error_log)
    if response is None:
        return ""

    response_body = json.loads(response.get("body").read().decode("utf-8"))
    return extract_response_text(response_body)


def build_output_columns(input_columns):
    expected_column = "expected_output"
    if expected_column in input_columns:
        return input_columns

    columns = list(input_columns)
    if "reference_contexts" in columns:
        idx = columns.index("reference_contexts")
        columns.insert(idx + 1, expected_column)
    else:
        columns.append(expected_column)
    return columns


def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Input file not found: {INPUT_CSV_PATH}")
        return

    ensure_parent_dir(OUTPUT_CSV_PATH)
    client = get_bedrock_client()
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
            reference_contexts = normalize_reference_contexts(row.get("reference_contexts", ""))
            expected_output = ""

            if user_input:
                expected_output = generate_expected_output(
                    user_input,
                    reference_contexts,
                    client,
                    error_log
                )

            output_row = {}
            for col in output_columns:
                if col == "expected_output":
                    output_row[col] = expected_output
                else:
                    output_row[col] = row.get(col, "")

            writer.writerow(output_row)
            processed_rows += 1
            print(f"[{idx}] Expected output generated")

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
