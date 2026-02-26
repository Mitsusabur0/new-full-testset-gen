import os
import json
import random
import glob
import re
import time
import unicodedata
from datetime import datetime
import pandas as pd
import boto3
from botocore.exceptions import ClientError

import config

NO_GENERATION_SENTINEL = "No se puede generar con este estilo"

QUERY_STYLES = [
    {
        "style_name": "Buscador de Palabras Clave",
        "description": "El usuario no redacta una oracion completa. Escribe fragmentos sueltos, como si estuviera buscando en Google. Ejemplo: 'requisitos pie', 'seguro desgravamen edad', 'renta minima postulacion'."
    },
    {
        "style_name": "Caso Hipotetico en Primera Persona",
        "description": "El usuario plantea una situacion personal (real o inventada) que incluye cifras o condiciones especificas para ver si el texto se aplica a el. Usa estructuras como 'Si yo tengo...', 'En caso de que gane...', 'Que pasa si...?'."
    },
    {
        "style_name": "Duda Directa sobre Restricciones",
        "description": "El usuario busca la letra chica, los limites o los impedimentos. Pregunta especificamente por lo que NO se puede hacer, los castigos, o los maximos/minimos. Tono serio y pragmatico."
    },
    {
        "style_name": "Colloquial Chileno Natural",
        "description": "Redaccion relajada, usando modismos locales suaves y un tono de conversacion por chat (WhatsApp). Usa terminos como 'depa', 'lucas', 'chao', 'consulta', 'al tiro'. Trata al asistente con cercania."
    },
    {
        "style_name": "Modismos Chilenos Muy Relajado",
        "description": "Redaccion muy relajada, usando marcados modismos locales."
    },
    {
        "style_name": "Principiante / Educativo",
        "description": "El usuario admite no saber del tema y pide definiciones o explicaciones de conceptos basicos mencionados en el texto. Pregunta 'Que significa...?', 'Como funciona...?', 'Explicame eso de...'."
    },
    {
        "style_name": "Orientado a la Accion",
        "description": "El usuario quiere saber el como operativo. Pregunta por pasos a seguir, documentos a llevar, lugares donde ir o botones que apretar. Ejemplo: 'Donde mando los papeles?', 'Como activo esto?', 'Con quien hablo?'."
    },
    {
        "style_name": "Mal redactado / Errores ortograficos",
        "description": "El usuario escribe de forma informal, con errores ortograficos o mal redactado. Puede usar abreviaturas, faltas de puntuacion o estructura incoherente. Ejemplo: 'kiero saver los reqs pa el subsidio'."
    },
    {
        "style_name": "Errores ortograficos",
        "description": "El usuario escribe con errores ortograficos en palabras clave. Además, usa abreviaturas, faltas de puntuacion o estructura incoherente'."
    },
]


def extract_bd_code(filename):
    if not filename:
        return ""
    return filename[:9]


def get_bedrock_client():
    session = boto3.Session(profile_name=config.AWS_PROFILE_LLM)
    return session.client(service_name="bedrock-runtime", region_name=config.AWS_REGION)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_row_to_csv(path, row):
    ensure_parent_dir(path)
    file_exists = os.path.exists(path)
    has_content = file_exists and os.path.getsize(path) > 0
    pd.DataFrame([row]).to_csv(
        path,
        mode="a" if has_content else "w",
        header=not has_content,
        index=False,
    )


def append_progress(progress_log_path, file_path, style_name, status):
    ensure_parent_dir(progress_log_path)
    with open(progress_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_path": file_path,
            "style_name": style_name,
            "status": status
        }, ensure_ascii=False) + "\n")


def load_processed_pairs(progress_log_path):
    processed = set()
    if not os.path.exists(progress_log_path):
        return processed

    with open(progress_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                file_path = obj.get("file_path")
                style_name = obj.get("style_name")
                status = obj.get("status")
                if file_path and style_name and status in {"generated", "skipped_no_generation"}:
                    processed.add((file_path, style_name))
            except json.JSONDecodeError:
                continue
    return processed


def backoff_sleep(attempt):
    base = config.BACKOFF_BASE_SECONDS * (2 ** attempt)
    sleep_for = min(base, config.BACKOFF_MAX_SECONDS)
    sleep_for += random.uniform(0, config.BACKOFF_JITTER_SECONDS)
    time.sleep(sleep_for)


def call_with_retry(fn, operation_name, error_log):
    last_error = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return fn()
        except ClientError as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < config.MAX_RETRIES:
            backoff_sleep(attempt)
        else:
            if last_error is not None:
                error_log.append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "operation": operation_name,
                    "error": str(last_error),
                })
            return None


def clean_llm_output(text):
    cleaned_text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def normalize_style_name(style_name):
    if not style_name:
        return ""
    normalized = unicodedata.normalize("NFKD", style_name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def parse_llm_xml(content, allowed_styles):
    content_no_reasoning = clean_llm_output(content)

    style_match = re.search(r"<style_name>(.*?)</style_name>", content_no_reasoning, re.DOTALL | re.IGNORECASE)
    input_match = re.search(r"<user_input>(.*?)</user_input>", content_no_reasoning, re.DOTALL | re.IGNORECASE)

    if not (style_match and input_match):
        return None, None, content_no_reasoning

    style_found = style_match.group(1).strip()
    question_text = input_match.group(1).strip()
    question_text = question_text.replace('"', "").replace("\n", " ").strip()

    if not question_text:
        return None, None, content_no_reasoning

    if allowed_styles:
        style_found_norm = normalize_style_name(style_found)
        matched_allowed_style = None
        for allowed_style in allowed_styles:
            if normalize_style_name(allowed_style) == style_found_norm:
                matched_allowed_style = allowed_style
                break
        if not matched_allowed_style:
            return None, None, content_no_reasoning
        style_found = matched_allowed_style

    return question_text, style_found, content_no_reasoning


def repair_xml_response(raw_content, allowed_styles, client, error_log):
    allowed_str = ", ".join(allowed_styles) if allowed_styles else ""
    repair_prompt = f"""
Extrae la consulta del usuario y el estilo desde el siguiente texto y devuelve SOLO el formato XML requerido.

Texto original:
{raw_content}

Estilos permitidos: {allowed_str}

Formato requerido (sin markdown, sin texto adicional):
<style_name>NOMBRE_DEL_ESTILO</style_name>
<user_input>CONSULTA_DEL_USUARIO</user_input>
"""

    body = json.dumps({
        "messages": [{"role": "user", "content": repair_prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    })

    def _call():
        return client.invoke_model(
            modelId=config.MODEL_ID,
            body=body
        )

    response = call_with_retry(_call, "invoke_model_repair", error_log)
    if response is None:
        return None, None, None

    response_body = json.loads(response.get("body").read().decode("utf-8"))
    if "choices" in response_body:
        content = response_body["choices"][0]["message"]["content"]
    elif "output" in response_body:
        content = response_body["output"]["message"]["content"]
    else:
        content = str(response_body)

    return parse_llm_xml(content, allowed_styles)


def generate_question_for_style(chunk_text, query_style, client, error_log, parse_fail_log_path):
    style_name = query_style["style_name"]
    style_description = query_style["description"]
    allowed_styles = [style_name]

    system_prompt = f"""
### ROL DEL SISTEMA
Eres un Generador de Datos Sinteticos especializado en Banca y Bienes Raices de Chile.
Tu trabajo es crear el Test Set para evaluar un asistente de IA (RAG) del Banco Estado (Casaverso).

### TAREA PRINCIPAL
Se te entregara un fragmento de texto (La Respuesta).
Tu objetivo es redactar la Consulta del Usuario (El Input) que provocaria que el sistema recupere este texto como respuesta.

### REGLAS DE ORO (CRITICO: LEER CON ATENCION)
1. ASIMETRIA DE INFORMACION: El usuario NO ha leido el texto. No sabe los terminos tecnicos exactos, ni los porcentajes, ni los articulos de la ley que aparecen en el texto.
2. INTENCION vs CONTENIDO:
- MAL (Contaminado): Cuales son los requisitos del articulo 5 del subsidio DS19?
- BIEN (Realista): Oye, que papeles me piden para postular al subsidio?
3. ABSTRACCION: Si el texto habla de Tasa fija del 4.5%, el usuario NO pregunta Es la tasa del 4.5%?. El usuario pregunta Como estan las tasas hoy?.
4. SI EL TEXTO ES CORTO/PARCIAL: Si el fragmento es muy especifico o tecnico, el usuario debe hacer una pregunta mas amplia o vaga que este fragmento responderia parcialmente.
5. CONTEXTO CHILENO: Usa vocabulario local, modismos y el tono correspondiente al estilo solicitado.

### DOCUMENTO DE REFERENCIA:
Se te entregara un fragmento de texto que el asistente deberia recuperar como respuesta a la consulta del usuario.

### ESTILO DE CONSULTA OBLIGATORIO:
Debes redactar la consulta usando EXCLUSIVAMENTE el siguiente estilo:
- Nombre: {style_name}
- Descripcion: {style_description}

Si ese estilo no aplica al documento o no es posible crear una consulta realista para ese documento con ese estilo, debes responder dentro del tag <user_input> exactamente:
{NO_GENERATION_SENTINEL}

No puedes usar ninguna otra frase alternativa para ese caso.

### FORMATO DE SALIDA
Tu respuesta sera en dos tags xml: <style_name> y <user_input>.
El texto dentro de style_name debe ser EXACTAMENTE: {style_name}
El texto dentro del <user_input> debe ser la consulta generada, sin comillas, sin saltos de linea, sin explicaciones adicionales.
Responde UNICAMENTE con este formato XML (sin markdown, sin explicaciones):

<style_name>NOMBRE_DEL_ESTILO</style_name>
<user_input>TU_CONSULTA_GENERADA_AQUI</user_input>
"""

    prompt = f"""
### DOCUMENTO DE REFERENCIA:
{chunk_text}

### ESTILO DE CONSULTA OBLIGATORIO:
Nombre: {style_name}
Descripcion: {style_description}
"""

    body = json.dumps({
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": config.TEMPERATURE,
        "max_tokens": 2000
    })

    def _call():
        return client.invoke_model(
            modelId=config.MODEL_ID,
            body=body
        )

    response = call_with_retry(_call, "invoke_model", error_log)
    if response is None:
        return None, None

    response_body = json.loads(response.get("body").read().decode("utf-8"))

    if "choices" in response_body:
        content = response_body["choices"][0]["message"]["content"]
    elif "output" in response_body:
        content = response_body["output"]["message"]["content"]
    else:
        content = str(response_body)

    question_text, style_found, cleaned = parse_llm_xml(content, allowed_styles)
    if question_text and style_found:
        return question_text, style_found

    repair_q, repair_style, _ = repair_xml_response(content, allowed_styles, client, error_log)
    if repair_q and repair_style:
        return repair_q, repair_style

    ensure_parent_dir(parse_fail_log_path)
    with open(parse_fail_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reason": "parse_failed",
            "allowed_styles": allowed_styles,
            "raw_response": cleaned,
        }, ensure_ascii=False) + "\n")

    print(f"Warning: Could not parse XML from LLM response: {cleaned[:100]}...")
    return None, None


def main():
    random.seed(config.SEED)
    print(f"Using seed: {config.SEED}")
    print(f"Scanning for files in {config.KB_FOLDER}...")

    if not os.path.exists(config.KB_FOLDER):
        print(f"Error: Directory {config.KB_FOLDER} does not exist.")
        return

    search_pattern = os.path.join(config.KB_FOLDER, "**", "*.md")
    files = sorted(glob.glob(search_pattern, recursive=True))

    if not files:
        print(f"No .md files found in {config.KB_FOLDER} or its subdirectories.")
        return

    print(f"Found {len(files)} Markdown files.")

    client = get_bedrock_client()
    error_log = []
    parse_failures = 0
    generated_count = 0
    skipped_by_style_count = 0
    parse_fail_log_path = os.path.join(
        os.path.dirname(config.PIPELINE_CSV),
        "parse_failures.jsonl"
    )
    progress_log_path = os.path.join(
        os.path.dirname(config.PIPELINE_CSV),
        "generation_progress.jsonl"
    )

    processed_pairs = load_processed_pairs(progress_log_path)
    print(f"Resuming with {len(processed_pairs)} completed file/style pairs.")
    print("Generating synthetic questions...")

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunk_text = f.read()

            if len(chunk_text) < 30:
                continue

            print(f"[{i + 1}/{len(files)}] Processing {os.path.basename(file_path)}")
            for style_idx, style in enumerate(QUERY_STYLES, start=1):
                style_name = style["style_name"]
                pair_key = (file_path, style_name)
                if pair_key in processed_pairs:
                    continue

                print(f"  - Style [{style_idx}/{len(QUERY_STYLES)}]: {style_name}")
                generated_question, style_used = generate_question_for_style(
                    chunk_text,
                    style,
                    client,
                    error_log,
                    parse_fail_log_path
                )

                if not (generated_question and style_used):
                    parse_failures += 1
                    continue

                if generated_question == NO_GENERATION_SENTINEL:
                    skipped_by_style_count += 1
                    append_progress(progress_log_path, file_path, style_name, "skipped_no_generation")
                    processed_pairs.add(pair_key)
                    continue

                row = {
                    "user_input": generated_question,
                    "reference_contexts": [chunk_text],
                    "query_style": style_used,
                    "source_file": extract_bd_code(os.path.basename(file_path))
                }
                append_row_to_csv(config.PIPELINE_CSV, row)
                generated_count += 1
                append_progress(progress_log_path, file_path, style_name, "generated")
                processed_pairs.add(pair_key)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    if generated_count > 0:
        print(f"Successfully generated {generated_count} test cases. Saved incrementally to {config.PIPELINE_CSV}")
    else:
        print("No data generated.")

    if error_log or parse_failures or skipped_by_style_count:
        print(
            "Non-fatal errors: "
            f"{len(error_log)} | Parse failures: {parse_failures} | "
            f"Skipped by style mismatch: {skipped_by_style_count}"
        )
        summary_path = os.path.join(
            os.path.dirname(config.PIPELINE_CSV),
            "run_summary.json"
        )
        ensure_parent_dir(summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump({
                "generated": generated_count,
                "parse_failures": parse_failures,
                "skipped_by_style_mismatch": skipped_by_style_count,
                "errors": error_log,
            }, summary_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
