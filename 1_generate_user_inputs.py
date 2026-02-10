import os
import json
import random
import glob
import re
import time
from datetime import datetime
import pandas as pd
import boto3
from botocore.exceptions import ClientError

import config

# --- CHILEAN BANKING CONTEXT CONFIGURATION ---

QUERY_STYLES = [
    {
        "style_name": "Buscador de Palabras Clave",
        "description": "El usuario no redacta una oración completa. Escribe fragmentos sueltos, como si estuviera buscando en Google. Ejemplo: 'requisitos pie', 'seguro desgravamen edad', 'renta minima postulacion'."
    },
    {
        "style_name": "Caso Hipotético en Primera Persona",
        "description": "El usuario plantea una situación personal (real o inventada) que incluye cifras o condiciones específicas para ver si el texto se aplica a él. Usa estructuras como 'Si yo tengo...', 'En caso de que gane...', '¿Qué pasa si...?'."
    },
    {
        "style_name": "Duda Directa sobre Restricciones",
        "description": "El usuario busca la 'letra chica', los límites o los impedimentos. Pregunta específicamente por lo que NO se puede hacer, los castigos, o los máximos/mínimos. Tono serio y pragmático."
    },
    {
        "style_name": "Colloquial Chileno Natural",
        "description": "Redacción relajada, usando modismos locales suaves y un tono de conversación por chat (WhatsApp). Usa términos como 'depa', 'lucas', 'chao', 'consulta', 'al tiro'. Trata al asistente con cercanía."
    },
    {
        "style_name": "Principiante / Educativo",
        "description": "El usuario admite no saber del tema y pide definiciones o explicaciones de conceptos básicos mencionados en el texto. Pregunta '¿Qué significa...?', '¿Cómo funciona...?', 'Explícame eso de...'."
    },
    {
        "style_name": "Orientado a la Acción",
        "description": "El usuario quiere saber el 'cómo' operativo. Pregunta por pasos a seguir, documentos a llevar, lugares dónde ir o botones que apretar. Ejemplo: '¿Dónde mando los papeles?', '¿Cómo activo esto?', '¿Con quién hablo?'."
    },
    {
        "style_name": "Mal redactado / Errores ortográficos",
        "description": "El usuario escribe de forma informal, con errores ortográficos o mal redactado. Puede usar abreviaturas, faltas de puntuación o estructura incoherente. Ejemplo: 'kiero saver los reqs pa el subsidio' "
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
    """Removes <reasoning> tags and their content from the LLM output."""
    # The flag re.DOTALL makes '.' match newlines as well
    cleaned_text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def parse_llm_xml(content, allowed_styles):
    content_no_reasoning = clean_llm_output(content)

    style_match = re.search(r'<style_name>(.*?)</style_name>', content_no_reasoning, re.DOTALL | re.IGNORECASE)
    input_match = re.search(r'<user_input>(.*?)</user_input>', content_no_reasoning, re.DOTALL | re.IGNORECASE)

    if not (style_match and input_match):
        return None, None, content_no_reasoning

    style_found = style_match.group(1).strip()
    question_text = input_match.group(1).strip()
    question_text = question_text.replace('"', '').replace('\n', ' ').strip()

    if not question_text:
        return None, None, content_no_reasoning

    if allowed_styles and style_found not in allowed_styles:
        return None, None, content_no_reasoning

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

    response_body = json.loads(response.get('body').read().decode('utf-8'))
    if 'choices' in response_body:
        content = response_body['choices'][0]['message']['content']
    elif 'output' in response_body:
        content = response_body['output']['message']['content']
    else:
        content = str(response_body)

    return parse_llm_xml(content, allowed_styles)

def generate_question_only(chunk_text, query_styles, client, error_log, parse_fail_log_path):
    allowed_styles = [style["style_name"] for style in query_styles]
    
    system_prompt = f"""
### ROL DEL SISTEMA
Eres un Generador de Datos Sintéticos especializado en Banca y Bienes Raíces de Chile.
Tu trabajo es crear el "Test Set" para evaluar un asistente de IA (RAG) del Banco Estado (Casaverso).

### TAREA PRINCIPAL
Se te entregará un fragmento de texto (La "Respuesta").
Tu objetivo es redactar la **Consulta del Usuario** (El "Input") que provocaría que el sistema recupere este texto como respuesta.

### REGLAS DE ORO (CRÍTICO: LEER CON ATENCIÓN)
1. **ASIMETRÍA DE INFORMACIÓN:** El usuario NO ha leído el texto. No sabe los términos técnicos exactos, ni los porcentajes, ni los artículos de la ley que aparecen en el texto.
2. **INTENCIÓN vs CONTENIDO:**
- MAL (Contaminado): "¿Cuáles son los requisitos del artículo 5 del subsidio DS19?" (El usuario no sabe que existe el artículo 5).
- BIEN (Realista): "Oye, ¿qué papeles me piden para postular al subsidio?"
3. **ABSTRACCIÓN:** Si el texto habla de "Tasa fija del 4.5%", el usuario NO pregunta "¿Es la tasa del 4.5%?". El usuario pregunta "¿Cómo están las tasas hoy?".
4. **SI EL TEXTO ES CORTO/PARCIAL:** Si el fragmento es muy específico o técnico, el usuario debe hacer una pregunta más amplia o vaga que este fragmento respondería parcialmente.
5. **CONTEXTO CHILENO:** Usa vocabulario local, modismos y el tono correspondiente al estilo solicitado.


### DOCUMENTO DE REFERENCIA:
Se te etregará un fragmento de texto que el asistente debería recuperar como respuesta a la consulta del usuario.

### ESTILOS DE CONSULTA DISPONIBLES:
Se te entregará una lista de 2 estilos. 
Debes seleccionar uno de los estilos para redactar la pregunta. 
Si ambos estilos sirven para el fragmento, selecciona al azar, no siempre el más simple. Si sólo uno sirve, úsalo. Si ninguno sirve, elige el que mejor se adapte con modificaciones.
Luego, debes redactar la consulta adoptando el estilo seleccionado.

### FORMATO DE SALIDA
Tu respuesta serán dos tags xml: <style_name> y <user_input>.
El texto dentro de style_name es el nombre del estilo seleccionado. Debes mantener el MISMO style_name entregado.
El texto dentro del <user_input> debe ser la consulta generada, sin comillas, sin saltos de línea, sin explicaciones adicionales. Es el texto plano de la consulta del usuario.
Responde ÚNICAMENTE con este formato XML (sin markdown, sin explicaciones):

<style_name>NOMBRE_DEL_ESTILO</style_name>
<user_input>TU_CONSULTA_GENERADA_AQUI</user_input>
"""

    prompt = f"""
### DOCUMENTO DE REFERENCIA:
{chunk_text}    
### ESTILOS DE CONSULTA DISPONIBLES:
{query_styles}
"""

    # Payload structure
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

    response_body = json.loads(response.get('body').read().decode('utf-8'))

    if 'choices' in response_body:
        content = response_body['choices'][0]['message']['content']
    elif 'output' in response_body:
        content = response_body['output']['message']['content']
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

    # Recursive glob search for .md files
    search_pattern = os.path.join(config.KB_FOLDER, "**", "*.md")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No .md files found in {config.KB_FOLDER} or its subdirectories.")
        return

    print(f"Found {len(files)} Markdown files.")

    client = get_bedrock_client()
    dataset = []
    error_log = []
    parse_failures = 0
    parse_fail_log_path = os.path.join(
        os.path.dirname(config.OUTPUT_TESTSET_CSV),
        "parse_failures.jsonl"
    )

    print("Generating synthetic questions...")

    for i, file_path in enumerate(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_text = f.read()
                
                # Skip empty files
                if len(chunk_text) < 30: 
                    continue
                
                # --- PROGRAMMATIC SELECTION ---
                # Select 3 random styles from the global list
                selected_styles = random.sample(QUERY_STYLES, 2)
                
                print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)}")
                
                # --- CALL LLM FOR USER INPUT ONLY ---
                # Now expects a tuple return (question, style_used)
                generated_question, style_used = generate_question_only(
                    chunk_text,
                    selected_styles,
                    client,
                    error_log,
                    parse_fail_log_path
                )
                
                if generated_question and style_used:
                    # --- CONSTRUCT ROW PROGRAMMATICALLY ---
                    row = {
                        "user_input": generated_question,
                        "reference_contexts": [chunk_text], 
                        "query_style": style_used,
                        "source_file": extract_bd_code(os.path.basename(file_path))
                    }
                    dataset.append(row)
                else:
                    parse_failures += 1
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    if dataset:
        df = pd.DataFrame(dataset)
        ensure_parent_dir(config.OUTPUT_TESTSET_CSV)
        df.to_csv(config.OUTPUT_TESTSET_CSV, index=False)
        print(f"Successfully generated {len(df)} test cases. Saved to {config.OUTPUT_TESTSET_CSV}")
    else:
        print("No data generated.")

    if error_log or parse_failures:
        print(f"Non-fatal errors: {len(error_log)} | Parse failures: {parse_failures}")
        summary_path = os.path.join(
            os.path.dirname(config.OUTPUT_TESTSET_CSV),
            "run_summary.json"
        )
        ensure_parent_dir(summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump({
                "generated": len(dataset),
                "parse_failures": parse_failures,
                "errors": error_log,
            }, summary_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
