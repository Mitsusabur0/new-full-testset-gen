## Interpretación de los resultados

| Métrica | Cálculo | Interpretación |
|---|---|---|
| hit rate | 1 si el `source_file` coincide con alguna ruta recuperada, 0 si no. | 0.9357 indica que el 93.6 % de las consultas recuperan el archivo fuente esperado, señal de alta cobertura. |
| mrr | 1/rank del primer `source_file` coincidente; 0 si no hay coincidencia. | 0.8809 refleja que, en promedio, el documento relevante aparece muy alto en la lista (≈ posición 1.14). |
| precision@k | Ratio de contextos con `relevance_score` ≥ 0.5 entre k, usando modelo de similitud `user_input`‑`context`. | 0.4985 muestra que ~50 % de los k documentos son realmente relevantes; usamos umbral 0.5 para medir relevancia, lo que evita la limitación teórica de 1/3 si solo se considerara coincidencia de `source_file`. |
| recall@k | Igual que hit rate en este pipeline. | 0.9357 (recall@k = hit rate, dado que los casos sintéticos