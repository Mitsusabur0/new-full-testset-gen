## Interpretación de los resultados

| Métrica | Cálculo | Interpretación |
|---|---|---|
| hit rate | Proporción de consultas con al menos un source_file coincidente (1 si hay coincidencia, 0 si no). | 0.936 indica que el 93.6 % de las consultas recuperan el archivo fuente esperado, mostrando alta cobertura. |
| mrr | 1/rank del primer hit cuando existe, 0 si no hay hit. | 0.881 refleja que, en promedio, el primer documento relevante aparece muy alto en la lista (≈1.13 posición). |
| precision@k | Ratio de contextos con relevance_score ≥ 0.5 entre k (umbral 0.5); respaldo 1/3 si no hay scores válidos. | 0.378 muestra que ~37.8 % de los k documentos recuperados son realmente relevantes según el modelo, evidenciando espacio de mejora. |
| recall@k | Igual que hit rate en este pipeline. | 0.936 confirma la misma cobertura que hit rate; recall@k = hit rate, dado que los casos sintéticos usan un solo archivo fuente por consulta y solo se garantiza si ese archivo fue recuperado. |