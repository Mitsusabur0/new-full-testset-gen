## Interpretación de los resultados

| Métrica | Cálculo | Interpretación |
|---|---|---|
| hit rate | Se asigna 1 si el `source_file` coincide con algún documento recuperado; 0 en caso contrario. | 0.9397 indica que el 93.97 % de las consultas recuperaron el archivo fuente esperado, mostrando alta capacidad de recuperación. |
| mrr | Si hay hit, se calcula 1/rank del primer match; 0 si no hay hit. | 0.8882 refleja que, en promedio, el primer hit aparece cerca del top de la lista, indicando buen ordenamiento. |
| precision@k | Ratio de contextos con `relevance_score` ≥ 0.5 entre k, usando el modelo de similitud `user_input` ↔ contexto y umbral 0.5. | 0.4773 muestra que menos de la mitad de los k documentos recuperados son relevantes según el modelo, señalando espacio de mejora; sin `relevance_score`, el máximo teórico sería 1/3. |
| recall@k | Se reporta igual que hit rate en este pipeline. | 0.9397 (igual a hit rate) porque solo se evalúa la presencia del único `source_file`; recall@k = hit rate, dado que los casos sintéticos usan un solo archivo fuente por consulta y solo se garantiza si ese archivo fue recuperado. |