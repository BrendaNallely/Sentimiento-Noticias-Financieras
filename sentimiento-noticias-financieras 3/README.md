# Análisis de sentimiento en noticias financieras

Comparación de FinBERT, FinBERT Tone y DistilRoBERTa para clasificar noticias financieras en inglés como positivas, negativas o neutrales.

## Entorno de ejecución
Google Colab con GPU. La ejecución original utilizó Tesla T4, Transformers 4.57.1 y Hugging Face Hub 0.36.0. Dependencias en requirements.txt.

## Archivos
- notebooks/Mod5_Act4.ipynb: cuaderno autónomo para Colab.
- src: scripts de descarga, preparación y evaluación.
- data/README.md: origen y estructura de los datos.
- results/comparacion_modelos.csv: métricas originales transcritas del PDF del notebook.

## Ejecución
1. Subir notebooks/Mod5_Act4.ipynb a Colab.
2. Activar GPU y ejecutar las celdas en orden.
3. Introducir un token de lectura de Hugging Face en el campo oculto o dejarlo vacío para modelos públicos.
4. El cuaderno descarga los datos, selecciona la muestra y evalúa los tres modelos.
5. Descargar los resultados nuevos desde la última celda.

Para utilizar los scripts, ejecutar desde la raíz:
```bash
pip install -r requirements.txt
python src/descargar_dataset.py
python src/preparar_dataset.py
python src/evaluar_modelos.py
```

## Resultados originales
| Modelo | Accuracy | F1 macro | Latencia media ms | Lotes s |
|---|---:|---:|---:|---:|
| FinBERT | 0.8930 | 0.8845 | 43.7008 | 5.3323 |
| DistilRoBERTa | 0.8530 | 0.8314 | 19.4131 | 2.6576 |
| FinBERT Tone | 0.7900 | 0.7573 | 16.2065 | 3.7152 |
Muestra: 1,000 noticias; semilla 110; lotes de 16; máximo 128 tokens. Latencia individual sobre 30 noticias. Los CSV incluyen también precision y recall macro.

## Conclusiones
Seleccioné provisionalmente FinBERT por calidad. DistilRoBERTa fue más rápido por lotes y FinBERT Tone por solicitud individual. La evaluación es exploratoria: algunos modelos se ajustaron con Financial PhraseBank y requieren validación independiente.

## Reproducibilidad
El notebook fue reconstruido desde el PDF original; este paquete no fue ejecutado nuevamente con GPU. Los resultados nuevos se guardan por separado. Se fija Kaggle versión 5 y semilla 110; las revisiones originales de los modelos no estaban registradas y las métricas pueden variar. Cada nueva evaluación registra versiones y revisiones.

