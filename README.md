# Sentimiento-Noticias-Financieras

# Análisis de sentimiento en noticias financieras

## Descripción del proyecto

En este proyecto comparé tres modelos preentrenados de Hugging Face para clasificar noticias financieras en inglés como positivas, negativas o neutrales.

Desarrollé el prototipo en Google Colab con GPU. Evalué accuracy, precision macro, recall macro y F1 macro, además de la latencia individual y el tiempo de procesamiento por lotes.

El objetivo es identificar qué modelo ofrece el mejor equilibrio entre calidad de clasificación y eficiencia para apoyar el monitoreo de información financiera.

## Dataset

Utilicé el dataset Sentiment Analysis for Financial News, disponible en Kaggle:

https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

El archivo utilizado fue `all-data.csv`, asociado a Financial PhraseBank. Contiene dos campos, cuyos nombres asigné al cargarlo:

| Campo | Descripción |
|---|---|
| sentimiento | Etiqueta original: negative, neutral o positive. |
| texto | Frase de una noticia financiera en inglés. |

Después de eliminar registros vacíos y textos duplicados, quedaron 4,838 noticias. Seleccioné una muestra de 1,000 registros con `random_state=110`.

| Sentimiento | Cantidad |
|---|---:|
| Neutral | 589 |
| Positivo | 302 |
| Negativo | 109 |
| Total | 1,000 |

Los datos se descargan al ejecutar el proyecto; no están incluidos en el repositorio.

## Modelos evaluados

- `ProsusAI/finbert`
- `yiyanghkust/finbert-tone`
- `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`

## Entorno de ejecución

- Google Colab con GPU.
- GPU Tesla T4 en la ejecución original.
- PyTorch compatible con el entorno de Colab.
- Transformers 4.57.1.
- Hugging Face Hub 0.36.0.
- Lotes de 16 textos.
- Longitud máxima de 128 tokens.

Las dependencias se encuentran en `requirements.txt`.

## Organización del repositorio

| Archivo o carpeta | Contenido |
|---|---|
| README.md | Descripción, instrucciones y conclusiones. |
| requirements.txt | Dependencias del proyecto. |
| notebooks/ | Cuaderno ejecutable en Google Colab. |
| src/ | Scripts de descarga, preparación y evaluación. |
| data/ | Documentación del origen y estructura del dataset. |
| results/ | Tabla comparativa de resultados originales. |

## Ejecución en Google Colab

1. Descargar `notebooks/Mod5_Act4.ipynb` y abrirlo en Google Colab.
2. Seleccionar **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Ejecutar las celdas en orden.
4. Introducir un token de lectura de Hugging Face en el campo oculto. También se puede dejar vacío para acceder a los modelos públicos.
5. Descargar y preparar el dataset mediante las celdas del notebook.
6. Ejecutar la evaluación de los tres modelos.
7. Revisar la tabla comparativa y las conclusiones.
8. Descargar los resultados nuevos mediante la última celda.

El notebook es autónomo y no requiere ejecutar previamente los scripts de `src`.

## Ejecución de los scripts de soporte

Desde la carpeta principal del proyecto:

```bash
pip install -r requirements.txt
python src/descargar_dataset.py
python src/preparar_dataset.py
python src/evaluar_modelos.py
```

Los scripts descargan los datos en `data/raw`, generan la muestra en `data/processed` y guardan los resultados nuevos en `results`.

## Resultados originales

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Latencia media individual (ms) | Tiempo por lotes (s) |
|---|---:|---:|---:|---:|---:|---:|
| FinBERT | 0.8930 | 0.8586 | 0.9212 | 0.8845 | 43.7008 | 5.3323 |
| DistilRoBERTa | 0.8530 | 0.8367 | 0.8267 | 0.8314 | 19.4131 | 2.6576 |
| FinBERT Tone | 0.7900 | 0.8034 | 0.7363 | 0.7573 | 16.2065 | 3.7152 |

Estas cifras fueron transcritas de las salidas del notebook original exportado a PDF.

Las métricas corresponden a 1,000 noticias. La latencia individual se midió sobre 30 textos y el tiempo por lotes sobre la muestra completa. Los tiempos excluyen la descarga y carga de los modelos.

## Conclusiones

Seleccioné provisionalmente FinBERT de ProsusAI porque obtuvo la mayor accuracy y el mejor F1 macro.

DistilRoBERTa fue la alternativa más rápida para procesar lotes, mientras que FinBERT Tone presentó la menor latencia individual.

Para este caso de uso prioricé la calidad de clasificación. Si la prioridad fuera procesar grandes volúmenes en menos tiempo, consideraría DistilRoBERTa como alternativa.

## Limitaciones

- Algunos modelos fueron ajustados previamente con Financial PhraseBank, por lo que puede existir coincidencia entre sus datos de entrenamiento y los textos evaluados.
- Los resultados son exploratorios y requieren confirmación con noticias independientes.
- No realicé fine-tuning, calibración de confianza ni atribución de palabras para explicar las predicciones.
- No medí memoria máxima de GPU ni costo monetario.
- Los resultados en inglés no demuestran desempeño en español.
- El límite de 128 tokens puede truncar algunos textos.

## Reproducibilidad y mantenimiento

El notebook incluido fue reconstruido a partir del código visible en el PDF original. El paquete no se ejecutó nuevamente con GPU antes de su entrega.

Los scripts fijan la versión 5 del dataset de Kaggle y la semilla 110. La descarga registra la huella SHA-256 del archivo utilizado.

Las revisiones originales de los modelos no estaban registradas, por lo que no garantizo reproducir exactamente las métricas históricas. Las nuevas ejecuciones registran las versiones del entorno y las revisiones de los modelos.

Los resultados nuevos se guardan por separado para conservar la tabla original. Como siguiente mejora, evaluaría métricas por clase y repetiría las mediciones de latencia.


## Referencias

- Dataset: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- FinBERT: https://huggingface.co/ProsusAI/finbert
- FinBERT Tone: https://huggingface.co/yiyanghkust/finbert-tone
- DistilRoBERTa: https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
