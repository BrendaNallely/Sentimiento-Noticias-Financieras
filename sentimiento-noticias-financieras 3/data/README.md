# Dataset

Fuente: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
Versión utilizada en los scripts: 5. Archivo: all-data.csv (Latin-1, sin encabezados).

| Campo | Descripción |
|---|---|
| sentimiento | Etiqueta original: negative, neutral o positive |
| texto | Frase de una noticia financiera en inglés |

Los scripts descargan el CSV y su documentación en data/raw. La preparación elimina registros vacíos y textos duplicados, normaliza espacios externos y etiquetas y selecciona hasta 1,000 noticias con random_state=110. Guarda los CSV preparados en data/processed.
La ejecución documentada obtuvo 4,838 noticias limpias; la muestra contiene 589 neutrales, 302 positivas y 109 negativas.

No se redistribuyen los datos en este paquete. El script registra SHA-256 de la descarga. Revisar License.txt original; la ficha de Financial PhraseBank declara CC BY-NC-SA 3.0.
