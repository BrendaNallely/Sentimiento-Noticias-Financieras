import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import gc
import json
import time
import importlib.metadata as metadata
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODELOS = {
    "FinBERT": "ProsusAI/finbert",
    "FinBERT Tone": "yiyanghkust/finbert-tone",
    "DistilRoBERTa": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
}
ETIQUETAS = {
    "FinBERT Tone": {"label_0": "neutral", "label_1": "positive", "label_2": "negative"},
    "DistilRoBERTa": {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
}

def evaluar(datos):
    # Utilizo la misma muestra y configuración para los tres modelos.
    device = 0 if torch.cuda.is_available() else -1
    textos = datos["texto"].tolist()
    reales = datos["sentimiento"].tolist()
    resultados, predicciones = [], []
    revisiones = {}
    for nombre, identificador in MODELOS.items():
        print("Evaluando", nombre)
        clasificador = pipeline("text-classification", model=identificador, tokenizer=identificador, device=device)
        revisiones[nombre] = getattr(clasificador.model.config, "_commit_hash", None)
        # Caliento el modelo antes de medir las solicitudes individuales.
        clasificador(textos[0], truncation=True, max_length=128)
        tiempos = []
        for texto in textos[:30]:
            if device == 0: torch.cuda.synchronize()
            inicio = time.perf_counter()
            clasificador(texto, truncation=True, max_length=128)
            if device == 0: torch.cuda.synchronize()
            tiempos.append((time.perf_counter() - inicio) * 1000)
        if device == 0: torch.cuda.synchronize()
        inicio = time.perf_counter()
        salidas = clasificador(textos, batch_size=16, truncation=True, max_length=128)
        if device == 0: torch.cuda.synchronize()
        tiempo_total = time.perf_counter() - inicio
        mapa = ETIQUETAS.get(nombre, {})
        pred = [mapa.get(s["label"].lower(), s["label"].lower()) for s in salidas]
        precision, recall, f1, _ = precision_recall_fscore_support(
            reales, pred, labels=["negative", "neutral", "positive"], average="macro", zero_division=0
        )
        resultados.append({
            "Modelo": nombre, "Accuracy": accuracy_score(reales, pred),
            "Precision macro": precision, "Recall macro": recall, "F1 macro": f1,
            "Latencia media (ms)": float(np.mean(tiempos)),
            "Tiempo total por lotes (s)": tiempo_total
        })
        for texto, real, etiqueta, salida in zip(textos, reales, pred, salidas):
            predicciones.append({"Modelo": nombre, "texto": texto, "sentimiento": real,
                                 "prediccion": etiqueta, "confianza": salida["score"]})
        del clasificador
        gc.collect()
        if device == 0: torch.cuda.empty_cache()
    tabla = pd.DataFrame(resultados).sort_values("F1 macro", ascending=False).reset_index(drop=True)
    Path("results").mkdir(exist_ok=True)
    # Los resultados nuevos tienen otro nombre para conservar la evidencia original.
    tabla.to_csv("results/comparacion_modelos_nueva.csv", index=False)
    pd.DataFrame(predicciones).to_csv("results/predicciones_nuevas.csv", index=False)
    entorno = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": torch.cuda.get_device_name(0) if device == 0 else "CPU",
        "registros": len(datos), "semilla": 110, "batch_size": 16, "max_length": 128,
        "textos_latencia": min(30, len(datos)), "revisiones_modelos": revisiones,
        "versiones": {p: metadata.version(p) for p in ["torch", "transformers", "huggingface_hub", "kagglehub", "pandas", "numpy", "scikit-learn"]}
    }
    Path("results/entorno_nuevo.json").write_text(json.dumps(entorno, indent=2, ensure_ascii=False), encoding="utf-8")
    return tabla

if __name__ == "__main__":
    tabla = evaluar(pd.read_csv("data/processed/muestra_evaluacion.csv"))
    print(tabla.round(4).to_string(index=False))
