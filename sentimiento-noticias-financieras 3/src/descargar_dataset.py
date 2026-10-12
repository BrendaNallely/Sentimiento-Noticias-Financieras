from pathlib import Path
import hashlib
import json
import shutil
import kagglehub

def descargar():
    # Fijo la versión 5, correspondiente a la descarga documentada.
    identificador = "ankurzing/sentiment-analysis-for-financial-news/versions/5"
    ruta = Path(kagglehub.dataset_download(identificador))
    destino = Path("data/raw")
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / "all-data.csv"
    shutil.copy2(ruta / "all-data.csv", archivo)
    if (ruta / "FinancialPhraseBank").exists():
        shutil.copytree(ruta / "FinancialPhraseBank", destino / "FinancialPhraseBank", dirs_exist_ok=True)
    metadatos = {
        "fuente": "Kaggle", "identificador": identificador,
        "archivo": "all-data.csv",
        "sha256": hashlib.sha256(archivo.read_bytes()).hexdigest()
    }
    (destino / "origen_dataset.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Dataset descargado:", archivo)
    return archivo

if __name__ == "__main__":
    descargar()
