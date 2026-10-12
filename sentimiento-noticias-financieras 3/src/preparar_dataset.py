from pathlib import Path
import pandas as pd

def preparar(archivo="data/raw/all-data.csv"):
    # Replico la limpieza y el muestreo del notebook documentado.
    df = pd.read_csv(archivo, encoding="latin-1", header=None, names=["sentimiento", "texto"])
    df = df.dropna().drop_duplicates(subset="texto").copy()
    df["sentimiento"] = df["sentimiento"].str.strip().str.lower()
    df["texto"] = df["texto"].str.strip()
    datos = df.sample(n=min(1000, len(df)), random_state=110).reset_index(drop=True)
    destino = Path("data/processed")
    destino.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino / "noticias_limpias.csv", index=False)
    datos.to_csv(destino / "muestra_evaluacion.csv", index=False)
    print("Noticias disponibles:", len(df))
    print("Noticias que evaluaré:", len(datos))
    print(datos["sentimiento"].value_counts())
    return datos

if __name__ == "__main__":
    preparar()
