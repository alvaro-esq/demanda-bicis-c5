"""Paso 4: carga el modelo y predice cnt para cada hora.

Nunca entrena. Recorta las predicciones a >= 0 y las redondea a entero.
Escribe fecha, hr, cnt_predicho.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bicis import config

ENTRADA = "features.csv"
SALIDA = "predicciones.csv"


def cargar_modelo():
    ruta = config.ruta("modelo")
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el modelo {ruta}. Corre `make entrenar` o `dvc pull`.")
    modelo = joblib.load(ruta)
    usadas = list(getattr(modelo, "feature_names_in_", []))
    if usadas != config.features():
        raise ValueError(
            f"El modelo usa {usadas} pero params.yaml define {config.features()}: reentrena"
        )
    return modelo


def ejecutar(fecha) -> Path:
    fecha = config.validar_fecha(fecha)
    df = pd.read_csv(config.leer_paso_anterior(fecha, ENTRADA))
    modelo = cargar_modelo()

    faltan = [c for c in modelo.feature_names_in_ if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan features en {ENTRADA}: {faltan}")
    pred = modelo.predict(df[list(modelo.feature_names_in_)])

    salida_df = pd.DataFrame({
        "fecha": fecha,
        "hr": df["hr"].astype("int64"),
        "cnt_predicho": np.rint(np.clip(pred, 0, None)).astype("int64"),
    })
    salida = config.salida_paso(fecha, SALIDA)
    salida_df.to_csv(salida, index=False)
    return salida
