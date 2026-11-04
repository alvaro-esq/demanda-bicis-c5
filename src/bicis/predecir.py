"""Paso 4: carga el modelo champion de MLflow y predice cnt para cada hora.

Carga models:/<mlflow.modelo_registrado>@<mlflow.alias> (demanda-bicis-c5@champion)
del servidor en MLFLOW_TRACKING_URI y aplica el factor de ajuste guardado como
tag de esa versión: cnt_predicho = round(max(predicción, 0) * (1 + factor)).
Nunca entrena. Escribe fecha, hr, cnt_predicho y version_modelo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bicis import config, costos, registro
from bicis.transformar import columnas_salida

ENTRADA = "features.csv"
SALIDA = "predicciones.csv"


def ejecutar(fecha) -> Path:
    fecha = config.validar_fecha(fecha)
    df = pd.read_csv(config.leer_paso_anterior(fecha, ENTRADA))
    champion = registro.cargar_champion()

    cols = columnas_salida()
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {ENTRADA}: {faltan}")
    pred = np.asarray(champion.modelo.predict(df[cols]), dtype=float)

    salida_df = pd.DataFrame({
        "fecha": fecha,
        "hr": df["hr"].astype("int64"),
        # Mismo plan que se evaluó al elegir el factor: round(max(pred, 0) * (1 + factor)).
        "cnt_predicho": costos.planificar(np.rint(np.clip(pred, 0, None)), champion.factor).astype("int64"),
        "version_modelo": champion.etiqueta,
    })
    salida = config.salida_paso(fecha, SALIDA)
    salida_df.to_csv(salida, index=False)
    return salida
