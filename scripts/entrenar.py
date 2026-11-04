"""Entrena el Random Forest de cnt y mide el MAE.

Entrena con modelo.entrenamiento (enero-noviembre de 2011) y mide el MAE en
modelo.validacion (diciembre de 2011), leyendo la fuente en solo lectura.
Nunca lee casual ni registered. Guarda el modelo (rutas.modelo) y las métricas
(rutas.metricas).

Uso:
    python scripts/entrenar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bicis import config  # noqa: E402


def leer_periodo(desde: str, hasta: str, columnas: list[str]) -> pd.DataFrame:
    assert not config.prohibidas().intersection(columnas) - {"dteday"}
    consulta = (
        f"SELECT {', '.join(columnas)} FROM {config.tabla_fuente()} "
        "WHERE dteday BETWEEN ? AND ? ORDER BY dteday, hr"
    )
    con = config.conectar_fuente()
    try:
        return pd.read_sql_query(consulta, con, params=(desde, hasta))
    finally:
        con.close()


def entrenar() -> dict:
    p = config.params()["modelo"]
    feats = config.features()
    objetivo = p["objetivo"]
    cols = list(dict.fromkeys(["dteday", "hr", *feats, objetivo]))

    ent = leer_periodo(p["entrenamiento"]["desde"], p["entrenamiento"]["hasta"], cols)
    val = leer_periodo(p["validacion"]["desde"], p["validacion"]["hasta"], cols)
    if ent.empty or val.empty:
        raise ValueError("No hay filas en el periodo de entrenamiento o de validación")

    modelo = RandomForestRegressor(**p["random_forest"])
    modelo.fit(ent[feats], ent[objetivo])

    # Mismo posproceso que predecir.py: recorte a >= 0 y redondeo a entero.
    pred = np.rint(np.clip(modelo.predict(val[feats]), 0, None))
    mae = mean_absolute_error(val[objetivo], pred)

    # Referencia ingenua: promedio de entrenamiento por (workingday, hr).
    base = ent.groupby(["workingday", "hr"])[objetivo].mean()
    pred_base = val.set_index(["workingday", "hr"]).index.map(base).to_numpy()
    mae_base = mean_absolute_error(val[objetivo], pred_base)

    metricas = {
        "mae_validacion": round(float(mae), 3),
        "mae_referencia_promedio_hora": round(float(mae_base), 3),
        "cnt_medio_validacion": round(float(val[objetivo].mean()), 3),
        "filas_entrenamiento": int(len(ent)),
        "filas_validacion": int(len(val)),
        "entrenamiento": p["entrenamiento"],
        "validacion": p["validacion"],
        "features": feats,
        "random_forest": p["random_forest"],
    }

    ruta_modelo = config.ruta("modelo")
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo, compress=3)
    ruta_metricas = config.ruta("metricas")
    ruta_metricas.write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metricas


def main() -> int:
    m = entrenar()
    print(f"MAE diciembre 2011: {m['mae_validacion']}  "
          f"(referencia promedio por hora: {m['mae_referencia_promedio_hora']})")
    print(f"Modelo  -> {config.ruta('modelo')}")
    print(f"Métricas -> {config.ruta('metricas')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
