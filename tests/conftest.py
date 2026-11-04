"""Fixtures compartidas: entorno aislado en tmp_path con una fuente sintética."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

RAIZ = Path(__file__).resolve().parent.parent


def cargar_script(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


crear_fuente = cargar_script("crear_fuente")


def datos_sinteticos(desde: str, hasta: str, faltantes: dict[str, list[int]] | None = None,
                     semilla: int = 0) -> pd.DataFrame:
    """Filas con el esquema de hour.csv. `faltantes` = {fecha: [horas a omitir]}."""
    faltantes = faltantes or {}
    rng = np.random.default_rng(semilla)
    filas, instant = [], 0
    d, fin = date.fromisoformat(desde), date.fromisoformat(hasta)
    while d <= fin:
        weekday = (d.weekday() + 1) % 7  # como en el dataset: 0 = domingo
        workingday = int(1 <= weekday <= 5)
        season = (d.month % 12) // 3 + 1
        for hr in range(24):
            if hr in faltantes.get(d.isoformat(), []):
                continue
            instant += 1
            temp = float(np.clip(0.3 + 0.4 * np.sin(np.pi * d.month / 12) + rng.normal(0, 0.05), 0, 1))
            pico = 1.0 if (workingday and hr in (8, 17, 18)) or (not workingday and 11 <= hr <= 16) else 0.2
            cnt = int(20 + 400 * pico * (0.5 + temp) + rng.integers(0, 20))
            casual = cnt // 4
            filas.append({
                "instant": instant, "dteday": d.isoformat(), "season": season,
                "yr": d.year - 2011, "mnth": d.month, "hr": hr, "holiday": 0,
                "weekday": weekday, "workingday": workingday,
                "weathersit": int(rng.integers(1, 4)), "temp": round(temp, 4),
                "atemp": round(temp * 0.9, 4), "hum": round(float(rng.uniform(0.3, 0.9)), 4),
                "windspeed": round(float(rng.uniform(0, 0.5)), 4),
                "casual": casual, "registered": cnt - casual, "cnt": cnt,
            })
        d += timedelta(days=1)
    return pd.DataFrame(filas)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """params.yaml propio en tmp_path (RF pequeño) y fuente con todo 2011 y enero de 2012.

    El 2011-03-05 no tiene las horas 3, 4 y 23.
    """
    for var in ["BICIS_ZIP_ORIGEN", "BICIS_FUENTE_DB", "BICIS_STAGING_DIR",
                "BICIS_SALIDAS_DB", "BICIS_MODELO", "BICIS_METRICAS",
                "MLFLOW_TRACKING_URI"]:
        monkeypatch.delenv(var, raising=False)

    params = yaml.safe_load((RAIZ / "params.yaml").read_text(encoding="utf-8"))
    for hp in params["modelo"]["hiperparametros"].values():
        hp["n_estimators"] = 10
        if "n_jobs" in hp:
            hp["n_jobs"] = 1
    params["mlflow"]["repeticiones_latencia"] = 3
    ruta_params = tmp_path / "params.yaml"
    ruta_params.write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.setenv("BICIS_PARAMS", str(ruta_params))

    df = datos_sinteticos("2011-01-01", "2012-01-31", faltantes={"2011-03-05": [3, 4, 23]})
    fuente_db = tmp_path / "fuente" / "bicis.db"
    crear_fuente.escribir_sqlite(df, fuente_db, "registros_horarios")

    return SimpleNamespace(
        raiz=tmp_path, params=ruta_params, fuente_db=fuente_db, datos=df,
        staging=tmp_path / "data" / "staging",
        salidas_db=tmp_path / "data" / "salidas" / "predicciones.db",
        modelo=tmp_path / "models" / "modelo.joblib",
        metricas=tmp_path / "metrics.json",
    )


@pytest.fixture
def modelo_entrenado(entorno):
    from bicis import entrenar
    metricas = entrenar.entrenar(registrar=False)
    return SimpleNamespace(entorno=entorno, metricas=metricas)


def escribir_staging(entorno, fecha: str, nombre: str, df: pd.DataFrame) -> Path:
    d = entorno.staging / fecha
    d.mkdir(parents=True, exist_ok=True)
    ruta = d / nombre
    df.to_csv(ruta, index=False)
    return ruta
