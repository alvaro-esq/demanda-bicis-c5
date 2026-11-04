"""Costo del operador y factor de ajuste de la predicción.

Los valores salen de configs/costos.yaml (rutas.costos o BICIS_COSTOS), y son
SUPUESTOS del caso, no costos medidos.

Por hora, con plan = bicicletas planificadas y real = cnt:
    costo = costo_faltante * max(real - plan, 0) + costo_sobrante * max(plan - real, 0)

El operador planifica con round(predicción * (1 + factor)), y el factor se
elige en validación como el que minimiza el costo medio por hora.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from bicis import config


@dataclass(frozen=True)
class Costos:
    faltante: float
    sobrante: float
    factor_minimo: float = 0.0
    factor_maximo: float = 0.40
    factor_paso: float = 0.01

    def __post_init__(self):
        if self.faltante < 0 or self.sobrante < 0:
            raise ValueError("Los costos no pueden ser negativos")
        if not 0 <= self.factor_minimo <= self.factor_maximo:
            raise ValueError("Rango de factor inválido: se espera 0 <= mínimo <= máximo")
        if self.factor_paso <= 0:
            raise ValueError("El paso del factor debe ser positivo")

    def factores(self) -> np.ndarray:
        n = int(round((self.factor_maximo - self.factor_minimo) / self.factor_paso))
        return np.round(self.factor_minimo + self.factor_paso * np.arange(n + 1), 6)


def cargar() -> Costos:
    with open(config.ruta("costos"), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    rango = c.get("factor_ajuste", {})
    return Costos(
        faltante=float(c["costo_faltante"]),
        sobrante=float(c["costo_sobrante"]),
        factor_minimo=float(rango.get("minimo", 0.0)),
        factor_maximo=float(rango.get("maximo", 0.40)),
        factor_paso=float(rango.get("paso", 0.01)),
    )


def costo_por_hora(real, plan, costos: Costos) -> np.ndarray:
    """Costo de cada hora: faltante * faltantes + sobrante * sobrantes."""
    real = np.asarray(real, dtype=float)
    plan = np.asarray(plan, dtype=float)
    if real.shape != plan.shape:
        raise ValueError(f"real y plan tienen tamaños distintos: {real.shape} vs {plan.shape}")
    if np.isnan(real).any() or np.isnan(plan).any():
        raise ValueError("real y plan no pueden tener valores vacíos")
    faltantes = np.clip(real - plan, 0, None)
    sobrantes = np.clip(plan - real, 0, None)
    return costos.faltante * faltantes + costos.sobrante * sobrantes


def costo_medio(real, plan, costos: Costos) -> float:
    c = costo_por_hora(real, plan, costos)
    if c.size == 0:
        raise ValueError("No hay horas para calcular el costo")
    return float(c.mean())


def planificar(pred, factor: float) -> np.ndarray:
    """Bicicletas que planifica el operador: predicción * (1 + factor), entera y >= 0."""
    return np.rint(np.clip(np.asarray(pred, dtype=float) * (1 + factor), 0, None))


def curva_factor(real, pred, costos: Costos) -> pd.DataFrame:
    """Costo medio por hora para cada factor del rango configurado."""
    filas = []
    for f in costos.factores():
        plan = planificar(pred, f)
        c = costo_por_hora(real, plan, costos)
        filas.append({
            "factor": float(f),
            "costo_medio_hora": float(c.mean()),
            "faltantes_medios_hora": float(np.clip(np.asarray(real, float) - plan, 0, None).mean()),
            "sobrantes_medios_hora": float(np.clip(plan - np.asarray(real, float), 0, None).mean()),
        })
    return pd.DataFrame(filas)


def mejor_factor(real, pred, costos: Costos) -> tuple[float, pd.DataFrame]:
    """Factor que minimiza el costo medio por hora; en empate, el menor."""
    curva = curva_factor(real, pred, costos)
    minimo = curva["costo_medio_hora"].min()
    elegido = curva.loc[np.isclose(curva["costo_medio_hora"], minimo), "factor"].min()
    return float(elegido), curva
