"""Constructores de los modelos candidatos (escalones de complejidad).

Cada constructor recibe los hiperparámetros de params.yaml
(modelo.hiperparametros.<tipo>) y devuelve un estimador de scikit-learn sin
entrenar. Todos se entrenan con un DataFrame de features (config.features()), así
que todos exponen feature_names_in_ y predecir.py los puede usar sin cambios.

  perfil_historico        promedio de cnt por hora y tipo de día (laborable o no)
  ridge                   Ridge con hr y mnth one-hot; el resto escalado
  random_forest           RandomForestRegressor
  extra_trees             ExtraTreesRegressor
  gradient_boosting       GradientBoostingRegressor
  hist_gradient_boosting  HistGradientBoostingRegressor con el objetivo en log1p
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class PerfilHistorico(RegressorMixin, BaseEstimator):
    """Predice el promedio de entrenamiento de y para cada combinación de `claves`.

    Una combinación que no se vio en el entrenamiento recibe el promedio global.
    El estado son solo arreglos de numpy, para que skops lo serialice sin código.
    """

    def __init__(self, claves=("hr", "workingday")):
        self.claves = claves

    def fit(self, X: pd.DataFrame, y):
        X = self._validar(X, entrenando=True)
        grupos = (pd.DataFrame({**{c: X[c].to_numpy() for c in self.claves}, "y": np.asarray(y, float)})
                  .groupby(list(self.claves))["y"].mean())
        self.claves_ = np.asarray(grupos.index.to_list(), dtype=float).reshape(len(grupos), -1)
        self.valores_ = grupos.to_numpy(dtype=float)
        self.media_global_ = float(np.mean(np.asarray(y, float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "valores_")
        X = self._validar(X, entrenando=False)
        tabla = {tuple(k): v for k, v in zip(self.claves_.tolist(), self.valores_)}
        filas = X[list(self.claves)].to_numpy(dtype=float).tolist()
        return np.array([tabla.get(tuple(f), self.media_global_) for f in filas])

    def _validar(self, X, entrenando: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PerfilHistorico necesita un DataFrame con nombres de columna")
        faltan = [c for c in self.claves if c not in X.columns]
        if faltan:
            raise ValueError(f"Faltan columnas para el perfil: {faltan}")
        if entrenando:
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]
        return X


def perfil_historico(claves=("hr", "workingday")):
    return PerfilHistorico(claves=tuple(claves))


def ridge(alpha: float = 1.0, categoricas=("hr", "mnth"), features: list[str] | None = None):
    categoricas = list(categoricas)
    numericas = [f for f in (features or []) if f not in categoricas]
    prep = ColumnTransformer(
        [
            # handle_unknown="ignore": una categoría no vista (p. ej. mnth = 12 si se
            # entrena con enero-noviembre) queda en ceros y aporta solo el intercepto.
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
            ("numericas", StandardScaler(), numericas),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", prep), ("ridge", Ridge(alpha=alpha))])


def hist_gradient_boosting(objetivo_log: bool = True, **hp):
    modelo = HistGradientBoostingRegressor(**hp)
    if not objetivo_log:
        return modelo
    # Entrena sobre log1p(cnt) y deshace la transformación al predecir.
    return TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1,
                                      check_inverse=False)


CONSTRUCTORES = {
    "perfil_historico": lambda hp, feats: perfil_historico(**hp),
    "ridge": lambda hp, feats: ridge(features=feats, **hp),
    "random_forest": lambda hp, feats: RandomForestRegressor(**hp),
    "extra_trees": lambda hp, feats: ExtraTreesRegressor(**hp),
    "gradient_boosting": lambda hp, feats: GradientBoostingRegressor(**hp),
    "hist_gradient_boosting": lambda hp, feats: hist_gradient_boosting(**hp),
}

# Tipos que skops no carga por defecto y que confiamos porque los archivos los
# generamos nosotros. Lista explícita: no se agrega nada de forma automática.
TIPOS_CONFIABLES_SKOPS = [
    "sklearn.tree._tree.Tree",          # árboles de RF, ExtraTrees y GradientBoosting
    "bicis.modelos.PerfilHistorico",    # nuestro perfil histórico
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",  # árboles de HGB
]
