import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from bicis import extraer, limpiar, predecir, transformar

FECHA = "2012-01-10"  # fuera del periodo de entrenamiento


def _hasta_transformar(fecha=FECHA):
    for paso in (extraer, limpiar, transformar):
        paso.ejecutar(fecha)


def test_predice_cnt_entero_no_negativo_para_cada_hora(modelo_entrenado):
    _hasta_transformar()
    df = pd.read_csv(predecir.ejecutar(FECHA))
    assert list(df.columns) == ["fecha", "hr", "cnt_predicho"]
    assert df["hr"].tolist() == list(range(24))
    assert (df["fecha"] == FECHA).all()
    assert pd.api.types.is_integer_dtype(df["cnt_predicho"]) and (df["cnt_predicho"] >= 0).all()


def test_captura_el_patron_por_hora(modelo_entrenado):
    _hasta_transformar()  # 2012-01-10 es martes: pico a las 8 y a las 17-18
    df = pd.read_csv(predecir.ejecutar(FECHA)).set_index("hr")
    assert df.loc[8, "cnt_predicho"] > 2 * df.loc[3, "cnt_predicho"]


def test_falla_sin_modelo(entorno):
    _hasta_transformar()
    with pytest.raises(FileNotFoundError, match="modelo"):
        predecir.ejecutar(FECHA)


def test_falla_si_el_modelo_usa_otras_features(entorno):
    _hasta_transformar()
    X = pd.DataFrame({"hr": [0, 1], "casual": [1, 2]})
    entorno.modelo.parent.mkdir(parents=True)
    joblib.dump(RandomForestRegressor(n_estimators=2).fit(X, [1, 2]), entorno.modelo)
    with pytest.raises(ValueError, match="reentrena"):
        predecir.ejecutar(FECHA)


def test_falla_si_falta_el_archivo_anterior(modelo_entrenado):
    with pytest.raises(FileNotFoundError):
        predecir.ejecutar(FECHA)
