import json

import joblib
import numpy as np

from bicis import config
from conftest import cargar_script, crear_fuente


def test_guarda_modelo_y_metricas(modelo_entrenado):
    e = modelo_entrenado.entorno
    assert e.modelo.exists() and e.metricas.exists()
    m = json.loads(e.metricas.read_text())
    assert m["mae_validacion"] >= 0
    assert m["validacion"] == {"desde": "2011-12-01", "hasta": "2011-12-31"}


def test_entrena_con_enero_a_noviembre_y_valida_con_diciembre(modelo_entrenado):
    e, m = modelo_entrenado.entorno, modelo_entrenado.metricas
    d = e.datos
    assert m["filas_entrenamiento"] == len(d[(d.dteday >= "2011-01-01") & (d.dteday <= "2011-11-30")])
    assert m["filas_validacion"] == len(d[d.dteday.str.startswith("2011-12")])


def test_el_modelo_no_usa_casual_ni_registered(modelo_entrenado):
    modelo = joblib.load(modelo_entrenado.entorno.modelo)
    assert list(modelo.feature_names_in_) == config.features()
    assert not {"casual", "registered", "cnt"} & set(modelo.feature_names_in_)


def test_casual_y_registered_no_influyen_en_el_modelo(entorno):
    entrenar = cargar_script("entrenar")
    entrenar.entrenar()
    X = entorno.datos[config.features()].head(200)
    antes = joblib.load(entorno.modelo).predict(X)

    # Misma fuente con casual/registered alterados (cnt intacto).
    rng = np.random.default_rng(1)
    otra = entorno.datos.copy()
    otra["casual"] = rng.integers(0, otra["cnt"] + 1)
    otra["registered"] = otra["cnt"] - otra["casual"]
    crear_fuente.escribir_sqlite(otra, entorno.fuente_db, "registros_horarios")
    entrenar.entrenar()
    assert np.array_equal(joblib.load(entorno.modelo).predict(X), antes)
