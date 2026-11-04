import json

import joblib
import numpy as np
import pandas as pd
import pytest

from bicis import config, entrenar
from bicis.transformar import columnas_salida
from conftest import crear_fuente


def test_guarda_modelo_y_metricas(modelo_entrenado):
    e = modelo_entrenado.entorno
    assert e.modelo.exists() and e.metricas.exists()
    m = json.loads(e.metricas.read_text())
    assert m["mae_validacion"] >= 0 and m["rmse_validacion"] >= m["mae_validacion"]
    assert m["latencia_p95_ms_dia"] > 0 and m["modelo_tamano_mb"] > 0
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
    entrenar.entrenar(registrar=False)
    X = entorno.datos[config.features()].head(200)
    antes = joblib.load(entorno.modelo).predict(X)

    # Misma fuente con casual/registered alterados (cnt intacto).
    rng = np.random.default_rng(1)
    otra = entorno.datos.copy()
    otra["casual"] = rng.integers(0, otra["cnt"] + 1)
    otra["registered"] = otra["cnt"] - otra["casual"]
    crear_fuente.escribir_sqlite(otra, entorno.fuente_db, "registros_horarios")
    entrenar.entrenar(registrar=False)
    assert np.array_equal(joblib.load(entorno.modelo).predict(X), antes)


def test_error_medio_es_positivo_si_sobreestima():
    real = pd.Series([10, 20, 30])
    m = entrenar.metricas_error(real, np.array([12, 22, 32]))
    assert m == {"mae": 2.0, "rmse": 2.0, "error_medio": 2.0}
    assert entrenar.metricas_error(real, np.array([8, 18, 28]))["error_medio"] == -2.0


def test_huella_cambia_con_los_datos():
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.25]})
    assert entrenar.huella(df) == entrenar.huella(df.copy())
    assert entrenar.huella(df) != entrenar.huella(df.assign(b=[0.5, 0.26]))


@pytest.mark.parametrize("tipo", ["extra_trees", "gradient_boosting"])
def test_otros_tipos_de_modelo(entorno, tipo):
    m = entrenar.entrenar(tipo, registrar=False)
    assert m["modelo_tipo"] == tipo
    assert type(joblib.load(entorno.modelo)).__name__.lower().startswith(tipo.split("_")[0])


def test_modelo_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        entrenar.construir("svm", {})


def test_registra_el_run_en_mlflow(entorno, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (entorno.raiz / "mlruns").as_uri())
    mlflow.set_tracking_uri((entorno.raiz / "mlruns").as_uri())
    m = entrenar.entrenar()
    run = mlflow.get_run(m["mlflow_run_id"])
    assert mlflow.get_experiment(run.info.experiment_id).name == "demanda-bicis-c5"

    # 1. Parámetros
    p = run.data.params
    assert p["modelo_tipo"] == "random_forest" and p["hp_n_estimators"] == "10"
    assert (p["entrenamiento_inicio"], p["entrenamiento_fin"], p["validacion_mes"]) == \
        ("2011-01-01", "2011-11-30", "2011-12")
    # 2. Métricas
    assert {"val_mae", "val_rmse", "val_error_medio", "latencia_p95_ms_dia",
            "modelo_tamano_mb"} <= set(run.data.metrics)
    assert run.data.metrics["val_mae"] == pytest.approx(m["mae_validacion"], abs=1e-3)
    # 4. Linaje
    t = run.data.tags
    assert len(t["git_commit"]) == 40 or t["git_commit"] == "desconocido"
    assert t["features"].split(",") == config.features()
    assert len(t["huella_datos_entrenamiento"]) == 64
    contextos = {i.tags[0].value for i in run.inputs.dataset_inputs}
    assert contextos == {"training", "validation"}
    for i in run.inputs.dataset_inputs:
        columnas = {c["name"] for c in json.loads(i.dataset.schema)["mlflow_colspec"]}
        assert "cnt" in columnas and not {"casual", "registered"} & columnas
    # 3. Artefactos
    cliente = mlflow.MlflowClient()
    nombres = {a.path for a in cliente.list_artifacts(run.info.run_id, "evaluacion")}
    assert {"evaluacion/real_vs_prediccion.png", "evaluacion/mae_por_hora.png",
            "evaluacion/mae_por_hora.csv", "evaluacion/importancia_features.png",
            "evaluacion/importancia_features.csv", "evaluacion/linaje.json"} <= nombres

    # 5. Modelo con firma = columnas de transformar.py; se carga y predice esa tabla
    modelo = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/modelo")
    assert modelo.metadata.get_input_schema().input_names() == columnas_salida()
    dia = entorno.datos.loc[entorno.datos.dteday == "2012-01-10", columnas_salida()].reset_index(drop=True)
    esperado = joblib.load(entorno.modelo).predict(dia[config.features()])
    assert np.allclose(modelo.predict(dia), esperado)
