import mlflow
import numpy as np
import pandas as pd
import pytest

from bicis import config, costos, extraer, limpiar, predecir, registro, transformar

FECHA = "2012-01-10"  # fuera del periodo de entrenamiento


def _hasta_transformar(fecha=FECHA):
    for paso in (extraer, limpiar, transformar):
        paso.ejecutar(fecha)


def test_predice_cada_hora_con_la_version_champion(champion):
    _hasta_transformar()
    df = pd.read_csv(predecir.ejecutar(FECHA))
    assert list(df.columns) == ["fecha", "hr", "cnt_predicho", "version_modelo"]
    assert df["hr"].tolist() == list(range(24)) and (df["fecha"] == FECHA).all()
    assert pd.api.types.is_integer_dtype(df["cnt_predicho"]) and (df["cnt_predicho"] >= 0).all()
    assert (df["version_modelo"] == f"demanda-bicis-c5/{champion.registro['version']}").all()


def test_aplica_el_factor_de_la_version(champion):
    _hasta_transformar()
    feats = pd.read_csv(config.dir_staging(FECHA) / "features.csv")
    c = registro.cargar_champion()
    assert c.factor == pytest.approx(champion.metricas["factor_ajuste"])
    crudo = np.rint(np.clip(c.modelo.predict(feats), 0, None))
    esperado = costos.planificar(crudo, c.factor)
    assert pd.read_csv(predecir.ejecutar(FECHA))["cnt_predicho"].tolist() == esperado.astype(int).tolist()


def test_usa_el_factor_guardado_en_la_version(champion):
    _hasta_transformar()
    sin = pd.read_csv(predecir.ejecutar(FECHA))["cnt_predicho"]
    v = champion.registro["version"]
    mlflow.MlflowClient().set_model_version_tag("demanda-bicis-c5", v, "factor_ajuste", "0.5")
    con = pd.read_csv(predecir.ejecutar(FECHA))["cnt_predicho"]
    assert (con >= sin).all() and con.sum() > sin.sum()


def test_sigue_al_alias(champion):
    _hasta_transformar()
    nueva = registro.registrar_champion(champion.metricas["mlflow_run_id"], motivo="otra")
    df = pd.read_csv(predecir.ejecutar(FECHA))
    assert (df["version_modelo"] == f"demanda-bicis-c5/{nueva['version']}").all()


def test_falla_si_no_hay_champion(entorno, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((entorno.raiz / "mlruns").as_uri())
    _hasta_transformar()
    with pytest.raises(FileNotFoundError, match="champion"):
        predecir.ejecutar(FECHA)


def test_falla_si_la_version_no_tiene_factor(champion):
    _hasta_transformar()
    mlflow.MlflowClient().delete_model_version_tag("demanda-bicis-c5", champion.registro["version"],
                                                   "factor_ajuste")
    with pytest.raises(ValueError, match="factor_ajuste"):
        predecir.ejecutar(FECHA)


def test_registro_guarda_los_tags_pedidos(champion):
    mv = mlflow.MlflowClient().get_model_version_by_alias("demanda-bicis-c5", "champion")
    for tag in ("factor_ajuste", "val_costo_medio_hora_ajustado", "val_mae",
                "entrenamiento_inicio", "entrenamiento_fin", "features"):
        assert tag in mv.tags
    assert mv.tags["features"].split(",") == config.features()
    assert (mv.tags["entrenamiento_inicio"], mv.tags["entrenamiento_fin"]) == ("2011-01-01", "2011-11-30")


def test_falla_si_falta_el_archivo_anterior(champion):
    with pytest.raises(FileNotFoundError):
        predecir.ejecutar(FECHA)
