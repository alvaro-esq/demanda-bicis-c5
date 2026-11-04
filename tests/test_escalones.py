import mlflow
import pytest
import yaml

from bicis import escalones


def test_regla_menor_costo_y_en_empate_menor_mae():
    a = {"costo": 10.0, "mae": 5.0}
    b = {"costo": 10.0, "mae": 4.0}
    c = {"costo": 9.0, "mae": 50.0}
    assert min([a, b], key=escalones.clave_eleccion) is b
    assert min([a, b, c], key=escalones.clave_eleccion) is c


@pytest.mark.parametrize("ventana", ["entrenamiento", "validacion"])
def test_nunca_usa_2012_para_elegir(entorno, ventana):
    p = yaml.safe_load(entorno.params.read_text())
    p["modelo"][ventana]["hasta"] = "2012-01-15"
    with pytest.raises(ValueError, match="2012"):
        escalones.verificar_sin_2012(p)


def test_escalones_con_busqueda_anidada(entorno, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((entorno.raiz / "mlruns").as_uri())
    r = escalones.ejecutar()

    tipos = [x["tipo"] for x in r["resultados"]]
    assert tipos == ["perfil_historico", "ridge", "random_forest", "hist_gradient_boosting"]
    assert r["ganador"] == min(r["resultados"], key=escalones.clave_eleccion)
    assert not entorno.modelo.exists()   # producción intacta

    hgb = r["resultados"][-1]
    assert [h["learning_rate"] for h in hgb["busqueda"]] == [0.05, 0.1, 0.2]
    for h in hgb["busqueda"]:
        run = mlflow.get_run(h["run_id"])
        assert run.data.tags["mlflow.parentRunId"] == hgb["run_padre"]
        assert run.data.params["hp_learning_rate"] == str(h["learning_rate"])
    for x in r["resultados"]:
        tags = mlflow.get_run(x["run_id"]).data.tags
        assert tags["grupo"] == r["grupo"]
        assert tags["ganador_escalones"] == str(x is r["ganador"]).lower()


def test_reusar_run_rf_exige_que_sea_comparable(entorno, monkeypatch):
    from bicis import entrenar
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((entorno.raiz / "mlruns").as_uri())
    run_rf = entrenar.entrenar("random_forest", guardar=False)["mlflow_run_id"]
    escalones._validar_run_reutilizado(run_rf, escalones.config.params())   # comparable: no falla

    run_et = entrenar.entrenar("extra_trees", guardar=False)["mlflow_run_id"]
    with pytest.raises(ValueError, match="no es comparable"):
        escalones._validar_run_reutilizado(run_et, escalones.config.params())
