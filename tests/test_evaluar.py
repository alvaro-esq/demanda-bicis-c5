import mlflow
import pytest

from bicis import entrenar, evaluar


def test_rango_mes():
    assert evaluar.rango_mes("2012-02") == ("2012-02-01", "2012-02-29")
    assert evaluar.rango_mes("2011-12") == ("2011-12-01", "2011-12-31")


def test_evalua_otro_mes_y_lo_registra_en_el_mismo_run(entorno, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((entorno.raiz / "mlruns").as_uri())
    run_id = entrenar.entrenar()["mlflow_run_id"]

    r = evaluar.evaluar(run_id, "2012-01")
    d = entorno.datos
    assert r["mes_evaluado"]["horas"] == len(d[d.dteday.str.startswith("2012-01")])
    assert r["comparado_con"] == "2011-12"
    # Diciembre recalculado con el modelo del run = MAE de validación del entrenamiento.
    run = mlflow.get_run(run_id)
    assert r["validacion"]["mae"] == pytest.approx(run.data.metrics["val_mae"])
    assert r["sentido"] in {"sobreestima", "subestima"}
    assert (r["sentido"] == "sobreestima") == (r["mes_evaluado"]["error_medio"] > 0)

    run = mlflow.get_run(run_id)  # mismo run, no uno nuevo
    assert run.info.status == "FINISHED"
    assert len(mlflow.search_runs(experiment_ids=[run.info.experiment_id])) == 1
    assert run.data.metrics["eval_2012_01_mae"] == pytest.approx(r["mes_evaluado"]["mae"])
    assert "eval_2012_01_error_medio" in run.data.metrics
    historia = mlflow.MlflowClient().get_metric_history(run_id, "eval_2012_01_mae_hora")
    assert sorted(m.step for m in historia) == list(range(24))
    assert {i.tags[0].value for i in run.inputs.dataset_inputs} == {"training", "validation", "evaluation"}
    nombres = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "evaluacion_2012-01")}
    assert {"evaluacion_2012-01/comparacion_por_hora.csv", "evaluacion_2012-01/comparacion_por_hora.png",
            "evaluacion_2012-01/resumen.json"} <= nombres
