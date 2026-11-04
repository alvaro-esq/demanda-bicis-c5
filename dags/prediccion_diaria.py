"""Predicción diaria de alquileres de bicicletas por hora (demanda-bicis-c5).

Cada mañana de enero de 2012 corre los cinco pasos del pipeline en cadena:
extraer -> limpiar -> transformar -> predecir -> almacenar.

El DAG solo orquesta: cada tarea llama a ejecutar(fecha) del módulo
correspondiente de src/bicis/ con la fecha lógica (ds) y devuelve la ruta que
escribió; esa ruta llega a la tarea siguiente por XCom. Toda la lógica vive
en src/bicis/ (ver CLAUDE.md).
"""

from __future__ import annotations

import importlib
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


def _ejecutar(paso: str, ds: str, ruta_anterior: str | None = None) -> str:
    # Import diferido: el dag-processor no carga pandas ni scikit-learn al parsear.
    modulo = importlib.import_module(f"bicis.{paso}")
    if ruta_anterior is not None:
        print(f"{paso}: entrada del paso anterior -> {ruta_anterior}")
    ruta = modulo.ejecutar(ds)
    print(f"{paso}: {ds} -> {ruta}")
    return str(ruta)


@dag(
    dag_id="prediccion_diaria",
    description="Predice cada mañana el cnt de cada hora del día",
    schedule="0 6 * * *",  # cada mañana a las 06:00 UTC; ds = el día a predecir
    start_date=pendulum.datetime(2012, 1, 1, tz="UTC"),
    end_date=pendulum.datetime(2012, 1, 31, 23, 59, tz="UTC"),
    catchup=True,  # al despausar, crea las corridas de todo enero de 2012
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["demanda-bicis-c5"],
)
def prediccion_diaria():
    @task
    def extraer(ds: str | None = None) -> str:
        return _ejecutar("extraer", ds)

    @task
    def limpiar(ruta_anterior: str, ds: str | None = None) -> str:
        return _ejecutar("limpiar", ds, ruta_anterior)

    @task
    def transformar(ruta_anterior: str, ds: str | None = None) -> str:
        return _ejecutar("transformar", ds, ruta_anterior)

    @task
    def predecir(ruta_anterior: str, ds: str | None = None) -> str:
        return _ejecutar("predecir", ds, ruta_anterior)

    @task
    def almacenar(ruta_anterior: str, ds: str | None = None) -> str:
        return _ejecutar("almacenar", ds, ruta_anterior)

    almacenar(predecir(transformar(limpiar(extraer()))))


prediccion_diaria()
