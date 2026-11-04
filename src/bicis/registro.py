"""Registro del modelo en MLflow y lectura del champion.

  registrar_champion(run_id)  registra runs:/<run_id>/modelo como una versión nueva
                              de mlflow.modelo_registrado, le pone el alias
                              mlflow.alias (champion) y guarda en la versión los
                              tags factor_ajuste, costo, MAE, ventana y features.
  cargar_champion()           devuelve (modelo pyfunc, versión, factor) del alias.

Mover el alias a mano es excepcional (la primera promoción); se deja registrado
en la versión con el tag promocion=manual.

Uso:
    python -m bicis.registro --run-id <id>
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from bicis import config
from bicis.transformar import columnas_salida


def nombre_y_alias() -> tuple[str, str]:
    m = config.params()["mlflow"]
    return m["modelo_registrado"], m["alias"]


def registrar_champion(run_id: str, motivo: str = "") -> dict:
    import mlflow

    nombre, alias = nombre_y_alias()
    cliente = mlflow.MlflowClient()
    run = cliente.get_run(run_id)
    met, par, tags = run.data.metrics, run.data.params, run.data.tags
    faltan = [k for k in ("factor_ajuste", "val_costo_medio_hora_ajustado", "val_mae") if k not in met]
    if faltan:
        raise ValueError(f"El run {run_id} no tiene las métricas {faltan}")

    version = mlflow.register_model(f"runs:/{run_id}/modelo", nombre)
    tags_version = {
        "factor_ajuste": met["factor_ajuste"],
        "val_costo_medio_hora_ajustado": met["val_costo_medio_hora_ajustado"],
        "val_costo_medio_hora_sin_ajuste": met["val_costo_medio_hora"],
        "val_mae": met["val_mae"],
        "entrenamiento_inicio": par["entrenamiento_inicio"],
        "entrenamiento_fin": par["entrenamiento_fin"],
        "validacion_mes": par["validacion_mes"],
        "features": tags["features"],
        "modelo_tipo": par["modelo_tipo"],
        "run_id": run_id,
        "git_commit": tags.get("git_commit", ""),
        "promocion": "manual",
        "motivo": motivo,
    }
    for k, v in tags_version.items():
        cliente.set_model_version_tag(nombre, version.version, k, str(v))
    cliente.set_registered_model_alias(nombre, alias, version.version)
    return {"nombre": nombre, "version": version.version, "alias": alias, "tags": tags_version}


@dataclass(frozen=True)
class Champion:
    modelo: object
    nombre: str
    version: str
    factor: float

    @property
    def etiqueta(self) -> str:
        return f"{self.nombre}/{self.version}"


def cargar_champion() -> Champion:
    """Carga la versión a la que apunta el alias y su factor de ajuste.

    Resuelve el alias a un número de versión y carga esa versión exacta, así el
    modelo y el factor salen siempre de la misma versión.
    """
    import mlflow

    nombre, alias = nombre_y_alias()
    try:
        mv = mlflow.MlflowClient().get_model_version_by_alias(nombre, alias)
    except mlflow.exceptions.MlflowException as e:
        raise FileNotFoundError(f"No hay modelo models:/{nombre}@{alias} en "
                                f"{mlflow.get_tracking_uri()}: {e}") from e
    if "factor_ajuste" not in mv.tags:
        raise ValueError(f"La versión {mv.version} de {nombre} no tiene el tag factor_ajuste")
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{mv.version}")
    entradas = modelo.metadata.get_input_schema().input_names()
    if entradas != columnas_salida():
        raise ValueError(f"La versión {mv.version} espera {entradas}, pero transformar.py produce "
                         f"{columnas_salida()}")
    return Champion(modelo=modelo, nombre=nombre, version=str(mv.version),
                    factor=float(mv.tags["factor_ajuste"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registra un run como champion.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--motivo", default="")
    args = parser.parse_args(argv)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("Define MLFLOW_TRACKING_URI")
    r = registrar_champion(args.run_id, args.motivo)
    print(f"models:/{r['nombre']}@{r['alias']} -> versión {r['version']}")
    for k, v in r["tags"].items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
