# demanda-bicis-c5 — contrato del proyecto

Cada mañana el pipeline toma una fecha `F` (AAAA-MM-DD) y predice cuántas
bicicletas se alquilarán (`cnt`) en cada una de las 24 horas de `F`.

Este proyecto es independiente de `demanda-bicis`: no comparte código, repositorio
de GitHub ni remoto DVC con él.

## Estructura

```
src/bicis/      paquete del pipeline (los cinco pasos + config.py)
scripts/        utilidades fuera del pipeline (crear_fuente.py, entrenar.py)
tests/          pruebas con pytest
dags/           DAGs de Airflow
models/         modelo entrenado (versionado con DVC, no con git)
airflow/        docker-compose.yaml, Dockerfile y .env.example de Airflow
fuente/         bicis.db: sistema de origen simulado (no se versiona)
data/           staging/ y salidas/ (no se versiona)
```

## Fuente de datos (solo lectura)

- `fuente/bicis.db`, tabla `registros_horarios`: las columnas de `hour.csv` del
  dataset Bike Sharing de UCI, con clave `(dteday, hr)` y `dteday` en texto ISO.
- La crea **únicamente** `scripts/crear_fuente.py` (`make fuente`). Simula el
  sistema de origen: el pipeline **nunca** escribe en ella. Hay que abrirla en solo
  lectura (`sqlite3.connect("file:...?mode=ro", uri=True)`). En Airflow además está
  montada como `:ro`.
- La fuente tiene horas faltantes en algunos días: no todos los días traen 24 filas.

## Configuración: rutas

Las rutas salen de variables de entorno. Si una variable no está definida, se usa el
valor por defecto de `params.yaml`. Las rutas relativas se resuelven desde la
carpeta que contiene `params.yaml`, no desde el directorio de trabajo.

| Variable            | Clave en params.yaml | Por defecto                    |
|---------------------|----------------------|--------------------------------|
| `BICIS_PARAMS`      | —                    | `params.yaml` de la raíz       |
| `BICIS_ZIP_ORIGEN`  | `rutas.zip_origen`   | `bike_sharing_dataset.zip`     |
| `BICIS_FUENTE_DB`   | `rutas.fuente_db`    | `fuente/bicis.db`              |
| `BICIS_STAGING_DIR` | `rutas.staging_dir`  | `data/staging`                 |
| `BICIS_SALIDAS_DB`  | `rutas.salidas_db`   | `data/salidas/predicciones.db` |
| `BICIS_MODELO`      | `rutas.modelo`       | `models/modelo.joblib`         |
| `BICIS_METRICAS`    | `rutas.metricas`     | `metrics.json`                 |

- La resolución de rutas vive en `src/bicis/config.py`, un módulo auxiliar que no es
  un paso del pipeline. Ningún módulo escribe rutas a mano ni lee `os.environ`
  directamente: todos usan `config`.
- Los hiperparámetros, la lista de features y el año de entrenamiento salen de
  `params.yaml`; no se repiten en el código.

## Pipeline: cinco módulos en `src/bicis/`

`extraer` → `limpiar` → `transformar` → `predecir` → `almacenar`

Cada módulo expone una sola función pública:

```python
def ejecutar(fecha: str) -> pathlib.Path: ...
```

- `fecha` es `"AAAA-MM-DD"`. Un formato inválido es un error.
- Lee **solo** el archivo del paso anterior en `data/staging/<fecha>/` (`extraer` lee
  la fuente y `predecir` lee además el modelo). Escribe su propio archivo y devuelve
  su ruta.
- Si falta el archivo del paso anterior, lanza `FileNotFoundError`. No lo regenera.
- Es idempotente: si se vuelve a ejecutar con la misma fecha, sobrescribe su salida.
- No hay estado compartido entre pasos fuera de esos archivos. Así Airflow puede
  correr cada paso como una tarea separada y reintentarla sola.

| Paso          | Lee                                   | Escribe                                  |
|---------------|---------------------------------------|------------------------------------------|
| `extraer`     | `fuente/bicis.db` (filas de `fecha`)  | `data/staging/<fecha>/extraido.csv`      |
| `limpiar`     | `extraido.csv`                        | `data/staging/<fecha>/limpio.csv`        |
| `transformar` | `limpio.csv`                          | `data/staging/<fecha>/features.csv`      |
| `predecir`    | `features.csv` + `models/modelo.joblib` | `data/staging/<fecha>/predicciones.csv` |
| `almacenar`   | `predicciones.csv`                    | `data/salidas/predicciones.db`           |

### Responsabilidades

- **extraer**: selecciona las filas con `dteday = fecha`, pero solo las columnas que
  se conocen por la mañana: `dteday`, `hr` y las features de `params.yaml`. **No
  extrae `cnt`, `casual` ni `registered`.** Si la fecha no tiene filas, escribe el
  archivo solo con el encabezado: quien falla es `limpiar`.
- **limpiar**: falla (`ValueError`) si el día no tiene filas. Revisa columnas (las que
  faltan y las prohibidas), tipos, rangos válidos (`hr` 0–23, `weathersit` 1–4, `temp`,
  `atemp`, `hum` y `windspeed` en [0, 1]) y exactamente 24 filas, `hr` 0–23. Una hora
  que falta se agrega con las columnas de calendario del día y el clima de la hora
  vecina más cercana, y se marca con `hora_imputada = 1`.
- **transformar**: deja `dteday`, `hr` y las columnas de `modelo.features`, en ese
  orden. Lanza un error si aparece alguna columna de `modelo.prohibidas`.
- **predecir**: carga el modelo y predice `cnt`. Recorta las predicciones a ≥ 0 y las
  redondea a entero. Escribe `fecha, hr, cnt_predicho`. **Nunca entrena.**
- **almacenar**: guarda en `data/salidas/predicciones.db`, tabla `predicciones`
  (`fecha TEXT, hr INTEGER, cnt_predicho INTEGER, creado_en TEXT`, PK
  `(fecha, hr)`). **Reemplaza el día completo**: en una sola transacción borra todas
  las filas de `fecha` e inserta las 24 nuevas. Devuelve la ruta de la base.

## Modelo

- `RandomForestRegressor` de scikit-learn para predecir `cnt`. Los hiperparámetros
  están en `params.yaml` → `modelo.random_forest`.
- Se entrena con **enero a noviembre de 2011** y el MAE se mide en **diciembre de
  2011** (`modelo.entrenamiento` y `modelo.validacion` en `params.yaml`), leyendo de la
  fuente en solo lectura. El modelo que se guarda es el entrenado con enero–noviembre.
- `scripts/entrenar.py` (`make entrenar`) produce `models/modelo.joblib`, versionado con
  DVC, y `metrics.json` (en git), con el MAE de diciembre y el de una referencia
  ingenua (promedio de entrenamiento por `workingday` y `hr`).
- **Nunca usar `casual` ni `registered`**: ni como feature, ni para derivar features,
  ni para calcular el objetivo. Son los componentes de `cnt` (`cnt = casual +
  registered`), así que usarlos filtraría el objetivo. Las pruebas deben verificarlo.
- Tampoco son features `instant` ni `dteday` (identificadores), ni `yr` (es constante
  en el año de entrenamiento, así que el modelo no puede aprender nada de ella).
- El clima del día se usa como si fuera el pronóstico disponible por la mañana.

## Versionado

- git: código, `params.yaml`, `metrics.json` y los archivos `.dvc`.
- DVC: `models/modelo.joblib` (`dvc add` + `dvc push` después de reentrenar). Remoto por defecto `local` →
  `/home/andres/dvc-storage/demanda-bicis-c5`.
- No se versionan: el `.zip`, `fuente/`, `data/` ni `.venv/` (todos se pueden
  regenerar).

## Airflow

- `airflow/docker-compose.yaml` (Airflow 3.3.2, LocalExecutor, Postgres) usa una
  imagen propia que instala `requirements.txt`.
- El repositorio se monta en `/opt/proyecto`, fuera de `/opt/airflow`, y encima
  `fuente/` se monta en `/opt/proyecto/fuente` en **solo lectura**. Los DAGs se leen de
  `/opt/proyecto/dags` y `PYTHONPATH` incluye `/opt/proyecto/src`.
- Lo que depende de la máquina va en `airflow/.env`, que no se versiona; la
  plantilla es `airflow/.env.example`.
- Cada paso del pipeline es una tarea del DAG y recibe `fecha = {{ ds }}`.
- `dags/prediccion_diaria.py`: diario a las 06:00 UTC del 2012-01-01 al 2012-01-31
  (`catchup=True`), cinco tareas en cadena que se pasan por XCom la ruta escrita. Solo
  orquesta; la lógica está en `src/bicis/`.
- Backfill (por ejemplo, de enero con 4 corridas a la vez): `airflow backfill create --dag-id
  prediccion_diaria --from-date 2012-01-01 --to-date 2012-01-31T23:59:59
  --max-active-runs 4`. El `--to-date` tiene que llegar hasta el final del día, porque
  la corrida es a las 06:00.

## Dependencias y comandos

- `requirements.txt`: dependencias del pipeline, con versiones fijas y compatibles
  con las restricciones de Airflow 3.3.2 (Python 3.12). `requirements-dev.txt` agrega
  `dvc` y `pytest` para el `.venv`.
- `make venv` · `make fuente` · `make test` · `make entrenar` ·
  `make pipeline FECHA=AAAA-MM-DD` · `make airflow-build` ·
  `make airflow-up` · `make airflow-down`

## Pruebas

- pytest con datos sintéticos en `tmp_path` (fixtures `entorno` y `modelo_entrenado` en
  `tests/conftest.py`; `pyproject.toml` agrega `src/` al path). Las pruebas no dependen de que exista
  `fuente/bicis.db` ni el modelo real. Para apuntar a otras rutas, usan las variables
  `BICIS_*`.
