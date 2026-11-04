# Imagen propia de Airflow para demanda-bicis-c5.
# Python 3.12 para coincidir con el .venv local.
ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

ARG AIRFLOW_VERSION
# Se fija apache-airflow en la misma línea para que pip no lo cambie al
# resolver las dependencias del pipeline.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
