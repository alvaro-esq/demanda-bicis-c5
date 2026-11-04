import sqlite3

from bicis import almacenar, extraer, limpiar, predecir, transformar


def test_pipeline_completo_dos_veces_el_mismo_dia(champion):
    fecha = "2011-03-05"  # día con horas faltantes en la fuente
    for _ in range(2):
        ruta = None
        for paso in (extraer, limpiar, transformar, predecir, almacenar):
            ruta = paso.ejecutar(fecha)
    with sqlite3.connect(ruta) as con:
        n, horas, versiones = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT hr), GROUP_CONCAT(DISTINCT version_modelo) "
            "FROM predicciones WHERE fecha = ?", (fecha,)
        ).fetchone()
    assert (n, horas, versiones) == (24, 24, "demanda-bicis-c5/1")
