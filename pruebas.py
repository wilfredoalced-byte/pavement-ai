"""
Pruebas automáticas del módulo modelo.py (sin interfaz).

Ejecutar:  python pruebas.py
"""
import pandas as pd

import modelo as M


def prueba_limpieza():
    crudo = M.cargar_datos()
    datos, r = M.limpiar(crudo)
    assert r.duplicados == 2 and r.nulos == 1 and r.fuera_de_rango == 1, r
    assert datos["pci"].between(0, 100).all()
    assert not datos.duplicated(subset=[c for c in M.COLUMNAS if c != "id"]).any()
    assert len(datos) >= 50, "Se requieren al menos 50 registros válidos"
    return datos


def prueba_division(datos):
    train, test = M.dividir_cronologicamente(datos)
    assert train["fecha"].max() <= test["fecha"].min(), "La prueba debe ser posterior al entrenamiento"
    assert len(test) == round(len(datos) * 0.2)


def prueba_modelo(datos):
    res = M.entrenar(datos)
    assert res.pendiente < 0, "El PCI debe bajar con la antigüedad"
    assert res.r2 > 0.8 and res.mae < 6, (res.r2, res.mae)
    assert 0 <= res.predecir(40) <= 100, "La predicción debe limitarse a 0–100"
    assert res.es_extrapolacion(15) and not res.es_extrapolacion(5)
    return res


def prueba_validacion():
    ok = {"tramo": "T-01", "via": "Jr. Dos de Mayo", "fecha": "2026-01-10", "antiguedad": 3, "transito": "Bajo",
          "falla": "Losa dividida", "severidad": "Alta", "pci": 40}
    assert M.validar_inspeccion(ok) == []
    assert M.validar_inspeccion({**ok, "pci": 120})
    assert M.validar_inspeccion({**ok, "pci": "abc"})
    assert M.validar_inspeccion({**ok, "antiguedad": -1})
    assert M.validar_inspeccion({**ok, "severidad": ""})
    assert M.validar_inspeccion({**ok, "fecha": "2999-01-01"})
    assert M.validar_inspeccion({**ok, "tramo": " "})


def prueba_condicion():
    assert M.condicion(90)["nombre"] == "Bueno"
    assert M.condicion(70)["nombre"] == "Regular"
    assert M.condicion(45)["nombre"] == "Pobre"
    assert M.condicion(5)["nombre"] == "Fallado"


if __name__ == "__main__":
    datos = prueba_limpieza()
    prueba_division(datos)
    res = prueba_modelo(datos)
    prueba_validacion()
    prueba_condicion()
    print(f"Todas las pruebas pasaron. R²={res.r2:.3f}  MAE={res.mae:.2f}  RMSE={res.rmse:.2f}")
