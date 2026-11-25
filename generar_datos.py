"""
Genera el dataset SIMULADO de inspecciones de pavimento rígido (Huánuco).

IMPORTANTE: los datos son FICTICIOS. Se crean solo para entrenar y probar
la aplicación. Para un trabajo real se deben usar PCI obtenidos de una
evaluación técnica (método PCI, ASTM D6433).

Uso:  python generar_datos.py
Crea: data/inspecciones_pavimento.csv
"""
import os
import numpy as np
import pandas as pd

SEMILLA = 2026
CARPETA = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(CARPETA, "data", "inspecciones_pavimento.csv")

# tramo, vía, antigüedad (años) el 01/01/2025, tránsito
TRAMOS = [
    ("T-01", "Jr. Dos de Mayo", 0.2, "Bajo"),
    ("T-02", "Jr. General Prado", 0.6, "Bajo"),
    ("T-03", "Jr. Abtao", 1.4, "Medio"),
    ("T-04", "Av. Alameda de la República", 6.9, "Alto"),
    ("T-05", "Jr. Huánuco", 3.0, "Alto"),
    ("T-06", "Jr. Ayacucho", 2.2, "Medio"),
    ("T-07", "Jr. Huallayco", 4.1, "Medio"),
    ("T-08", "Jr. Leoncio Prado", 5.2, "Bajo"),
    ("T-09", "Av. Universitaria", 5.6, "Alto"),
    ("T-10", "Jr. San Martín", 7.6, "Medio"),
    ("T-11", "Av. Esteban Pavletich", 6.3, "Alto"),
    ("T-12", "Jr. Constitución", 8.0, "Bajo"),
]
FECHAS = pd.to_datetime(["2025-01-20", "2025-06-15", "2025-11-10", "2026-04-20", "2026-11-05"])
EFECTO_TRANSITO = {"Bajo": 3.0, "Medio": 0.0, "Alto": -4.0}


def patologia_para(pci, rng):
    """Asigna una falla y severidad coherentes con el PCI simulado."""
    if pci >= 90:
        return "Ninguna", "—"
    if pci >= 71:
        falla = rng.choice(["Grietas longitudinales", "Grietas transversales"])
        return falla, "Baja"
    if pci >= 41:
        falla = rng.choice(["Grietas de esquina", "Despostillamiento", "Escalonamiento", "Grietas transversales"])
        return falla, rng.choice(["Media", "Media", "Baja"])
    falla = rng.choice(["Losa dividida", "Escalonamiento", "Despostillamiento", "Grietas de esquina"])
    return falla, rng.choice(["Alta", "Alta", "Media"])


def generar():
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for tramo, via, edad_ini, transito in TRAMOS:
        for k, fecha in enumerate(FECHAS):
            dia = fecha + pd.Timedelta(days=int(rng.integers(-6, 7)) + int(TRAMOS.index((tramo, via, edad_ini, transito))))
            edad = round(edad_ini + (dia - pd.Timestamp("2025-01-01")).days / 365.25, 1)
            if edad < 1.0:          # tramos muy nuevos aún no se inspeccionan
                continue
            pci = 111 - 8.1 * edad + EFECTO_TRANSITO[transito] + rng.normal(0, 3.2)
            pci = int(round(np.clip(pci, 0, 100)))
            falla, sev = patologia_para(pci, rng)
            filas.append({"tramo": tramo, "via": via, "fecha": dia.strftime("%Y-%m-%d"),
                          "antiguedad": edad, "transito": transito, "falla": falla,
                          "severidad": sev, "pci": pci, "foto": "", "observaciones": "Registro simulado"})

    df = pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)

    # Errores intencionales para practicar la limpieza con Pandas:
    sucio = pd.concat([df, df.iloc[[5, 20]]], ignore_index=True)                   # 2 duplicados
    nulo = df.iloc[[12]].copy(); nulo["pci"] = np.nan; nulo["fecha"] = "2025-11-30"  # 1 PCI vacío
    fuera = df.iloc[[30]].copy(); fuera["pci"] = 135; fuera["fecha"] = "2026-04-30"  # 1 PCI fuera de 0–100
    sucio = pd.concat([sucio, nulo, fuera], ignore_index=True).sort_values("fecha").reset_index(drop=True)
    sucio.insert(0, "id", range(1, len(sucio) + 1))
    return sucio


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    datos = generar()
    datos.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"Dataset simulado creado: {SALIDA}  ({len(datos)} filas, incluye 4 filas con errores para limpiar)")
