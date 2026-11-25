"""
Lógica de datos y Machine Learning de PAVEMENT AI.

- Carga y limpieza del dataset con Pandas.
- División cronológica en entrenamiento y prueba.
- Entrenamiento de LinearRegression (antigüedad -> PCI).
- Métricas R², MAE y RMSE en el conjunto de prueba.
- Predicción del PCI y clasificación de la condición.

Este módulo no depende de Streamlit, así se puede probar por separado:
    python modelo.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(CARPETA, "data", "inspecciones_pavimento.csv")
CARPETA_FOTOS = os.path.join(CARPETA, "data", "fotos")

COLUMNAS = ["id", "tramo", "via", "fecha", "antiguedad", "transito", "falla",
            "severidad", "pci", "foto", "observaciones"]
OBLIGATORIAS = ["tramo", "fecha", "antiguedad", "pci"]

PATOLOGIAS = ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
              "Losa dividida", "Despostillamiento", "Escalonamiento"]
SEVERIDADES = ["Baja", "Media", "Alta"]
TRANSITOS = ["Bajo", "Medio", "Alto"]

PROPORCION_PRUEBA = 0.20
EDAD_MAX = 50
PCI_MIN, PCI_MAX = 0, 100

# Escala de condición PCI (ASTM D6433): límite inferior, nombre, color fuerte, fondo, texto
CONDICIONES = [
    (86, "Bueno", "#2E8B57", "#E3F2E9", "#17623C"),
    (71, "Satisfactorio", "#6AA84F", "#EDF6E1", "#3D6417"),
    (56, "Regular", "#D9B23A", "#FBF3D6", "#735A06"),
    (41, "Pobre", "#E8902E", "#FDEBD6", "#8A4705"),
    (26, "Muy pobre", "#D9622B", "#FBE1D4", "#8E3510"),
    (11, "Grave", "#C0392B", "#F8DAD6", "#86231A"),
    (0, "Fallado", "#7B241C", "#EED3D0", "#6B1A12"),
]


def condicion(pci: float) -> dict:
    """Devuelve la condición PCI con sus colores."""
    for lim, nombre, color, fondo, texto in CONDICIONES:
        if pci >= lim:
            return {"nombre": nombre, "color": color, "fondo": fondo, "texto": texto}
    lim, nombre, color, fondo, texto = CONDICIONES[-1]
    return {"nombre": nombre, "color": color, "fondo": fondo, "texto": texto}


# --------------------------------------------------------------------- datos
def cargar_datos(ruta: str = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV crudo (puede traer errores; se limpian después)."""
    df = pd.read_csv(ruta, encoding="utf-8")
    return normalizar_columnas(df)


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    faltan = [c for c in OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas obligatorias: {', '.join(faltan)}")
    for c in COLUMNAS:
        if c not in df.columns:
            df[c] = "" if c not in ("id",) else range(1, len(df) + 1)
    return df[COLUMNAS]


@dataclass
class ResumenLimpieza:
    filas_leidas: int = 0
    duplicados: int = 0
    nulos: int = 0
    fuera_de_rango: int = 0
    finales: int = 0


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, ResumenLimpieza]:
    """Limpieza con Pandas: tipos, duplicados, nulos y valores fuera de rango."""
    r = ResumenLimpieza(filas_leidas=len(df))
    d = df.copy()
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
    d["antiguedad"] = pd.to_numeric(d["antiguedad"], errors="coerce")
    d["pci"] = pd.to_numeric(d["pci"], errors="coerce")
    for c in ("tramo", "via", "transito", "falla", "severidad", "foto", "observaciones"):
        d[c] = d[c].fillna("").astype(str).str.strip()

    antes = len(d)
    d = d.drop_duplicates(subset=[c for c in COLUMNAS if c != "id"])
    r.duplicados = antes - len(d)

    antes = len(d)
    d = d.dropna(subset=["fecha", "antiguedad", "pci"])
    d = d[d["tramo"] != ""]
    r.nulos = antes - len(d)

    antes = len(d)
    d = d[d["pci"].between(PCI_MIN, PCI_MAX) & d["antiguedad"].between(0, EDAD_MAX)]
    r.fuera_de_rango = antes - len(d)

    d = d.sort_values("fecha").reset_index(drop=True)
    d["condicion"] = d["pci"].apply(lambda p: condicion(p)["nombre"])
    r.finales = len(d)
    return d, r


# -------------------------------------------------------------------- modelo
@dataclass
class ResultadoModelo:
    modelo: LinearRegression
    pendiente: float
    intercepto: float
    r2: float
    mae: float
    rmse: float
    train: pd.DataFrame
    test: pd.DataFrame
    pred_test: np.ndarray
    edad_min: float
    edad_max: float
    fecha_corte: pd.Timestamp
    extra: dict = field(default_factory=dict)

    def predecir(self, antiguedad: float) -> float:
        """PCI estimado limitado al rango 0–100."""
        valor = self.predecir_crudo(antiguedad)
        return float(np.clip(valor, PCI_MIN, PCI_MAX))

    def predecir_crudo(self, antiguedad: float) -> float:
        """Valor que da la recta sin limitar (puede salir < 0 o > 100)."""
        return float(self.modelo.predict(pd.DataFrame({"antiguedad": [antiguedad]}))[0])

    def es_extrapolacion(self, antiguedad: float) -> bool:
        return antiguedad < self.edad_min or antiguedad > self.edad_max


def dividir_cronologicamente(d: pd.DataFrame, proporcion_prueba: float = PROPORCION_PRUEBA):
    """Las inspecciones más recientes van a prueba (no se mezclan fechas)."""
    d = d.sort_values("fecha").reset_index(drop=True)
    n_test = max(1, int(round(len(d) * proporcion_prueba)))
    return d.iloc[:-n_test].copy(), d.iloc[-n_test:].copy()


def entrenar(d: pd.DataFrame) -> ResultadoModelo:
    if len(d) < 10:
        raise ValueError("Se necesitan al menos 10 registros válidos para entrenar el modelo.")
    train, test = dividir_cronologicamente(d)

    modelo = LinearRegression()
    modelo.fit(train[["antiguedad"]], train["pci"])          # entrada: antigüedad, salida: PCI

    pred = modelo.predict(test[["antiguedad"]])
    return ResultadoModelo(
        modelo=modelo,
        pendiente=float(modelo.coef_[0]),
        intercepto=float(modelo.intercept_),
        r2=float(r2_score(test["pci"], pred)),
        mae=float(mean_absolute_error(test["pci"], pred)),
        rmse=float(np.sqrt(mean_squared_error(test["pci"], pred))),
        train=train, test=test, pred_test=pred,
        edad_min=float(train["antiguedad"].min()), edad_max=float(train["antiguedad"].max()),
        fecha_corte=test["fecha"].min(),
    )


# ------------------------------------------------------------- validaciones
def validar_inspeccion(datos: dict) -> list[str]:
    """Devuelve la lista de errores de un registro nuevo (vacía si es válido)."""
    errores = []
    if not str(datos.get("tramo", "")).strip():
        errores.append("Indique el tramo.")
    if not str(datos.get("via", "")).strip():
        errores.append("Indique el nombre de la vía.")
    try:
        a = float(datos.get("antiguedad"))
        if not 0 <= a <= EDAD_MAX:
            errores.append(f"La antigüedad debe estar entre 0 y {EDAD_MAX} años.")
    except (TypeError, ValueError):
        errores.append("La antigüedad debe ser un número.")
    try:
        p = float(datos.get("pci"))
        if not PCI_MIN <= p <= PCI_MAX:
            errores.append("El PCI observado debe estar entre 0 y 100.")
    except (TypeError, ValueError):
        errores.append("El PCI observado debe ser un número.")
    if datos.get("transito") not in TRANSITOS:
        errores.append("Seleccione el nivel de tránsito.")
    if datos.get("falla") not in PATOLOGIAS + ["Ninguna"]:
        errores.append("Seleccione el tipo de patología.")
    if datos.get("falla") != "Ninguna" and datos.get("severidad") not in SEVERIDADES:
        errores.append("Seleccione la severidad.")
    fecha = pd.to_datetime(datos.get("fecha"), errors="coerce")
    if pd.isna(fecha):
        errores.append("La fecha no es válida.")
    elif fecha > pd.Timestamp.today().normalize():
        errores.append("La fecha de inspección no puede ser futura.")
    return errores


def agregar_inspeccion(datos: dict, ruta: str = RUTA_DATOS) -> int:
    """Agrega una fila al CSV y devuelve su id."""
    crudo = pd.read_csv(ruta, encoding="utf-8") if os.path.exists(ruta) else pd.DataFrame(columns=COLUMNAS)
    nuevo_id = int(pd.to_numeric(crudo["id"], errors="coerce").max() or 0) + 1 if len(crudo) else 1
    fila = {c: datos.get(c, "") for c in COLUMNAS}
    fila["id"] = nuevo_id
    pd.concat([crudo, pd.DataFrame([fila])], ignore_index=True).to_csv(ruta, index=False, encoding="utf-8")
    return nuevo_id


if __name__ == "__main__":
    crudo = cargar_datos()
    datos, resumen = limpiar(crudo)
    res = entrenar(datos)
    print(resumen)
    print(f"Entrenamiento: {len(res.train)}  Prueba: {len(res.test)} (desde {res.fecha_corte:%d/%m/%Y})")
    print(f"PCI = {res.intercepto:.2f} {res.pendiente:+.2f} x antigüedad")
    print(f"R² = {res.r2:.3f}   MAE = {res.mae:.2f}   RMSE = {res.rmse:.2f}")
    for a in (2, 5, 8, 15):
        print(f"  {a:>2} años -> PCI {res.predecir(a):5.1f} ({condicion(res.predecir(a))['nombre']})"
              f"{'  [extrapolación]' if res.es_extrapolacion(a) else ''}")
