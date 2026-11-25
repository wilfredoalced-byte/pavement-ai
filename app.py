"""
PAVEMENT AI – Predicción del deterioro del pavimento rígido mediante regresión lineal.
SENATI · Inteligencia Artificial · Reto individual · Huánuco, Perú

Ejecutar:  python -m streamlit run app.py
Alumno: [Nombre del alumno]

Los datos incluidos son SIMULADOS. El modelo no reemplaza la metodología PCI
ni una inspección estructural.
"""
from __future__ import annotations

import io
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

import modelo as M
import ui
from ui import esc, md

st.set_page_config(page_title="PAVEMENT AI", page_icon=":material/add_road:", layout="wide",
                   initial_sidebar_state="expanded")
ui.aplicar_estilos()

PAGINAS = ["Dashboard", "Datos y modelo", "Simulador PCI", "Registro de patologías"]
TAM_MAX_FOTO = 5 * 1024 * 1024


# ======================================================================
# Estado: datos y modelo entrenado
# ======================================================================
def iniciar_estado():
    ss = st.session_state
    if "crudo" not in ss:
        if not os.path.exists(M.RUTA_DATOS):
            import generar_datos
            os.makedirs(os.path.dirname(M.RUTA_DATOS), exist_ok=True)
            generar_datos.generar().to_csv(M.RUTA_DATOS, index=False, encoding="utf-8")
        ss.crudo = M.cargar_datos()
        ss.origen = os.path.basename(M.RUTA_DATOS)
        ss.version = 0
    if ss.get("version_entrenada") != ss.version:
        entrenar_modelo()


def entrenar_modelo():
    ss = st.session_state
    datos, resumen = M.limpiar(ss.crudo)
    ss.datos, ss.resumen = datos, resumen
    try:
        ss.res = M.entrenar(datos)
        ss.error_modelo = ""
    except ValueError as e:
        ss.res, ss.error_modelo = None, str(e)
    ss.version_entrenada = ss.version
    ss.entrenado_en = datetime.now()


def recargar_desde_disco():
    st.session_state.crudo = M.cargar_datos()
    st.session_state.origen = os.path.basename(M.RUTA_DATOS)
    st.session_state.version += 1


def a_csv(df: pd.DataFrame) -> bytes:
    d = df.copy()
    if "fecha" in d and pd.api.types.is_datetime64_any_dtype(d["fecha"]):
        d["fecha"] = d["fecha"].dt.strftime("%Y-%m-%d")
    return d.to_csv(index=False).encode("utf-8-sig")       # utf-8-sig: Excel muestra bien las tildes


def a_excel(df: pd.DataFrame) -> bytes:
    d = df.copy()
    if "fecha" in d and pd.api.types.is_datetime64_any_dtype(d["fecha"]):
        d["fecha"] = d["fecha"].dt.date
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        d.to_excel(w, index=False, sheet_name="Inspecciones")
    return buf.getvalue()


# ======================================================================
# Barra lateral
# ======================================================================
def barra_lateral() -> str:
    res = st.session_state.res
    with st.sidebar:
        md("""<div class="pv-side-brand">
          <span style="width:42px;height:42px;border-radius:12px;background:#0A2747;display:flex;align-items:center;justify-content:center">
          <svg aria-hidden="true" width="26" height="26" viewBox="0 0 26 26" fill="none"><rect x="2" y="7" width="22" height="12" rx="2" fill="#51667E"/>
          <path d="M2 13h22" stroke="#FFF" stroke-width="1.6" stroke-dasharray="3 2.5"/><path d="M9 7v12M17 7v12" stroke="#0A2747" stroke-width="1.6"/></svg></span>
          <div><b>PAVEMENT AI</b><small>Análisis vial · Huánuco</small></div></div>""")
        pagina = st.radio("MENÚ", PAGINAS, key="pagina")
        if res:
            md(f"""<div class="pv-side-box">
              <span style="display:flex;align-items:center;gap:8px;color:#17623C;font-weight:700;font-size:13px">
              <span class="pv-dot" style="width:8px;height:8px;background:#2E8B57"></span>Modelo entrenado</span>
              <code>PCI = {res.intercepto:.2f} {res.pendiente:+.2f}·t</code>
              R² {res.r2:.2f} · {len(res.train)} registros · {st.session_state.entrenado_en:%d/%m/%Y %H:%M}</div>""")
        else:
            ui.aviso(esc(st.session_state.error_modelo), "error")
        md("""<div style="display:flex;align-items:center;gap:10px;margin-top:18px">
          <span style="width:36px;height:36px;border-radius:50%;background:#0B74C8;color:#FFF;font-size:13px;font-weight:700;display:flex;align-items:center;justify-content:center">AL</span>
          <div style="font-size:12px;line-height:1.35"><b style="font-size:13px">[Wilfredo alcedo santiago]</b><br><span style="color:#4A5E78">SENATI · Reto individual</span></div></div>""")
    return pagina


# ======================================================================
# 1. Dashboard
# ======================================================================
def pagina_dashboard():
    datos, res = st.session_state.datos, st.session_state.res
    ui.encabezado("CENTRO DE ANÁLISIS VIAL", "Dashboard de la red vial",
                  "Estado del pavimento rígido de Huánuco a partir de las inspecciones registradas y del modelo de regresión lineal.")

    # ---- filtros
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 0.8], vertical_alignment="bottom")
    tramos = ["Todos los tramos"] + [f"{t} · {v}" for t, v in datos[["tramo", "via"]].drop_duplicates("tramo").sort_values("tramo").values]
    tramo_sel = c1.selectbox("Tramo", tramos)
    fmin, fmax = datos["fecha"].min().date(), datos["fecha"].max().date()
    periodo = c2.date_input("Periodo", (fmin, fmax), min_value=fmin, max_value=max(fmax, date.today()), format="DD/MM/YYYY")
    transito = c3.selectbox("Tránsito", ["Todos"] + M.TRANSITOS)

    f = datos.copy()
    if tramo_sel != "Todos los tramos":
        f = f[f["tramo"] == tramo_sel.split(" · ")[0]]
    if isinstance(periodo, tuple) and len(periodo) == 2:
        f = f[(f["fecha"].dt.date >= periodo[0]) & (f["fecha"].dt.date <= periodo[1])]
    if transito != "Todos":
        f = f[f["transito"] == transito]
    c4.download_button("Exportar reporte", a_csv(f.drop(columns=["condicion"])), "reporte_inspecciones.csv",
                       "text/csv", use_container_width=True, icon=":material/download:")

    if f.empty:
        ui.aviso("No hay inspecciones con esos filtros. Cambie el tramo, el periodo o el tránsito.", "warn")
        return

    # ---- indicadores
    ultimas = f.sort_values("fecha").groupby("tramo").tail(1)
    criticos = ultimas[ultimas["pci"] < 56].sort_values("pci")
    prom = f["pci"].mean()
    cp = M.condicion(prom)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.kpi("doc", "Inspecciones registradas", len(f), f"{f['tramo'].nunique()} tramos", "#E7F1FC", "#0B4F8A")
    with k2:
        ui.kpi("gauge", "PCI promedio", f"{prom:.1f}", cp["nombre"], cp["fondo"], cp["texto"], cp["fondo"], cp["texto"])
    with k3:
        ui.kpi("alert", "Tramos en condición pobre o peor (última inspección)", len(criticos),
               "Priorizar" if len(criticos) else "Sin alertas", "#F8DAD6", "#86231A", "#F8DAD6", "#86231A", "#86231A")
    with k4:
        if res:
            ui.kpi("target", "R² del modelo en prueba", f"{res.r2:.2f}", f"MAE {res.mae:.1f}", "#E3F2E9", "#17623C", "#E3F2E9", "#17623C")

    # ---- gráficos
    g1, g2 = st.columns([3, 2])
    with g1, st.container(border=True, key="card_disp"):
        ui.titulo_tarjeta("chart", "PCI observado vs antigüedad", "Cada punto es una inspección; el fondo marca la condición PCI")
        if res:
            st.plotly_chart(ui.grafico_dispersion(f, res), use_container_width=True, config=ui.CONFIG_PLOTLY)
        md('<div class="pv-legend"><span><span class="pv-dot" style="width:9px;height:9px;background:#1F6FB8"></span>Inspecciones</span>'
           '<span><span style="width:18px;border-top:3px dashed #E07B00"></span>Recta de regresión</span></div>')
    with g2, st.container(border=True, key="card_dona"):
        ui.titulo_tarjeta("pie", "Distribución por condición", "Escala PCI de 0 a 100")
        fig, conteo = ui.grafico_dona(f)
        d1, d2 = st.columns([1, 1], vertical_alignment="center")
        d1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        filas = ""
        total = conteo.sum()
        for nombre, n in conteo.items():
            c = next(x for x in M.CONDICIONES if x[1] == nombre)
            filas += (f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;margin:6px 0">'
                      f'<span style="width:10px;height:10px;border-radius:3px;background:{c[2]}"></span>'
                      f'<span style="flex:1">{nombre}</span><b>{int(n)}</b>'
                      f'<span style="color:#4A5E78;width:38px;text-align:right">{n / total * 100:.0f}%</span></div>')
        with d2:
            md(filas)
        if len(criticos):
            lista = ", ".join(esc(t) for t in criticos["tramo"])
            ui.aviso(f"<b>{len(criticos)} tramo(s)</b> en condición pobre o peor: programar mantenimiento en {lista}.", "naranja")
        else:
            ui.aviso("Ningún tramo está en condición pobre o peor en su última inspección.", "ok", "check")

    # ---- últimas inspecciones
    with st.container(border=True, key="card_tabla"):
        ui.titulo_tarjeta("table", "Últimas inspecciones", "Ordenadas por fecha, de la más reciente a la más antigua")
        filas = ""
        for _, r in f.sort_values("fecha", ascending=False).head(8).iterrows():
            c = M.condicion(r.pci)
            filas += (f"<tr><td style='font-weight:700;color:#0B4F8A'>{esc(r.tramo)}</td><td>{esc(r.via)}</td>"
                      f"<td style='color:#4A5E78'>{r.fecha:%d/%m/%Y}</td><td style='text-align:center'>{r.antiguedad:.1f}</td>"
                      f"<td>{esc(r.transito)}</td><td>{esc(r.falla)}</td><td>{ui.etiqueta_severidad(r.severidad)}</td>"
                      f"<td><b style='display:inline-block;width:28px'>{r.pci:.0f}</b><span class='pv-bar'><span style='width:{r.pci:.0f}%;background:{c['color']}'></span></span></td>"
                      f"<td>{ui.etiqueta_condicion(r.pci)}</td></tr>")
        md(f"""<div class="pv-tablebox"><table class="pv-table"><thead><tr><th>TRAMO</th><th>VÍA</th><th>FECHA</th><th>ANTIG. (AÑOS)</th>
           <th>TRÁNSITO</th><th>PATOLOGÍA</th><th>SEVERIDAD</th><th>PCI</th><th>CONDICIÓN</th></tr></thead><tbody>{filas}</tbody></table></div>""")
        st.write("")
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button("Exportar CSV", a_csv(f.drop(columns=["condicion"])), "inspecciones_filtradas.csv", "text/csv",
                           type="primary", use_container_width=True, icon=":material/download:")
        b2.download_button("Exportar Excel", a_excel(f.drop(columns=["condicion"])), "inspecciones_filtradas.xlsx",
                           use_container_width=True, icon=":material/table:")


# ======================================================================
# 2. Datos y modelo
# ======================================================================
def pagina_modelo():
    ss = st.session_state
    datos, res, r = ss.datos, ss.res, ss.resumen
    ui.encabezado("ENTRENAMIENTO", "Datos y modelo de regresión",
                  "Carga y limpieza con Pandas, división cronológica, entrenamiento de LinearRegression (antigüedad → PCI) y métricas en prueba.",
                  f'<span class="pv-chip-ok">Entrenado {ss.entrenado_en:%d/%m/%Y · %H:%M}</span>')

    with st.container(border=True, key="card_flujo"):
        a, b = st.columns([3, 2], vertical_alignment="center")
        with a:
            ui.titulo_tarjeta("layers", "Flujo de entrenamiento", f"Fuente: {esc(ss.origen)}")
        with b:
            archivo = st.file_uploader("Subir otro CSV (columnas: tramo, fecha, antiguedad, pci, …)", type=["csv"],
                                       key=f"csv_{ss.version}")
        if archivo is not None:
            try:
                nuevo = M.normalizar_columnas(pd.read_csv(archivo))
                if len(nuevo) == 0:
                    raise ValueError("El archivo está vacío.")
                ss.crudo, ss.origen = nuevo, archivo.name
                ss.version += 1
                st.rerun()
            except Exception as e:  # el archivo lo sube el usuario: se informa sin detener la app
                ui.aviso(f"No se pudo usar el archivo: {esc(e)}", "error")
        pasos = [("Cargar CSV", f"{r.filas_leidas} filas"), ("Limpiar", f"{r.finales} válidas"),
                 ("Dividir", f"{len(res.train)} / {len(res.test)}" if res else "—"),
                 ("Entrenar", "LinearRegression"), ("Evaluar", "R² · MAE · RMSE")]
        html = ""
        for i, (t, s) in enumerate(pasos):
            ok = res is not None or i < 2
            html += (f'<div class="pv-step"><span class="pv-step-dot" style="background:{"#2E8B57" if ok else "#9AAABD"}">'
                     f'{ui.icono("check", 18, "#FFF", 2.6)}</span><b>{i + 1}. {t}</b><small>{s}</small></div>')
            if i < len(pasos) - 1:
                html += '<span class="pv-step-line"></span>'
        md(f'<div class="pv-steps">{html}</div>')
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Reentrenar modelo", type="primary", use_container_width=True, icon=":material/refresh:"):
            entrenar_modelo()
            st.toast("Modelo reentrenado con los datos actuales.", icon=":material/check_circle:")
            st.rerun()
        if ss.origen != os.path.basename(M.RUTA_DATOS) and c2.button("Volver al CSV original", use_container_width=True):
            recargar_desde_disco()
            st.rerun()

    if res is None:
        ui.aviso(esc(ss.error_modelo), "error")
        return

    p1, p2 = st.columns([2, 1])
    with p1, st.container(border=True, key="card_prev"):
        ui.titulo_tarjeta("table", "Vista previa del dataset limpio", f"{len(datos)} registros · {datos.shape[1] - 1} columnas")
        vista = datos.drop(columns=["foto", "observaciones"]).copy()
        st.dataframe(vista, hide_index=True, use_container_width=True, height=320, column_config={
            "fecha": st.column_config.DateColumn("fecha", format="DD/MM/YYYY"),
            "pci": st.column_config.ProgressColumn("pci", min_value=0, max_value=100, format="%d"),
            "antiguedad": st.column_config.NumberColumn("antigüedad", format="%.1f años")})
        st.download_button("Descargar dataset limpio (CSV)", a_csv(datos.drop(columns=["condicion"])),
                           "dataset_limpio.csv", "text/csv", icon=":material/download:")
    with p2, st.container(border=True, key="card_limpieza"):
        ui.titulo_tarjeta("check", "Limpieza y validación", "Reglas aplicadas con Pandas", "#E3F2E9", "#17623C")
        md(f"""<div class="pv-row"><span>Filas leídas</span><b>{r.filas_leidas}</b></div>
        <div class="pv-row"><span>Duplicados eliminados</span><b style="color:#8A4705">−{r.duplicados}</b></div>
        <div class="pv-row"><span>Filas con datos vacíos</span><b style="color:#8A4705">−{r.nulos}</b></div>
        <div class="pv-row"><span>PCI fuera de 0–100 o antigüedad inválida</span><b style="color:#8A4705">−{r.fuera_de_rango}</b></div>
        <div class="pv-row" style="border:0"><span>Registros finales</span><b style="color:#17623C">{r.finales}</b></div>
        <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin-top:8px">
        <span style="flex:{len(res.train)};background:#1F6FB8"></span><span style="flex:{len(res.test)};background:#E07B00"></span></div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:#3A4F69;margin-top:6px;gap:8px">
        <span><span style="color:#1F6FB8">■</span> Entrenamiento {len(res.train)}</span>
        <span><span style="color:#E07B00">■</span> Prueba {len(res.test)} (desde {res.fecha_corte:%d/%m/%Y})</span></div>
        <p style="font-size:12px;color:#4A5E78;margin:10px 0 0">División cronológica: las inspecciones más recientes se usan para probar el modelo, como ocurriría con datos futuros.</p>""")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ui.kpi("target", "R² en prueba: proporción de la variación del PCI que explica el modelo", f"{res.r2:.2f}",
               f"{res.r2 * 100:.0f} %", "#E3F2E9", "#17623C", "#E3F2E9", "#17623C", "#17623C")
    with m2:
        ui.kpi("gauge", "MAE: error absoluto medio, en puntos PCI", f"{res.mae:.2f}", "± puntos", "#E7F1FC", "#0B4F8A")
    with m3:
        ui.kpi("chart", "RMSE: penaliza más los errores grandes", f"{res.rmse:.2f}", "puntos PCI", "#E7F1FC", "#0B4F8A")
    with m4:
        ui.kpi("alert", "Pendiente: puntos PCI que se pierden por año", f"{res.pendiente:.2f}", "por año",
               "#FBE1D4", "#8E3510", "#FBE1D4", "#8E3510", "#8E3510")

    g1, g2 = st.columns(2)
    with g1, st.container(border=True, key="card_rvp"):
        ui.titulo_tarjeta("chart", "PCI real vs predicho (prueba)", "Las líneas naranjas son los residuos; la punteada es la predicción perfecta")
        st.plotly_chart(ui.grafico_real_vs_pred(res), use_container_width=True, config=ui.CONFIG_PLOTLY)
    with g2, st.container(border=True, key="card_interp"):
        ui.titulo_tarjeta("book", "Interpretación del modelo")
        md(f"""<div class="pv-eq"><span class="k">ECUACIÓN AJUSTADA</span>
           <span class="f">PCI = {res.intercepto:.2f} {'−' if res.pendiente < 0 else '+'} {abs(res.pendiente):.2f} × t</span>
           <span style="font-size:12px;color:#C9D8EA">t = antigüedad en años · rango de entrenamiento {res.edad_min:.1f}–{res.edad_max:.1f} años</span></div>
           <div style="font-size:14px;line-height:1.6;margin-top:12px">
           <p><b>Pendiente ({res.pendiente:.2f}):</b> por cada año de servicio el PCI {'baja' if res.pendiente < 0 else 'sube'}, en promedio, {abs(res.pendiente):.2f} puntos.</p>
           <p><b>Intercepto ({res.intercepto:.2f}):</b> valor teórico a los 0 años. {'Supera 100, por eso la app limita el PCI a 0–100.' if res.intercepto > 100 else ''}</p>
           <p><b>MAE {res.mae:.2f}:</b> en promedio, la predicción se aleja {res.mae:.1f} puntos del PCI observado en las inspecciones de prueba.</p>
           <p><b>R² {res.r2:.2f}:</b> el modelo explica el {res.r2 * 100:.0f} % de la variación del PCI en prueba.</p></div>""")
        ui.aviso("<b>Limitación:</b> la antigüedad sola no basta. El tránsito, el drenaje, la calidad del concreto, el clima y el "
                 "mantenimiento también influyen en el PCI.", "warn")

    with st.expander("Preguntas de sustentación"):
        st.markdown(f"""
**¿Qué representa la pendiente?** Es el cambio promedio del PCI por cada año de antigüedad. Aquí vale **{res.pendiente:.2f}**: cada año el pavimento pierde unos {abs(res.pendiente):.1f} puntos PCI.

**¿Qué significa MAE?** *Mean Absolute Error*, el error absoluto medio: el promedio de |PCI real − PCI predicho|. Un MAE de **{res.mae:.2f}** indica que, en promedio, el modelo se equivoca en {res.mae:.1f} puntos PCI.

**¿Por qué no basta la antigüedad para determinar el PCI?** Dos losas con la misma edad pueden estar muy distintas según el tránsito pesado, el drenaje, el tipo de suelo, la calidad del concreto y el mantenimiento. El modelo resume una tendencia promedio; el PCI real debe medirse en campo con la metodología PCI.

**¿Qué riesgos tiene extrapolar?** El modelo solo vio antigüedades entre {res.edad_min:.1f} y {res.edad_max:.1f} años. Fuera de ese rango la recta puede dar valores imposibles (PCI negativo o mayor que 100) y el deterioro real no es necesariamente lineal, por lo que la predicción pierde confiabilidad.
""")


# ======================================================================
# 3. Simulador PCI
# ======================================================================
def _sync(origen, destino):
    st.session_state[destino] = st.session_state[origen]


def pagina_simulador():
    ss = st.session_state
    datos, res = ss.datos, ss.res
    ui.encabezado("PREDICCIÓN", "Simulador PCI",
                  "Estima el PCI actual y futuro de un tramo con el modelo entrenado. Es una estimación: no reemplaza la evaluación técnica PCI.")
    if res is None:
        ui.aviso(esc(ss.error_modelo), "error")
        return
    for k, v in (("edad_num", 5.0), ("edad_sl", 5.0), ("hor_num", 3), ("hor_sl", 3)):
        ss.setdefault(k, v)
    ss.setdefault("sim", (5.0, 3))

    izq, der = st.columns([1, 2])
    with izq, st.container(border=True, key="card_param"):
        ui.titulo_tarjeta("sliders", "Parámetros del tramo")
        st.number_input("Antigüedad actual (años)", 0.0, 30.0, step=0.5, key="edad_num", on_change=_sync, args=("edad_num", "edad_sl"))
        st.slider("Antigüedad", 0.0, 30.0, step=0.5, key="edad_sl", on_change=_sync, args=("edad_sl", "edad_num"), label_visibility="collapsed")
        st.number_input("Horizonte de proyección (años)", 1, 20, step=1, key="hor_num", on_change=_sync, args=("hor_num", "hor_sl"))
        st.slider("Horizonte", 1, 20, step=1, key="hor_sl", on_change=_sync, args=("hor_sl", "hor_num"), label_visibility="collapsed")
        if st.button("Ejecutar predicción", type="primary", use_container_width=True, icon=":material/query_stats:"):
            edad, hor = ss.edad_num, ss.hor_num
            errores = []
            if edad is None or not 0 <= edad <= 30:
                errores.append("La antigüedad debe estar entre 0 y 30 años.")
            if hor is None or not 1 <= hor <= 20:
                errores.append("El horizonte debe estar entre 1 y 20 años.")
            if errores:
                ss.sim_error = " ".join(errores)
            else:
                ss.sim, ss.sim_error = (float(edad), int(hor)), ""
        if ss.get("sim_error"):
            ui.aviso(esc(ss.sim_error), "error")
        st.caption(f"Rango de entrenamiento del modelo: {res.edad_min:.1f} a {res.edad_max:.1f} años.")

    edad, hor = ss.sim
    final = edad + hor
    p_hoy, p_fut = res.predecir(edad), res.predecir(final)
    c_hoy, c_fut = M.condicion(p_hoy), M.condicion(p_fut)
    with der, st.container(border=True, key="card_result"):
        ui.titulo_tarjeta("gauge", "Resultado de la predicción", f"Tramo de {edad:g} años proyectado a {final:g} años")
        g, t = st.columns([1.15, 1], vertical_alignment="center")
        with g:
            st.plotly_chart(ui.grafico_velocimetro(p_fut), use_container_width=True, config={"displayModeBar": False})
            md('<p style="text-align:center;font-size:13px;color:#4A5E78;margin:-8px 0 0">PCI futuro estimado</p>')
        with t:
            md(f"""<div class="pv-mini" style="display:flex;justify-content:space-between;align-items:center">
               <div><small>PCI actual estimado</small><b>{p_hoy:.1f}</b></div><span style="font-size:12px;font-weight:600;color:{c_hoy['texto']}">{c_hoy['nombre']}</span></div>
               <div class="pv-mini" style="background:{c_fut['fondo']}"><small style="color:{c_fut['texto']}">Condición futura</small><b style="color:{c_fut['texto']}">{c_fut['nombre']}</b></div>
               <div class="pv-mini"><small>Pérdida estimada</small><b style="font-size:20px;color:#8E3510">−{p_hoy - p_fut:.1f} puntos en {hor} años</b></div>""")
        if res.es_extrapolacion(final) or res.es_extrapolacion(edad):
            crudo = res.predecir_crudo(final)
            extra = f" La recta da {crudo:.1f} y se limita a 0–100." if not 0 <= crudo <= 100 else ""
            ui.aviso(f"<b>Extrapolación:</b> el cálculo usa edades fuera del rango de entrenamiento "
                     f"({res.edad_min:.1f}–{res.edad_max:.1f} años). El resultado es poco confiable.{extra}", "warn")

    with st.container(border=True, key="card_proy"):
        ui.titulo_tarjeta("chart", "Proyección del deterioro", "La línea negra une el PCI de hoy con el PCI futuro")
        st.plotly_chart(ui.grafico_proyeccion(datos, res, edad, final), use_container_width=True, config=ui.CONFIG_PLOTLY)
        md('<div class="pv-legend"><span><span class="pv-dot" style="width:9px;height:9px;background:#1F6FB8"></span>Historial</span>'
           '<span><span style="width:18px;border-top:3px dashed #E07B00"></span>Modelo (PCI limitado a 0–100)</span>'
           '<span><span style="width:10px;height:10px;border-radius:50%;border:3px solid #0B74C8"></span>Hoy</span>'
           '<span><span class="pv-dot" style="width:11px;height:11px;background:#E5533D"></span>Predicción</span></div>')

    # ---- tres escenarios
    st.write("")
    h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
    h1.markdown("#### Tres escenarios simulados")
    h1.caption("Comparan tramos de distinta antigüedad y muestran el riesgo de extrapolar.")
    escenarios = [("A", "Tramo nuevo", 2.0, 3), ("B", "Tramo intermedio", 5.0, 3), ("C", "Tramo antiguo", 10.0, 5)]
    filas = []
    cols = st.columns(3)
    for col, (tag, nombre, a, h) in zip(cols, escenarios):
        hoy, fut, crudo = res.predecir(a), res.predecir(a + h), res.predecir_crudo(a + h)
        extrap = res.es_extrapolacion(a + h)
        filas.append({"escenario": tag, "descripcion": nombre, "antiguedad": a, "horizonte": h, "edad_final": a + h,
                      "pci_actual": round(hoy, 1), "pci_futuro": round(fut, 1), "pci_sin_limitar": round(crudo, 1),
                      "condicion_futura": M.condicion(fut)["nombre"], "extrapolacion": "Sí" if extrap else "No"})
        c = M.condicion(fut)
        nota = (f'<div class="pv-alert" style="background:#F8DAD6;color:#86231A;padding:8px 10px;font-size:12px">{ui.icono("alert", 16)}'
                f'<div>Extrapolación: {a + h:g} años está fuera del rango {res.edad_min:.0f}–{res.edad_max:.0f}.'
                f'{f" La recta da {crudo:.1f}." if not 0 <= crudo <= 100 else ""}</div></div>' if extrap else
                f'<div class="pv-alert" style="background:#E3F2E9;color:#17623C;padding:8px 10px;font-size:12px">{ui.icono("check", 16)}<div>Dentro del rango de entrenamiento</div></div>')
        with col:
            md(f"""<div class="pv-card" style="display:flex;flex-direction:column;gap:12px">
               <div style="display:flex;justify-content:space-between;align-items:center"><span style="font-size:12px;font-weight:700;letter-spacing:1px;color:#0B5FA5">ESCENARIO {tag}</span>{ui.etiqueta_condicion(fut)}</div>
               <b style="font-size:16px">{nombre}</b>
               <span style="font-size:13px;color:#3A4F69">Antigüedad <b style="color:#10243E">{a:g} años</b> · Horizonte <b style="color:#10243E">+{h}</b></span>
               <div style="display:flex;align-items:center;gap:14px"><div><small style="font-size:12px;color:#4A5E78">PCI hoy</small><br><b style="font-size:24px">{hoy:.1f}</b></div>
               <span style="color:#8395AB;font-size:20px">→</span><div><small style="font-size:12px;color:#4A5E78">PCI a {a + h:g} años</small><br><b style="font-size:24px;color:{c['texto']}">{fut:.1f}</b></div></div>
               {nota}</div>""")
    st.write("")
    st.download_button("Exportar escenarios (CSV)", a_csv(pd.DataFrame(filas)), "escenarios_pci.csv", "text/csv",
                       icon=":material/download:")
    with st.expander("Extrapolación y limitaciones del modelo"):
        st.markdown(f"""
- El modelo se entrenó con antigüedades entre **{res.edad_min:.1f} y {res.edad_max:.1f} años**. Dentro de ese rango la predicción es razonable (MAE ≈ {res.mae:.1f} puntos).
- **Extrapolar** es usar el modelo fuera de ese rango. La recta sigue bajando sin límite y puede dar PCI negativos; por eso la app limita el resultado a 0–100 y muestra una advertencia.
- El deterioro real **no es lineal**: suele acelerarse cuando aparecen fallas estructurales, y el mantenimiento puede recuperar el PCI.
- Solo se usa la antigüedad como entrada. Tránsito, drenaje, clima y calidad del concreto no están en el modelo.
- El PCI observado debe venir de una evaluación técnica; esta app **no reemplaza** la metodología PCI ni una inspección estructural.
""")


# ======================================================================
# 4. Registro de patologías
# ======================================================================
NUEVO_TRAMO = "Nuevo tramo…"


def _limpiar_formulario():
    ss = st.session_state
    valores = {"r_pci": None, "r_falla": M.PATOLOGIAS[0], "r_sev": None, "r_obs": "", "r_codigo": "", "r_via": ""}
    for k, v in valores.items():
        ss[k] = v
    for k in [k for k in ss.keys() if str(k).startswith(("r_edad_", "r_tran_"))]:
        del ss[k]
    st.session_state.r_foto_key = st.session_state.get("r_foto_key", 0) + 1


def _guardar(datos_form, foto, codigo_repetido=False):
    """Callback del botón Guardar: valida, guarda la foto y agrega la fila al CSV."""
    ss = st.session_state
    errores = M.validar_inspeccion(datos_form)
    if codigo_repetido:
        errores.append(f"El tramo {datos_form['tramo']} ya existe; selecciónelo de la lista.")
    if foto is not None:
        if foto.size > TAM_MAX_FOTO:
            errores.append("La foto supera 5 MB.")
        elif os.path.splitext(foto.name)[1].lower() not in (".jpg", ".jpeg", ".png"):
            errores.append("La foto debe ser JPG o PNG.")
    if errores:
        ss.r_errores = errores
        return
    if foto is not None:
        os.makedirs(M.CARPETA_FOTOS, exist_ok=True)
        ext = os.path.splitext(foto.name)[1].lower()
        nombre = f"insp_{datetime.now():%Y%m%d_%H%M%S}{ext}"
        with open(os.path.join(M.CARPETA_FOTOS, nombre), "wb") as f:
            f.write(foto.getbuffer())
        datos_form["foto"] = nombre
    nuevo_id = M.agregar_inspeccion(datos_form)
    ss.r_ok = f"Inspección #{nuevo_id} guardada en {datos_form['tramo']}. El modelo se reentrenó con el nuevo registro."
    recargar_desde_disco()
    _limpiar_formulario()


def pagina_registro():
    ss = st.session_state
    datos, res = ss.datos, ss.res
    ui.encabezado("INSPECCIÓN", "Registro de patologías",
                  "Registra cada inspección del pavimento rígido: ubicación, falla, severidad, PCI observado y evidencia fotográfica.")
    if ss.get("r_ok"):
        st.success(ss.r_ok, icon=":material/check_circle:")
        ss.r_ok = ""

    ultimos = datos.sort_values("fecha").groupby("tramo").tail(1).set_index("tramo").sort_index()
    izq, der = st.columns([3, 2])
    with izq, st.container(border=True, key="card_form"):
        ui.titulo_tarjeta("doc", "Nueva inspección", "Los campos con * son obligatorios")

        md('<div class="pv-group"><span>1</span><b>Ubicación y tramo</b></div>')
        opciones = [f"{t} · {ultimos.loc[t, 'via']}" for t in ultimos.index] + [NUEVO_TRAMO]
        c1, c2 = st.columns([2, 1])
        tramo_op = c1.selectbox("Tramo *", opciones, key="r_tramo")
        fecha = c2.date_input("Fecha de inspección *", date.today(), max_value=date.today(), format="DD/MM/YYYY", key="r_fecha")
        if tramo_op == NUEVO_TRAMO:
            n1, n2 = st.columns([1, 2])
            codigo = n1.text_input("Código del tramo *", placeholder="T-13", max_chars=10, key="r_codigo").strip().upper()
            via = n2.text_input("Nombre de la vía *", placeholder="Jr. …", max_chars=60, key="r_via").strip()
            edad_sug, tran_sug = 1.0, "Medio"
        else:
            codigo = tramo_op.split(" · ")[0]
            via = ultimos.loc[codigo, "via"]
            anterior = ultimos.loc[codigo]
            edad_sug = round(float(anterior["antiguedad"]) + max(0, (pd.Timestamp(fecha) - anterior["fecha"]).days) / 365.25, 1)
            tran_sug = anterior["transito"] if anterior["transito"] in M.TRANSITOS else "Medio"
        c3, c4, c5 = st.columns(3)
        edad = c3.number_input("Antigüedad (años) *", 0.0, float(M.EDAD_MAX), value=min(edad_sug, float(M.EDAD_MAX)), step=0.1,
                               key=f"r_edad_{codigo}_{fecha}", help="Se sugiere a partir de la última inspección del tramo.")
        tran = c4.selectbox("Tránsito *", M.TRANSITOS, index=M.TRANSITOS.index(tran_sug), key=f"r_tran_{codigo}")
        pci = c5.number_input("PCI observado *", 0, 100, value=None, step=1, placeholder="0–100", key="r_pci",
                              help="Debe provenir de la evaluación técnica PCI.")

        md('<div class="pv-group"><span>2</span><b>Patología y severidad</b></div>')
        falla = ss.get("r_falla", M.PATOLOGIAS[0])
        tiles = ""
        for p in M.PATOLOGIAS:
            on = p == falla
            chk = f'<span class="pv-check">{ui.icono("check", 12, "#FFF", 3)}</span>' if on else ""
            tiles += f'<div class="pv-gtile{" on" if on else ""}">{chk}<span class="pv-gimg">{ui.icono_patologia(p, 48)}</span>{p}</div>'
        md(f'<div class="pv-gallery">{tiles}</div>')
        falla = st.radio("Tipo de patología *", M.PATOLOGIAS + ["Ninguna"], horizontal=True, key="r_falla")
        sev = st.radio("Severidad *", M.SEVERIDADES, horizontal=True, key="r_sev", index=None,
                       disabled=falla == "Ninguna")

        md('<div class="pv-group"><span>3</span><b>Evidencia</b></div>')
        e1, e2 = st.columns([1, 1])
        foto = e1.file_uploader("Fotografía (JPG o PNG, máx. 5 MB)", type=["jpg", "jpeg", "png"],
                                key=f"r_foto_{ss.get('r_foto_key', 0)}")
        if foto is not None:
            e1.image(foto, caption=esc(foto.name), width=220)
        obs = e2.text_area("Observaciones", max_chars=300, height=130, key="r_obs",
                           placeholder="Ubicación exacta, losas afectadas, drenaje…")

        datos_form = {"tramo": codigo, "via": via, "fecha": fecha.isoformat() if fecha else "", "antiguedad": edad,
                      "transito": tran, "falla": falla, "severidad": "—" if falla == "Ninguna" else (sev or ""),
                      "pci": pci, "foto": "", "observaciones": (obs or "").strip()}
        repetido = tramo_op == NUEVO_TRAMO and bool(codigo) and codigo in ultimos.index

        # Validación y comparación con el modelo
        pendientes = M.validar_inspeccion(datos_form)
        errores = ss.pop("r_errores", [])
        if errores:
            ui.aviso("<b>No se guardó. Revise los datos:</b><br>" + "<br>".join("• " + esc(e) for e in errores), "error")
        elif repetido:
            ui.aviso(f"El tramo {esc(codigo)} ya existe; selecciónelo de la lista.", "error")
        elif not pendientes and res:
            est = res.predecir(edad)
            ui.aviso(f"<b>Datos válidos.</b> El modelo estima PCI {est:.1f} para {edad:g} años; el PCI observado es {pci} "
                     f"(diferencia de {abs(est - pci):.1f} puntos).", "ok", "check")
        else:
            ui.aviso("Complete: " + " ".join(esc(e) for e in pendientes), "info")

        b1, b2 = st.columns([2, 1])
        b1.button("Guardar inspección", type="primary", use_container_width=True, icon=":material/save:",
                  on_click=_guardar, args=(datos_form, foto, repetido))
        b2.button("Limpiar", use_container_width=True, on_click=_limpiar_formulario)

    with der, st.container(border=True, key="card_regs"):
        ui.titulo_tarjeta("table", "Registros guardados", f"{len(datos)} inspecciones válidas")
        filtro = st.selectbox("Filtrar por patología", ["Todas"] + M.PATOLOGIAS + ["Ninguna"])
        vista = datos if filtro == "Todas" else datos[datos["falla"] == filtro]
        vista = vista.sort_values(["fecha", "id"], ascending=False)
        html = ""
        for _, r in vista.head(6).iterrows():
            c = M.condicion(r.pci)
            img = ui.imagen_base64(os.path.join(M.CARPETA_FOTOS, r.foto)) if r.foto else ""
            mini = f'<img src="{img}" alt="Foto de {esc(r.tramo)}">' if img else f'<span class="pv-gimg">{ui.icono_patologia(r.falla, 38)}</span>'
            html += (f'<div class="pv-rec">{mini}<div class="pv-rec-main"><div><b style="color:#0B4F8A;font-size:13px">{esc(r.tramo)}</b> '
                     f'<span class="t">{esc(r.via)} · {r.fecha:%d/%m/%Y}</span></div><b style="font-size:13px">{esc(r.falla)}</b>'
                     f'<div style="display:flex;gap:6px;flex-wrap:wrap">{ui.etiqueta_severidad(r.severidad)}{ui.etiqueta_condicion(r.pci)}</div></div>'
                     f'<div class="pv-rec-pci"><b style="color:{c["texto"]}">{r.pci:.0f}</b><small>PCI</small></div></div>')
        md(html or '<p style="color:#4A5E78;font-size:13px">No hay registros con esa patología.</p>')
        with st.expander(f"Ver los {len(vista)} registros"):
            st.dataframe(vista.drop(columns=["condicion"]), hide_index=True, use_container_width=True,
                         column_config={"fecha": st.column_config.DateColumn("fecha", format="DD/MM/YYYY")})
        x1, x2 = st.columns(2)
        x1.download_button("Exportar CSV", a_csv(vista.drop(columns=["condicion"])), "registros_patologias.csv", "text/csv",
                           type="primary", use_container_width=True, icon=":material/download:")
        x2.download_button("Exportar Excel", a_excel(vista.drop(columns=["condicion"])), "registros_patologias.xlsx",
                           use_container_width=True, icon=":material/table:")


# ======================================================================
iniciar_estado()
pagina = barra_lateral()
{"Dashboard": pagina_dashboard, "Datos y modelo": pagina_modelo,
 "Simulador PCI": pagina_simulador, "Registro de patologías": pagina_registro}[pagina]()
md('<p style="text-align:center;font-size:12px;color:#5B6F88;margin-top:28px">PAVEMENT AI · SENATI · Datos ficticios con fines '
   'académicos. El PCI observado debe provenir de una evaluación técnica; el modelo no reemplaza la metodología PCI ni una inspección estructural.</p>')
