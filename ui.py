"""
Componentes visuales de PAVEMENT AI: estilos CSS, encabezado, tarjetas,
etiquetas de condición y gráficos Plotly.
"""
from __future__ import annotations

import base64
import html as _html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import modelo as M

AZUL = "#0B74C8"
MARINO = "#0A2747"
TEXTO = "#10243E"
GRIS = "#4A5E78"

ICONOS = {
    "doc": '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h4"/>',
    "gauge": '<path d="M4 16a8 8 0 1 1 16 0"/><path d="M12 16l4-5"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5v.5"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r=".8"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 6-7"/>',
    "pie": '<path d="M12 3a9 9 0 1 0 9 9h-9z"/><path d="M15 3.5A9 9 0 0 1 20.5 9H15z"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 4v16"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "sliders": '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "book": '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "pin": '<path d="M12 21s-7-6.5-7-12a7 7 0 0 1 14 0c0 5.5-7 12-7 12z"/><circle cx="12" cy="9" r="2.5"/>',
}

PATOLOGIA_SVG = {
    "Grietas longitudinales": '<path d="M26 6l-4 14 4 12-4 14"/>',
    "Grietas transversales": '<path d="M4 24l12-2 12 4 12-4"/>',
    "Grietas de esquina": '<path d="M4 22L20 6"/>',
    "Losa dividida": '<path d="M24 4v40M4 28l20-6 18 8"/>',
    "Despostillamiento": '<path d="M4 20l6 4 5-6 6 8 6-7 6 6 5-4"/>',
    "Escalonamiento": '<path d="M4 20h20v8h20"/>',
    "Ninguna": '<path d="M6 24h36" stroke-dasharray="4 4"/>',
}

SEV_COLOR = {"Baja": ("#E7F1FC", "#0B4F8A"), "Media": ("#FDEBD6", "#8A4705"),
             "Alta": ("#F8DAD6", "#86231A"), "—": ("#EEF2F7", "#4A5E78"), "": ("#EEF2F7", "#4A5E78")}


def esc(texto) -> str:
    """Escapa texto ingresado por el usuario antes de ponerlo en HTML."""
    return _html.escape(str(texto if texto is not None else ""))


def md(html: str):
    """Muestra HTML. Quita la sangría para que Markdown no lo convierta en código."""
    limpio = "".join(linea.strip() for linea in html.splitlines())
    st.markdown(limpio, unsafe_allow_html=True)


def icono(nombre, tam=18, color="currentColor", grosor=1.8):
    return (f'<svg aria-hidden="true" width="{tam}" height="{tam}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{grosor}" stroke-linecap="round" stroke-linejoin="round">{ICONOS[nombre]}</svg>')


def icono_patologia(nombre, tam=44):
    d = PATOLOGIA_SVG.get(nombre, PATOLOGIA_SVG["Ninguna"])
    return (f'<svg aria-hidden="true" width="{tam}" height="{tam}" viewBox="0 0 48 48" fill="none" stroke="#1E2A38" '
            f'stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round">{d}</svg>')


def ficha(nombre, fondo="#E7F1FC", color="#0B5FA5", tam=36):
    return f'<span class="pv-tile" style="width:{tam}px;height:{tam}px;background:{fondo};color:{color}">{icono(nombre)}</span>'


def etiqueta(texto, fondo, color):
    return f'<span class="pv-pill" style="background:{fondo};color:{color}">{texto}</span>'


def etiqueta_condicion(pci):
    c = M.condicion(pci)
    return etiqueta(f'<span class="pv-dot" style="background:{c["color"]}"></span>{c["nombre"]}', c["fondo"], c["texto"])


def etiqueta_severidad(sev):
    fondo, color = SEV_COLOR.get(sev, SEV_COLOR["—"])
    return etiqueta(esc(sev or "—"), fondo, color)


# ------------------------------------------------------------------ estilos
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #F2F5F9; color: #10243E; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; display: flex; flex-direction: column; width: 100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { padding: 11px 14px; border-radius: 10px; width: 100%; margin: 0; min-height: 44px; display: flex; align-items: center; box-sizing: border-box; cursor: pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #F2F5F9; }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:not([data-testid="stMarkdownContainer"]),
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 15px; font-weight: 500; color: #3A4F69; }
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: #E7F1FC; box-shadow: inset 4px 0 0 #0B74C8; }
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p,
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: #0B4F8A; font-weight: 700; }
div[class*="st-key-card"] { background: #FFFFFF; border-radius: 14px !important; border: 1px solid #E2E8F0 !important; padding: 18px 20px !important; box-shadow: 0 1px 2px rgba(16,36,62,.05); }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 10px; min-height: 44px; font-weight: 600; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: #0B74C8; border-color: #0B74C8; }
.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea { border-radius: 10px; }
label[data-testid="stWidgetLabel"] p { font-weight: 600; color: #22374F; font-size: 13px; }
.pv-hero { background: #0A2747; color: #FFF; border-radius: 16px; padding: 24px 28px 0; overflow: hidden; margin-bottom: 6px; }
.pv-hero-top { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px; align-items: flex-start; }
.pv-kicker { font-size: 11px; letter-spacing: 2.4px; color: #7CCBFF; font-weight: 600; }
.pv-hero h1 { margin: 6px 0 6px; font-size: 32px; font-weight: 800; letter-spacing: -0.6px; line-height: 1.1; color: #FFF; padding: 0; }
.pv-hero p { margin: 0; font-size: 14px; color: #C9D8EA; line-height: 1.5; max-width: 700px; }
.pv-road { display: flex; gap: 8px; height: 22px; margin-top: 18px; }
.pv-road div { flex: 1; background: #51667E; border-radius: 6px 6px 0 0; display: flex; align-items: center; }
.pv-road div span { width: 100%; border-top: 2px dashed #FFF; }
.pv-chip-sim { font-size: 12px; font-weight: 600; color: #FFE2A8; border: 1px solid #8C6B2E; background: #2A3440; padding: 6px 12px; border-radius: 999px; white-space: nowrap; }
.pv-chip-ok { font-size: 12px; font-weight: 600; color: #CFE6FF; background: #173F67; padding: 6px 12px; border-radius: 999px; white-space: nowrap; }
.pv-card { background: #FFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 20px; box-shadow: 0 1px 2px rgba(16,36,62,.05); }
.pv-kpi { display: flex; flex-direction: column; justify-content: space-between; gap: 14px; min-height: 172px; box-sizing: border-box; }
.pv-kpi-top { display: flex; justify-content: space-between; align-items: center; }
.pv-kpi-val { font-size: 32px; font-weight: 800; letter-spacing: -1px; line-height: 1.1; }
.pv-kpi-lbl { font-size: 13px; color: #4A5E78; font-weight: 500; }
.pv-tile { border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; }
.pv-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px; white-space: nowrap; }
.pv-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.pv-head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.pv-head h3 { margin: 0; font-size: 16px; font-weight: 700; color: #10243E; padding: 0; }
.pv-head small { display: block; font-size: 12px; color: #4A5E78; margin-top: 2px; }
.pv-table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.pv-table th { padding: 10px 10px; text-align: left; font-size: 11.5px; font-weight: 600; letter-spacing: .3px; color: #4A5E78; background: #F5F8FB; border-bottom: 1px solid #E2E8F0; white-space: nowrap; }
.pv-table td { padding: 10px 10px; border-bottom: 1px solid #EEF2F7; white-space: nowrap; color: #10243E; }
.pv-table tr:hover td { background: #F6F9FD; }
.pv-tablebox { overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 10px; background: #FFF; }
.pv-bar { width: 70px; height: 6px; border-radius: 3px; background: #EEF2F7; display: inline-block; vertical-align: middle; }
.pv-bar span { display: block; height: 6px; border-radius: 3px; }
.pv-alert { border-radius: 10px; padding: 12px 14px; font-size: 13px; line-height: 1.5; display: flex; gap: 10px; align-items: flex-start; }
.pv-steps { display: flex; align-items: flex-start; overflow-x: auto; padding: 6px 0 2px; }
.pv-step { display: flex; flex-direction: column; align-items: center; gap: 5px; min-width: 104px; text-align: center; }
.pv-step b { font-size: 13px; } .pv-step small { font-size: 12px; color: #4A5E78; }
.pv-step-dot { width: 36px; height: 36px; border-radius: 50%; background: #2E8B57; display: flex; align-items: center; justify-content: center; }
.pv-step-line { flex: 1 1 20px; height: 2px; background: #2E8B57; margin-top: 17px; min-width: 16px; }
.pv-row { display: flex; justify-content: space-between; font-size: 13px; padding: 9px 0; border-bottom: 1px solid #EEF2F7; }
.pv-eq { background: #0A2747; border-radius: 12px; padding: 16px 18px; color: #FFF; }
.pv-eq span { display: block; } .pv-eq .k { font-size: 11px; letter-spacing: 1.6px; color: #7CCBFF; font-weight: 600; }
.pv-eq .f { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600; margin: 4px 0; }
.pv-mini { background: #F5F8FB; border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; }
.pv-mini small { font-size: 12px; color: #4A5E78; display: block; } .pv-mini b { font-size: 24px; font-weight: 800; }
.pv-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 10px; margin-bottom: 6px; }
.pv-gtile { position: relative; border: 1px solid #E2E8F0; background: #FFF; border-radius: 12px; padding: 10px 6px; display: flex; flex-direction: column; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; text-align: center; line-height: 1.25; }
.pv-gtile.on { border: 2px solid #0B74C8; background: #EEF6FF; box-shadow: 0 0 0 3px #D3E7FB; }
.pv-gimg { width: 58px; height: 58px; border-radius: 8px; background: #AAB5C0; display: flex; align-items: center; justify-content: center; }
.pv-check { position: absolute; top: 6px; right: 6px; width: 20px; height: 20px; border-radius: 50%; background: #0B74C8; display: flex; align-items: center; justify-content: center; }
.pv-group { display: flex; align-items: center; gap: 10px; margin: 14px 0 6px; padding-top: 14px; border-top: 1px solid #EEF2F7; }
.pv-group span { width: 26px; height: 26px; border-radius: 50%; background: #0A2747; color: #FFF; font-size: 13px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; }
.pv-group b { font-size: 15px; }
.pv-rec { display: flex; gap: 12px; align-items: center; padding: 10px 12px; border: 1px solid #E2E8F0; border-radius: 12px; margin-bottom: 8px; background: #FFF; }
.pv-rec img, .pv-rec .pv-gimg { width: 50px; height: 50px; border-radius: 8px; object-fit: cover; flex: 0 0 50px; }
.pv-rec-main { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.pv-rec-main > div:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pv-rec-main .t { font-size: 12px; color: #4A5E78; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pv-rec-pci { text-align: right; } .pv-rec-pci b { font-size: 22px; font-weight: 800; display: block; } .pv-rec-pci small { font-size: 11px; color: #4A5E78; }
.pv-side-brand { display: flex; align-items: center; gap: 12px; margin: 0 0 18px; }
.pv-side-brand b { font-size: 18px; font-weight: 800; letter-spacing: -.3px; display: block; color: #10243E; }
.pv-side-brand small { font-size: 12px; color: #4A5E78; }
.pv-side-box { border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; font-size: 12px; color: #4A5E78; margin-top: 18px; }
.pv-side-box code { display: block; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #10243E; background: #F2F5F9; padding: 8px; border-radius: 6px; margin: 8px 0; }
.pv-legend { display: flex; gap: 18px; font-size: 12px; color: #3A4F69; flex-wrap: wrap; }
.pv-legend span { display: inline-flex; align-items: center; gap: 6px; }
</style>
"""


def aplicar_estilos():
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(ruta, titulo, subtitulo, chip_extra=""):
    md(f"""
    <div class="pv-hero">
      <div class="pv-hero-top">
        <div>
          <div class="pv-kicker">SENATI / INTELIGENCIA ARTIFICIAL / {ruta}</div>
          <h1>{titulo}</h1>
          <p>{subtitulo}</p>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">{chip_extra}<span class="pv-chip-sim">Datos simulados</span></div>
      </div>
      <div class="pv-road" aria-hidden="true"><div><span></span></div><div><span></span></div><div><span></span></div><div><span></span></div></div>
    </div>""")


def titulo_tarjeta(icon, titulo, sub="", fondo="#E7F1FC", color="#0B5FA5"):
    s = f"<small>{sub}</small>" if sub else ""
    md(f'<div class="pv-head">{ficha(icon, fondo, color)}<div><h3>{titulo}</h3>{s}</div></div>')


def kpi(icon, etiqueta_txt, valor, insignia, ifondo, icolor, tfondo="#E7F1FC", tcolor="#0B5FA5", vcolor=TEXTO):
    md(f"""<div class="pv-card pv-kpi">
      <div class="pv-kpi-top">{ficha(icon, tfondo, tcolor)}{etiqueta(insignia, ifondo, icolor)}</div>
      <div><div class="pv-kpi-val" style="color:{vcolor}">{valor}</div><div class="pv-kpi-lbl">{etiqueta_txt}</div></div>
    </div>""")


def aviso(texto, tipo="info", icon="alert"):
    colores = {"ok": ("#E3F2E9", "#17623C"), "warn": ("#FBF3D6", "#5E4A05"), "error": ("#F8DAD6", "#86231A"),
               "naranja": ("#FDEBD6", "#8A4705"), "info": ("#E7F1FC", "#0B4F8A")}
    fondo, color = colores[tipo]
    md(f'<div class="pv-alert" style="background:{fondo};color:{color}">{icono(icon, 18)}<div>{texto}</div></div>')


def imagen_base64(ruta):
    try:
        with open(ruta, "rb") as f:
            ext = os.path.splitext(ruta)[1].lower().strip(".") or "jpeg"
            return f"data:image/{'jpeg' if ext == 'jpg' else ext};base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""


# ---------------------------------------------------------------- gráficos
FUENTE = dict(family="Inter, system-ui, sans-serif", size=12, color="#22374F")


def _layout(fig, alto=380, xtitulo="", ytitulo=""):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", font=FUENTE, showlegend=False,
                      hoverlabel=dict(bgcolor="white", font=FUENTE))
    fig.update_xaxes(title=xtitulo, showgrid=False, zeroline=False, linecolor="#9AAABD", ticks="outside", tickcolor="#9AAABD")
    fig.update_yaxes(title=ytitulo, showgrid=False, zeroline=False)
    return fig


def franjas_pci(fig, x0, x1):
    """Pinta las franjas de condición PCI en el fondo."""
    arriba = 100
    for lim, nombre, color, fondo, texto in M.CONDICIONES:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=lim, y1=arriba, fillcolor=fondo, opacity=0.8,
                      line_width=0, layer="below")
        fig.add_annotation(x=x1, y=(lim + arriba) / 2, text=nombre, showarrow=False, xanchor="right",
                           font=dict(size=10, color=texto), xshift=-4)
        arriba = lim


def recta_modelo(res, x_max):
    xs = np.linspace(0, x_max, 200)
    ys = np.clip(res.modelo.predict(pd.DataFrame({"antiguedad": xs})), 0, 100)
    return xs, ys


def grafico_dispersion(datos, res, alto=390):
    fig = go.Figure()
    x_max = max(15, float(datos["antiguedad"].max()) + 1 if len(datos) else 15)
    franjas_pci(fig, 0, x_max)
    xs, ys = recta_modelo(res, x_max)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#E07B00", width=3, dash="dash"),
                             name="Recta de regresión", hoverinfo="skip"))
    if len(datos):
        fig.add_trace(go.Scatter(
            x=datos["antiguedad"], y=datos["pci"], mode="markers", name="Inspecciones",
            marker=dict(size=10, color="#1F6FB8", line=dict(color="white", width=1.5)),
            customdata=np.stack([datos["tramo"], datos["via"], datos["fecha"].dt.strftime("%d/%m/%Y")], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
                          "Antigüedad: %{x} años<br>PCI: %{y}<extra></extra>"))
    _layout(fig, alto, "Antigüedad del pavimento (años)", "PCI")
    fig.update_xaxes(range=[0, x_max], dtick=1)
    fig.update_yaxes(range=[0, 102], dtick=25)
    return fig


def grafico_dona(datos):
    orden = [c[1] for c in M.CONDICIONES]
    conteo = datos["condicion"].value_counts().reindex(orden).fillna(0)
    conteo = conteo[conteo > 0]
    colores = {c[1]: c[2] for c in M.CONDICIONES}
    fig = go.Figure(go.Pie(labels=conteo.index, values=conteo.values, hole=0.62, sort=False,
                           marker=dict(colors=[colores[n] for n in conteo.index], line=dict(color="white", width=2)),
                           textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b style='font-size:26px'>{int(conteo.sum())}</b><br>inspecciones", showarrow=False,
                       font=dict(size=12, color=TEXTO))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor="white", font=FUENTE)
    return fig, conteo


def grafico_real_vs_pred(res, alto=360):
    real, pred = res.test["pci"].values, res.pred_test
    lo = float(min(real.min(), pred.min())) - 5
    hi = float(max(real.max(), pred.max())) + 5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=TEXTO, dash="dot", width=1.5),
                             hoverinfo="skip", name="Ideal"))
    for r, p in zip(real, pred):
        fig.add_trace(go.Scatter(x=[p, p], y=[p, r], mode="lines", line=dict(color="#E07B00", width=1.5),
                                 opacity=0.6, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=pred, y=real, mode="markers", marker=dict(size=10, color="#1F6FB8", line=dict(color="white", width=1.5)),
        customdata=np.stack([res.test["tramo"], real - pred], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Predicho: %{x:.1f}<br>Real: %{y}<br>Residuo: %{customdata[1]:+.1f}<extra></extra>"))
    _layout(fig, alto, "PCI predicho", "PCI real")
    fig.update_xaxes(range=[lo, hi], showgrid=True, gridcolor="#EEF2F7")
    fig.update_yaxes(range=[lo, hi], showgrid=True, gridcolor="#EEF2F7")
    return fig


def grafico_proyeccion(datos, res, edad, final, alto=400):
    x_max = max(15, final + 1)
    fig = go.Figure()
    franjas_pci(fig, 0, x_max)
    fig.add_shape(type="rect", x0=res.edad_max, x1=x_max, y0=0, y1=100, fillcolor=TEXTO, opacity=0.06, line_width=0)
    fig.add_annotation(x=(res.edad_max + x_max) / 2, y=97, text="<b>Zona de extrapolación</b>", showarrow=False,
                       font=dict(size=11, color="#22374F"))
    xs, ys = recta_modelo(res, x_max)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#E07B00", width=3, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=datos["antiguedad"], y=datos["pci"], mode="markers", opacity=0.55,
                             marker=dict(size=8, color="#1F6FB8", line=dict(color="white", width=1)),
                             hovertemplate="Historial<br>%{x} años · PCI %{y}<extra></extra>"))
    p_hoy, p_fut = res.predecir(edad), res.predecir(final)
    fig.add_trace(go.Scatter(x=[edad, final], y=[p_hoy, p_fut], mode="lines", line=dict(color=TEXTO, width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[edad], y=[p_hoy], mode="markers", marker=dict(size=16, color="white", line=dict(color=AZUL, width=4)),
                             hovertemplate="Hoy<br>%{x} años · PCI %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[final], y=[p_fut], mode="markers", marker=dict(size=20, color="#E5533D", line=dict(color="white", width=3)),
                             hovertemplate="Predicción<br>%{x} años · PCI %{y:.1f}<extra></extra>"))
    _layout(fig, alto, "Antigüedad del pavimento (años)", "PCI")
    fig.update_xaxes(range=[0, x_max], dtick=1)
    fig.update_yaxes(range=[0, 102], dtick=25)
    return fig


def grafico_velocimetro(pci):
    c = M.condicion(pci)
    pasos, arriba = [], 100
    for lim, nombre, color, fondo, texto in M.CONDICIONES:
        pasos.append(dict(range=[lim, arriba], color=color))
        arriba = lim
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(pci, 1),
        number=dict(font=dict(size=44, color=c["texto"], family="Inter"), valueformat=".1f"),
        gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor="#9AAABD", dtick=25),
                   bar=dict(color="rgba(0,0,0,0)", thickness=0), bgcolor="white", borderwidth=0, steps=pasos,
                   threshold=dict(line=dict(color=TEXTO, width=6), thickness=1, value=pci))))
    fig.update_layout(height=250, margin=dict(l=34, r=34, t=24, b=0), paper_bgcolor="white", font=FUENTE)
    return fig


CONFIG_PLOTLY = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "locale": "es"}
