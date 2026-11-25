# PAVEMENT AI
**Predicción del deterioro del pavimento rígido mediante regresión lineal**
SENATI · Inteligencia Artificial · Reto individual · Huánuco, Perú

Aplicación web en **Streamlit** que registra patologías del pavimento rígido y estima el
**Índice de Condición del Pavimento (PCI)** con un modelo de **regresión lineal** (scikit-learn)
entrenado con la antigüedad como entrada y el PCI como salida.

> **Datos simulados.** El archivo `data/inspecciones_pavimento.csv` es FICTICIO y se generó con
> `generar_datos.py` solo para fines académicos. El PCI observado debe provenir de una evaluación
> técnica; el modelo **no reemplaza** la metodología PCI ni una inspección estructural.

---

## 1. Estructura del proyecto
```
pavement_ai/
├── app.py                  # Aplicación Streamlit (4 páginas)
├── modelo.py               # Datos, limpieza, entrenamiento, métricas y validación
├── ui.py                   # Estilos, componentes visuales y gráficos Plotly
├── generar_datos.py        # Genera el dataset simulado
├── pruebas.py              # Pruebas automáticas del modelo y de las validaciones
├── requirements.txt
├── data/
│   ├── inspecciones_pavimento.csv   # 61 filas (57 válidas + 4 con errores para limpiar)
│   └── fotos/                       # Fotografías de las inspecciones registradas
├── informe/Informe_PAVEMENT_AI.pdf  # Informe con evidencias y conclusiones
├── .streamlit/config.toml  # Tema (colores) de la app
└── .vscode/launch.json     # Ejecutar con F5 en VS Code
```

## 2. Instalación y ejecución en VS Code
1. Instala **Python 3.10 o superior** (en Windows marca *Add Python to PATH*) y la extensión **Python** de VS Code.
2. Abre la carpeta en VS Code: **File → Open Folder… → `pavement_ai`**.
3. (Opcional, recomendado) crea un entorno virtual en la terminal (**Terminal → New Terminal**):
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
   ```
4. Instala las librerías:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Ejecuta la aplicación:
   ```bash
   python -m streamlit run app.py
   ```
   Se abre el navegador en `http://localhost:8501`. También puedes presionar **F5** y elegir
   *PAVEMENT AI (Streamlit)*.
6. Pruebas automáticas: `python pruebas.py`

Para volver al dataset original: `python generar_datos.py` (sobrescribe el CSV y borra los registros agregados).

## 3. Páginas de la aplicación
| Página | Qué hace |
|---|---|
| **Dashboard** | Filtros por tramo, periodo y tránsito; indicadores (inspecciones, PCI promedio, tramos críticos, R²); gráfico PCI vs antigüedad con franjas de condición; dona de distribución; últimas inspecciones; exportación CSV/Excel. |
| **Datos y modelo** | Carga de CSV, limpieza con Pandas (duplicados, nulos, rangos), división cronológica 80/20, entrenamiento de `LinearRegression`, métricas **R², MAE y RMSE** en prueba, gráfico real vs predicho, interpretación y preguntas de sustentación. |
| **Simulador PCI** | Antigüedad actual + horizonte → PCI actual, PCI futuro, condición y pérdida; velocímetro; gráfico de proyección con zona de extrapolación; **tres escenarios** exportables. |
| **Registro de patologías** | Formulario validado (tramo, fecha, antigüedad, tránsito, PCI observado, patología, severidad, foto, observaciones). Guarda en el CSV, almacena la foto, **reentrena el modelo** y exporta CSV/Excel. |

## 4. Modelo
- **Entrada:** antigüedad del pavimento (años). **Salida:** PCI (0–100).
- **División:** cronológica; el 20 % de inspecciones más recientes es el conjunto de prueba.
- **Resultado con los datos simulados:** `PCI = 110.93 − 8.25 × antigüedad`, R² = 0.97, MAE = 3.48, RMSE = 4.03.
- La predicción se limita a 0–100 y se advierte cuando se **extrapola** fuera del rango de entrenamiento (1.0–9.3 años).
- El modelo se vuelve a entrenar automáticamente al registrar inspecciones o cargar otro CSV (no hay predicciones estáticas).

## 5. Validación y seguridad
- Rango de antigüedad 0–50, PCI 0–100, fecha no futura, campos obligatorios, severidad requerida.
- Foto solo JPG/PNG y máximo 5 MB; el nombre del archivo lo genera la app.
- Todo texto ingresado se **escapa** antes de mostrarse en HTML.
- El CSV subido se valida (columnas obligatorias) y los errores se muestran sin detener la app.

## 6. Formato del CSV
`id, tramo, via, fecha (AAAA-MM-DD), antiguedad, transito (Bajo/Medio/Alto), falla, severidad (Baja/Media/Alta/—), pci, foto, observaciones`
Obligatorias para entrenar: `tramo, fecha, antiguedad, pci`.

## 7. Autor
[wilfredo alcedo santiago] · SENATI · 2026
