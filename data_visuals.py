import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# Cargar los datos
@st.cache_data
def cargar_datos():
    df = pd.read_excel("BaseDeDatosFinal(Demográficos+Test).xlsx")
    df.columns = df.columns.str.strip()
    return df

df = cargar_datos()

# Normalizar valores de "Género"
df["Genero"] = df["Genero"].str.lower()
mapeo_genero = {
    "masculino": "Masculino", "masculino ": "Masculino", "hombre": "Masculino", "masculina": "Masculino",
    "femenino": "Femenino", "femenino ": "Femenino", "femenina": "Femenino", "femenina ": "Femenino",
    "femenini": "Femenino", "mujer": "Femenino"
}
df["Genero"] = df["Genero"].replace(mapeo_genero)

# Título principal
st.title("📊 Análisis Interactivo del Test de Burnout - Equilibria")
st.markdown("Este dashboard permite explorar de forma dinámica los datos del test de burnout, cruzando variables demográficas con resultados de subescalas.")

# === FILTROS ===
st.sidebar.header("🎚️ Filtros")
generos = df["Genero"].dropna().unique()
genero_seleccionado = st.sidebar.multiselect("Género", generos, default=generos)

edad_min = int(df["Edad"].min())
edad_max = int(df["Edad"].max())
edad_rango = st.sidebar.slider("Edad", edad_min, edad_max, (edad_min, edad_max))

total_min = int(df["Total"].min())
total_max = int(df["Total"].max())
total_rango = st.sidebar.slider("Puntaje Total", total_min, total_max, (total_min, total_max))

df_filtrado = df[
    df["Genero"].isin(genero_seleccionado) &
    df["Edad"].between(*edad_rango) &
    df["Total"].between(*total_rango)
]

# === MATRIZ DE CORRELACIÓN ===
st.subheader("🔗 Matriz de Correlación: Demográficos vs Resultados del Test")

df_dummies = pd.get_dummies(df_filtrado, columns=["Genero"])
columnas_demograficas = ["Edad", "Genero_Femenino", "Genero_Masculino"]
columnas_test = ["Total", "AMF", "RFC", "RPD"]

df_corr = df_dummies[columnas_demograficas + columnas_test]
corr_matrix = df_corr.corr().loc[columnas_demograficas + columnas_test, columnas_demograficas + columnas_test]

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title='Matriz de Correlación',
    width=900,
    height=700
)
st.plotly_chart(fig_corr, use_container_width=True)

# === DISTRIBUCIÓN NORMAL ===
st.subheader("📈 Distribución del Puntaje Total de Burnout")

x_vals = df_filtrado["Total"].dropna().values
kde = gaussian_kde(x_vals)
x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
y_vals = kde(x_range)

fig_kde = go.Figure()
fig_kde.add_trace(go.Scatter(x=x_range, y=y_vals, mode='lines', line=dict(color='blue', width=3), name='Densidad'))

# Colores por zonas (visual)
zonas = [
    {"rango": (0, 29), "color": "#43a047", "nombre": "Nulo"},
    {"rango": (29, 36), "color": "#1e88e5", "nombre": "Leve"},
    {"rango": (36, 46), "color": "#fbc02d", "nombre": "Moderado"},
    {"rango": (46, 80), "color": "#e53935", "nombre": "Elevado"},
]

for zona in zonas:
    fig_kde.add_vrect(
        x0=zona["rango"][0], x1=zona["rango"][1],
        fillcolor=zona["color"], opacity=0.25, line_width=0,
        annotation_text=zona["nombre"], annotation_position="top left"
    )

fig_kde.update_layout(
    title="Distribución del Puntaje Total (Estimación KDE)",
    xaxis_title="Puntaje Total",
    yaxis_title="Densidad Estimada",
    template="plotly_white"
)

st.plotly_chart(fig_kde, use_container_width=True)

# === PORCENTAJE DE CONTRIBUCIÓN DE SUBESCALAS ===
st.subheader("📊 Contribución Promedio de Cada Subescala al Total")

promedios = {
    "Agotamiento Mental y Físico (AMF)": df_filtrado["AMF"].mean(),
    "Respuestas Cognitivas/Conductuales (RFC)": df_filtrado["RFC"].mean(),
    "Realización Personal/Despersonalización (RPD)": df_filtrado["RPD"].mean(),
}
total = df_filtrado["Total"].mean()
porcentajes = {k: round((v / total) * 100, 2) for k, v in promedios.items()}

fig_bar = px.bar(
    x=list(porcentajes.keys()),
    y=list(porcentajes.values()),
    labels={'x': 'Subescala', 'y': 'Contribución (%)'},
    text=[f'{v}%' for v in porcentajes.values()],
    title="Porcentaje Promedio de Contribución por Subescala",
    color=list(porcentajes.keys()),
    color_discrete_sequence=["#43a047", "#1e88e5", "#e53935"]
)
fig_bar.update_traces(textposition='outside')
fig_bar.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Desarrollado por Schwerdt Gonzalo | 🧠 Equilibria")
