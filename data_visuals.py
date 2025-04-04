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
total_rango = st.sidebar.slider("Total", total_min, total_max, (total_min, total_max))

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
corr_matrix = df_corr.corr()

fig_corr = px.imshow(
    corr_matrix.loc[columnas_demograficas + columnas_test, columnas_demograficas + columnas_test],
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title='Matriz de Correlación'
)
fig_corr.update_layout(height=600)  # Aumentar tamaño de imagen
st.plotly_chart(fig_corr, use_container_width=True)

# === TABLA DE CORRELACIONES TOP 15 ===
st.subheader("📋 Top 15 Correlaciones más Fuertes (absolutas)")

# Evitar correlaciones entre Género_Femenino y Género_Masculino y duplicadas
corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().reset_index()
corr_flat.columns = ['Variable 1', 'Variable 2', 'Correlación']

# Filtro para evitar género opuesto y auto-correlaciones (cercanas a 1)
corr_flat = corr_flat[
    ~(
        ((corr_flat["Variable 1"] == "Genero_Femenino") & (corr_flat["Variable 2"] == "Genero_Masculino")) |
        ((corr_flat["Variable 1"] == "Genero_Masculino") & (corr_flat["Variable 2"] == "Genero_Femenino")) |
        (corr_flat["Correlación"].abs() == 1)
    )
]

# Calcular valor absoluto y polaridad
corr_flat["Valor Absoluto"] = corr_flat["Correlación"].abs()
corr_flat["Polaridad"] = corr_flat["Correlación"].apply(lambda x: "Positiva" if x > 0 else "Negativa")

# Eliminar la columna "Correlación"
top_corr = corr_flat.drop(columns=["Correlación"]).sort_values(by="Valor Absoluto", ascending=False).head(15)

# Mostrar tabla
st.dataframe(top_corr.style.format({"Valor Absoluto": "{:.2f}"}))

# === DISTRIBUCIÓN KDE ===
st.subheader("📈 Distribución del Puntaje Total de Burnout")

x_vals = df_filtrado["Total"].dropna().values
kde = gaussian_kde(x_vals)
x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
y_vals = kde(x_range)

fig_kde = go.Figure()

# Áreas de color con más contraste
zonas = [
    {"rango": (0, 29), "color": "#2ecc71", "nombre": "Nulo"},
    {"rango": (29, 36), "color": "#16a085", "nombre": "Leve"},
    {"rango": (36, 46), "color": "#f1c40f", "nombre": "Moderado"},
    {"rango": (46, 80), "color": "#8e44ad", "nombre": "Elevado"},
]

for zona in zonas:
    centro = (zona["rango"][0] + zona["rango"][1]) / 2

    fig_kde.add_vrect(
        x0=zona["rango"][0], x1=zona["rango"][1],
        fillcolor=zona["color"], opacity=0.9, line_width=0,
        annotation=dict(
            text=f"<b>{zona['nombre']}</b>",
            x=centro,
            y=1.05,
            showarrow=False,
            font=dict(size=14, color="white"),
            xanchor="center",
            yanchor="bottom"
        )
    )

# Línea gris oscura tenue por debajo de la blanca (sombra)
fig_kde.add_trace(go.Scatter(
    x=x_range, y=y_vals,
    mode='lines',
    line=dict(color='rgba(80, 80, 80, 0.8)', width=6),
    name='Sombra',
    showlegend=False
))

# Línea blanca encima de todo
fig_kde.add_trace(go.Scatter(
    x=x_range, y=y_vals,
    mode='lines',
    line=dict(color='white', width=3),
    name='Densidad'
))

# Layout final con cuadrícula y fondo
fig_kde.update_layout(
    title="Distribución del Puntaje Total (Estimación KDE)",
    xaxis_title="Puntaje Total",
    yaxis_title="Densidad Estimada",
    height=500,
    plot_bgcolor='rgba(0,0,0,1)',
    paper_bgcolor='rgba(0,0,0,1)',
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(80,80,80,0.4)',
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(80,80,80,0.4)',
        zeroline=False
    ),
    font=dict(color="white"),
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
    color_discrete_sequence=["#8bff7d", "#9dbfff", "#ffb3b3"]
)
fig_bar.update_traces(textposition='outside')
fig_bar.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Desarrollado por Schwerdt Gonzalo | 🧠 Equilibria")
