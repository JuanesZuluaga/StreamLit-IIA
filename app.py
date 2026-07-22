import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Dashboard Mundial", layout="wide")

st.title("⚽ Dashboard de Datos Simulados del Mundial")

# -------------------------
# Generación de datos
# -------------------------

np.random.seed(42)

equipos = [
    "Argentina","Brasil","España","Francia","Alemania",
    "Portugal","México","Colombia","Uruguay","Japón",
    "Estados Unidos","Marruecos","Inglaterra","Croacia",
    "Bélgica","Países Bajos"
]

continentes = {
    "Argentina":"América",
    "Brasil":"América",
    "España":"Europa",
    "Francia":"Europa",
    "Alemania":"Europa",
    "Portugal":"Europa",
    "México":"América",
    "Colombia":"América",
    "Uruguay":"América",
    "Japón":"Asia",
    "Estados Unidos":"América",
    "Marruecos":"África",
    "Inglaterra":"Europa",
    "Croacia":"Europa",
    "Bélgica":"Europa",
    "Países Bajos":"Europa"
}

grupos = ["A","B","C","D"]

estilos = ["Ofensivo","Defensivo","Posesión","Contraataque"]

registros = []

for equipo in equipos:
    registros.append({
        "Equipo": equipo,
        "Grupo": np.random.choice(grupos),
        "Continente": continentes[equipo],
        "Estilo": np.random.choice(estilos),
        "Goles": np.random.randint(0,15),
        "Tiros": np.random.randint(20,90),
        "Posesión (%)": np.random.randint(35,70),
        "Faltas": np.random.randint(20,60),
        "Tarjetas Amarillas": np.random.randint(0,10),
        "Tarjetas Rojas": np.random.randint(0,3),
        "Puntos": np.random.randint(0,10)
    })

df = pd.DataFrame(registros)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Filtros")

grupo = st.sidebar.multiselect(
    "Grupo",
    options=df["Grupo"].unique(),
    default=df["Grupo"].unique()
)

continente = st.sidebar.multiselect(
    "Continente",
    options=df["Continente"].unique(),
    default=df["Continente"].unique()
)

datos = df[
    (df["Grupo"].isin(grupo)) &
    (df["Continente"].isin(continente))
]

# -------------------------
# KPIs
# -------------------------

c1,c2,c3,c4 = st.columns(4)

c1.metric("Equipos", len(datos))
c2.metric("Goles Totales", datos["Goles"].sum())
c3.metric("Promedio Posesión", round(datos["Posesión (%)"].mean(),1))
c4.metric("Promedio Puntos", round(datos["Puntos"].mean(),1))

st.divider()

# -------------------------
# Tabla
# -------------------------

st.subheader("Datos Simulados")

st.dataframe(datos, use_container_width=True)

# -------------------------
# Estadísticas
# -------------------------

st.subheader("Resumen Estadístico")

st.dataframe(datos.describe())

# -------------------------
# Gráficas
# -------------------------

col1,col2 = st.columns(2)

with col1:

    fig = px.bar(
        datos,
        x="Equipo",
        y="Goles",
        color="Continente",
        title="Goles por Equipo"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig2 = px.scatter(
        datos,
        x="Posesión (%)",
        y="Puntos",
        color="Grupo",
        size="Goles",
        hover_name="Equipo",
        title="Posesión vs Puntos"
    )

    st.plotly_chart(fig2, use_container_width=True)

col3,col4 = st.columns(2)

with col3:

    fig3 = px.pie(
        datos,
        names="Continente",
        title="Distribución por Continente"
    )

    st.plotly_chart(fig3, use_container_width=True)

with col4:

    fig4 = px.box(
        datos,
        x="Grupo",
        y="Goles",
        color="Grupo",
        title="Distribución de Goles por Grupo"
    )

    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Mapa de Calor de Correlaciones")

corr = datos.select_dtypes(include=np.number).corr()

fig5 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlación entre Variables Cuantitativas"
)

st.plotly_chart(fig5, use_container_width=True)
