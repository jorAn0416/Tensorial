import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funciones geométricas
def latlon_to_xyz(lat_deg, lon_deg, R):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return np.array([x, y, z]).T


def distancia_extrinseca(latitudes, longitudes, R):
    puntos = latlon_to_xyz(latitudes, longitudes, R)
    dif = np.diff(puntos, axis=0)
    distancias = np.linalg.norm(dif, axis=1)
    return np.sum(distancias)


def distancia_intrinseca(latitudes, longitudes, R):
    lat = np.radians(latitudes)
    lon = np.radians(longitudes)

    dlat = np.diff(lat)
    dlon = np.diff(lon)

    lat_media = (lat[:-1] + lat[1:]) / 2

    ds = R * np.sqrt(dlat**2 + (np.cos(lat_media) * dlon)**2)

    return np.sum(ds)


def generar_ecuador(N=500):
    lat = np.zeros(N)
    lon = np.linspace(0, 360, N)
    return lat, lon


def generar_paralelo(lat_paralelo, N=500):
    lat = np.ones(N) * lat_paralelo
    lon = np.linspace(0, 360, N)
    return lat, lon


# Gráficas
def graficar_esfera(latitudes, longitudes, R, titulo):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(-np.pi/2, np.pi/2, 40)

    x = R * np.outer(np.cos(v), np.cos(u))
    y = R * np.outer(np.cos(v), np.sin(u))
    z = R * np.outer(np.sin(v), np.ones_like(u))

    ax.plot_surface(x, y, z, alpha=0.25)

    puntos = latlon_to_xyz(latitudes, longitudes, R)

    ax.plot(
        puntos[:, 0],
        puntos[:, 1],
        puntos[:, 2],
        linewidth=3
    )

    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect([1, 1, 1])

    return fig


def graficar_mapa(latitudes, longitudes, titulo):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(longitudes, latitudes, marker="o")
    ax.set_title(titulo)
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    ax.grid(True)

    ax.set_xlim(-180, 360)
    ax.set_ylim(-90, 90)

    return fig


# Interfaz de la aplicación
#EN este bloque se generara la interfaz para que el usuario usando streamlit

st.title("Medición de longitudes sobre la Tierra")
st.write("Comparación entre métrica extrínseca en R³ y métrica intrínseca usando latitud-longitud.")

R = st.number_input(
    "Radio terrestre R",
    min_value=1.0,
    value=6371.0,
    step=10.0
)

st.sidebar.header("Opciones")

opcion = st.sidebar.selectbox(
    "Selecciona qué quieres medir",
    [
        "Ecuador",
        "Paralelo",
        "Trayectoria real"
    ]
)

# Ecuador
if opcion == "Ecuador":
    lat, lon = generar_ecuador()

    st.subheader("Ecuador")

    L_ext = distancia_extrinseca(lat, lon, R)
    L_int = distancia_intrinseca(lat, lon, R)
    L_teorica = 2 * np.pi * R

    tabla = pd.DataFrame({
        "Método": [
            "Extrínseca en R³",
            "Intrínseca sobre la esfera",
            "Valor teórico"
        ],
        "Longitud": [
            L_ext,
            L_int,
            L_teorica
        ]
    })

    st.dataframe(tabla)

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(graficar_esfera(lat, lon, R, "Ecuador sobre la esfera"))

    with col2:
        st.pyplot(graficar_mapa(lat, lon, "Ecuador en el mapa"))


# Paralelo
elif opcion == "Paralelo":
    st.subheader("Paralelo")

    lat_paralelo = st.slider(
        "Latitud del paralelo",
        min_value=-89.0,
        max_value=89.0,
        value=30.0,
        step=1.0
    )

    lat, lon = generar_paralelo(lat_paralelo)

    L_ext = distancia_extrinseca(lat, lon, R)
    L_int = distancia_intrinseca(lat, lon, R)
    L_teorica = 2 * np.pi * R * np.cos(np.radians(lat_paralelo))

    tabla = pd.DataFrame({
        "Método": [
            "Extrínseca en R³",
            "Intrínseca sobre la esfera",
            "Valor teórico"
        ],
        "Longitud": [
            L_ext,
            L_int,
            L_teorica
        ]
    })

    st.dataframe(tabla)

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(graficar_esfera(lat, lon, R, f"Paralelo {lat_paralelo}° sobre la esfera"))

    with col2:
        st.pyplot(graficar_mapa(lat, lon, f"Paralelo {lat_paralelo}° en el mapa"))


# Trayectoria real
elif opcion == "Trayectoria real":
    st.subheader("Trayectoria real mediante puntos de latitud y longitud")

    st.write("Ingresa los puntos como una tabla. Puedes editar, agregar o eliminar filas.")

    datos_iniciales = pd.DataFrame({
        "Latitud": [25.6866, 19.4326, 20.6597],
        "Longitud": [-100.3161, -99.1332, -103.3496]
    })

    datos = st.data_editor(
        datos_iniciales,
        num_rows="dynamic"
    )

    lat = datos["Latitud"].to_numpy()
    lon = datos["Longitud"].to_numpy()

    if len(lat) >= 2:
        L_ext = distancia_extrinseca(lat, lon, R)
        L_int = distancia_intrinseca(lat, lon, R)

        diferencia = L_int - L_ext
        porcentaje = 100 * diferencia / L_int

        tabla = pd.DataFrame({
            "Método": [
                "Extrínseca en R³",
                "Intrínseca sobre la esfera",
                "Diferencia",
                "Diferencia porcentual"
            ],
            "Resultado": [
                L_ext,
                L_int,
                diferencia,
                porcentaje
            ],
            "Unidad": [
                "km",
                "km",
                "km",
                "%"
            ]
        })

        st.subheader("Tabla comparativa")
        st.dataframe(tabla)

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(graficar_esfera(lat, lon, R, "Trayectoria sobre la esfera"))

        with col2:
            st.pyplot(graficar_mapa(lat, lon, "Trayectoria en el mapa"))

    else:
        st.warning("Debes ingresar al menos dos puntos.")

