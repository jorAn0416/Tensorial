import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuración simbólica
x, y = sp.symbols("x y", real=True)
r, theta = sp.symbols("r theta", real=True, positive=True)

# Transformaciones
x_polar = r * sp.cos(theta)
y_polar = r * sp.sin(theta)


# Funciones auxiliares
def latex_expr(expr):
    return sp.latex(sp.simplify(expr))

def calcular_cartesiano(f):
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    df = sp.Matrix([df_dx, df_dy])

    g = sp.Matrix([[1, 0], [0, 1]])
    g_inv = g.inv()

    grad = g_inv * df

    return df, g, g_inv, grad

def calcular_polar(f_cart):
    f_polar = sp.simplify(f_cart.subs({x: x_polar, y: y_polar}))

    df_dr = sp.diff(f_polar, r)
    df_dtheta = sp.diff(f_polar, theta)

    df = sp.Matrix([df_dr, df_dtheta])

    g = sp.Matrix([[1, 0], [0, r**2]])
    g_inv = sp.Matrix([[1, 0], [0, 1/r**2]])

    grad = sp.simplify(g_inv * df)

    return f_polar, df, g, g_inv, grad

def derivada_direccional_cart(f, vx, vy):
    grad_x = sp.diff(f, x)
    grad_y = sp.diff(f, y)

    norma = sp.sqrt(vx**2 + vy**2)

    ux = vx / norma
    uy = vy / norma

    D = sp.simplify(grad_x * ux + grad_y * uy)

    return D, (ux, uy)

def derivada_direccional_polar(f_polar, vr, vtheta):
    df_dr = sp.diff(f_polar, r)
    df_dtheta = sp.diff(f_polar, theta)

    g = sp.Matrix([[1, 0], [0, r**2]])

    v = sp.Matrix([vr, vtheta])

    norma2 = sp.simplify((v.T * g * v)[0])
    norma = sp.sqrt(norma2)

    u = v / norma

    df = sp.Matrix([df_dr, df_dtheta])

    D = sp.simplify((df.T * u)[0])

    return D, u, norma

def evaluar(expr, x0, y0):
    return float(expr.subs({x: x0, y: y0}))

def evaluar_polar(expr, r0, theta0):
    return float(expr.subs({r: r0, theta: theta0}))

# Gráficas
def graficar_potencial_cart(f, xmin=-5, xmax=5, ymin=-5, ymax=5):
    f_num = sp.lambdify((x, y), f, "numpy")

    X = np.linspace(xmin, xmax, 120)
    Y = np.linspace(ymin, ymax, 120)
    X, Y = np.meshgrid(X, Y)

    try:
        Z = f_num(X, Y)
    except:
        Z = np.zeros_like(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(X, Y, Z, levels=30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Función potencial en coordenadas cartesianas")
    fig.colorbar(c, ax=ax)

    return fig

def graficar_gradiente_cart(f, xmin=-5, xmax=5, ymin=-5, ymax=5):
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    fx_num = sp.lambdify((x, y), fx, "numpy")
    fy_num = sp.lambdify((x, y), fy, "numpy")
    f_num = sp.lambdify((x, y), f, "numpy")

    X = np.linspace(xmin, xmax, 20)
    Y = np.linspace(ymin, ymax, 20)
    X, Y = np.meshgrid(X, Y)

    U = fx_num(X, Y)
    V = fy_num(X, Y)

    Z = f_num(X, Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(X, Y, Z, levels=20)
    ax.quiver(X, Y, U, V)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Campo gradiente sobre curvas de nivel")

    return fig

# InterFaz de la APP
st.title("Diferencial, gradiente y tensor métrico")
st.write("Aplicación para estudiar cómo el tensor métrico convierte el covector diferencial df en el vector gradiente.")

st.sidebar.header("Función potencial")

funcion_texto = st.sidebar.text_input(
    "Escribe f(x,y)",
    value="x**2 + y**2"
)

try:
    f = sp.sympify(funcion_texto)
except:
    st.error("La función ingresada no es válida.")
    st.stop()

st.subheader("1. Función potencial elegida")

st.latex("f(x,y) = " + latex_expr(f))

# Cartesiano
df_cart, g_cart, ginv_cart, grad_cart = calcular_cartesiano(f)

st.subheader("2. Desarrollo en coordenadas cartesianas")

st.write("El diferencial de la función es:")

st.latex(
    "df = "
    + latex_expr(df_cart[0])
    + r"\,dx + "
    + latex_expr(df_cart[1])
    + r"\,dy"
)

st.write("Tensor métrico cartesiano:")

st.latex(r"g_{ij} = " + sp.latex(g_cart))

st.write("Métrica inversa:")

st.latex(r"g^{ij} = " + sp.latex(ginv_cart))

st.write("El gradiente se obtiene levantando índices:")

st.latex(
    r"\nabla f = g^{ij}\partial_j f = "
    + sp.latex(grad_cart)
)

# Polar
f_polar, df_polar, g_polar, ginv_polar, grad_polar = calcular_polar(f)

st.subheader("3. Transformación a coordenadas polares")

st.write("Usando:")

st.latex(r"x = r\cos\theta,\qquad y = r\sin\theta")

st.write("La función en polares queda:")

st.latex("f(r,\\theta) = " + latex_expr(f_polar))

st.write("Diferencial en polares:")

st.latex(
    "df = "
    + latex_expr(df_polar[0])
    + r"\,dr + "
    + latex_expr(df_polar[1])
    + r"\,d\theta"
)

st.write("Tensor métrico polar:")

st.latex(r"g_{ij} = " + sp.latex(g_polar))

st.write("Métrica inversa:")

st.latex(r"g^{ij} = " + sp.latex(ginv_polar))

st.write("Gradiente en la base coordenada polar:")

st.latex(
    r"\nabla f = "
    + sp.latex(grad_polar)
)

st.info(
    "Nota: en coordenadas polares, la componente angular del gradiente incluye el factor 1/r² debido a la métrica inversa."
)

# Derivadas direccionales
st.subheader("4. Derivadas direccionales")

col1, col2 = st.columns(2)

with col1:
    st.write("Dirección cartesiana")
    vx = st.number_input("v_x", value=1.0)
    vy = st.number_input("v_y", value=1.0)

with col2:
    st.write("Punto de evaluación")
    x0 = st.number_input("x0", value=1.0)
    y0 = st.number_input("y0", value=1.0)

D_cart, u_cart = derivada_direccional_cart(f, vx, vy)

st.write("Vector unitario cartesiano:")

st.latex(
    r"\hat{u} = \left("
    + latex_expr(u_cart[0])
    + ","
    + latex_expr(u_cart[1])
    + r"\right)"
)

st.write("Derivada direccional:")

st.latex(r"D_{\hat{u}}f = " + latex_expr(D_cart))

D_val = evaluar(D_cart, x0, y0)

st.success(f"Valor numérico de la derivada direccional en ({x0}, {y0}): {D_val:.6f}")


# Verificación numérica
st.subheader("5. Verificación numérica")

h = st.number_input("Paso h para diferencia finita", value=0.001, format="%.6f")

ux_num = float(sp.N(u_cart[0]))
uy_num = float(sp.N(u_cart[1]))

f_num = sp.lambdify((x, y), f, "numpy")

D_num = (
    f_num(x0 + h * ux_num, y0 + h * uy_num)
    - f_num(x0, y0)
) / h

tabla_verificacion = pd.DataFrame({
    "Método": [
        "Derivada direccional simbólica",
        "Diferencia finita numérica",
        "Error absoluto"
    ],
    "Valor": [
        D_val,
        D_num,
        abs(D_val - D_num)
    ]
})

st.dataframe(tabla_verificacion)


# Gráficas
st.subheader("6. Gráficas")

xmin = st.slider("x mínimo", -20.0, 0.0, -5.0)
xmax = st.slider("x máximo", 0.0, 20.0, 5.0)
ymin = st.slider("y mínimo", -20.0, 0.0, -5.0)
ymax = st.slider("y máximo", 0.0, 20.0, 5.0)

col1, col2 = st.columns(2)

with col1:
    st.pyplot(graficar_potencial_cart(f, xmin, xmax, ymin, ymax))

with col2:
    st.pyplot(graficar_gradiente_cart(f, xmin, xmax, ymin, ymax))


# Tabla comparativa
st.subheader("7. Tabla comparativa")

tabla = pd.DataFrame({
    "Cantidad": [
        "Función en cartesianas",
        "Diferencial cartesiano",
        "Gradiente cartesiano",
        "Función en polares",
        "Diferencial polar",
        "Gradiente polar"
    ],
    "Resultado": [
        str(f),
        str(df_cart),
        str(grad_cart),
        str(f_polar),
        str(df_polar),
        str(grad_polar)
    ]
})

st.dataframe(tabla)
