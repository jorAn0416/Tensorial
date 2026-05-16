import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Variables simbólicas
u, v = sp.symbols("u v", real=True)
R, a = sp.symbols("R a", positive=True, real=True)

# Parametrización del toro
X = sp.Matrix([
    (R + a * sp.cos(v)) * sp.cos(u),
    (R + a * sp.cos(v)) * sp.sin(u),
    a * sp.sin(v)
])

Xu = sp.diff(X, u)
Xv = sp.diff(X, v)

# Tensor métrico inducido
g = sp.Matrix([
    [sp.simplify(Xu.dot(Xu)), sp.simplify(Xu.dot(Xv))],
    [sp.simplify(Xv.dot(Xu)), sp.simplify(Xv.dot(Xv))]
])

g_inv = sp.simplify(g.inv())


# Funciones auxiliares
def latex_expr(expr):
    return sp.latex(sp.simplify(expr))


def calcular_diferencial(f):
    df_du = sp.diff(f, u)
    df_dv = sp.diff(f, v)
    return sp.Matrix([df_du, df_dv])


def calcular_gradiente(df):
    grad = sp.simplify(g_inv * df)
    return grad


def derivada_direccional(f, wu, wv):
    df = calcular_diferencial(f)
    w = sp.Matrix([wu, wv])

    norma2 = sp.simplify((w.T * g * w)[0])
    norma = sp.sqrt(norma2)

    w_unit = sp.simplify(w / norma)

    D = sp.simplify((df.T * w_unit)[0])

    return D, w_unit, norma


def torus_xyz(U, V, Rval, aval):
    X = (Rval + aval * np.cos(V)) * np.cos(U)
    Y = (Rval + aval * np.cos(V)) * np.sin(U)
    Z = aval * np.sin(V)
    return X, Y, Z


def xyz_to_torus(x, y, z, Rval):
    u_val = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    v_val = np.arctan2(z, rho - Rval)
    return u_val, v_val


def graficar_toro(f, Rval, aval):
    f_num = sp.lambdify((u, v, R, a), f, "numpy")

    U = np.linspace(0, 2*np.pi, 100)
    V = np.linspace(0, 2*np.pi, 60)
    U, V = np.meshgrid(U, V)

    Xp, Yp, Zp = torus_xyz(U, V, Rval, aval)

    try:
        C = f_num(U, V, Rval, aval)
    except:
        C = np.zeros_like(U)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Xp, Yp, Zp,
        facecolors=plt.cm.viridis((C - np.nanmin(C)) / (np.nanmax(C) - np.nanmin(C) + 1e-12)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    ax.set_title("Función potencial sobre el toro")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 0.5])

    return fig


def graficar_trayectoria_toro(Rval, aval, u0, v0, wu, wv):
    t = np.linspace(0, 1.5, 100)

    U_path = u0 + wu * t
    V_path = v0 + wv * t

    Xp, Yp, Zp = torus_xyz(U_path, V_path, Rval, aval)

    U = np.linspace(0, 2*np.pi, 80)
    V = np.linspace(0, 2*np.pi, 40)
    U, V = np.meshgrid(U, V)
    Xt, Yt, Zt = torus_xyz(U, V, Rval, aval)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Xt, Yt, Zt, alpha=0.25)
    ax.plot(Xp, Yp, Zp, linewidth=4)
    ax.scatter([Xp[0]], [Yp[0]], [Zp[0]], s=60)

    ax.set_title("Trayectoria direccional sobre el toro")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 0.5])

    return fig



# App principal
st.title("Diferencial, gradiente y tensor métrico sobre un toro")

st.write(
    "Aplicación para estudiar cómo el tensor métrico inducido en un toro permite convertir "
    "covectores en vectores, calcular gradientes y derivadas direccionales."
)

st.sidebar.header("Parámetros del toro")

Rval = st.sidebar.number_input("Radio mayor R", value=3.0, min_value=0.5, step=0.1)
aval = st.sidebar.number_input("Radio menor a", value=1.0, min_value=0.1, step=0.1)

if aval >= Rval:
    st.warning("Para un toro estándar sin autointersección conviene usar a < R.")

st.sidebar.header("Función potencial")

funcion_texto = st.sidebar.text_input(
    "Escribe f(u,v)",
    value="sin(u) + cos(v)"
)

try:
    f = sp.sympify(
        funcion_texto,
        locals={
            "u": u,
            "v": v,
            "R": R,
            "a": a,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "pi": sp.pi
        }
    )
except:
    st.error("La función ingresada no es válida.")
    st.stop()


# Función elegida
st.subheader("1. Función potencial elegida")

st.latex("f(u,v)=" + latex_expr(f))


# Parametrización
st.subheader("2. Transformación desde coordenadas toroidales hacia R³")

st.latex(
    r"\vec X(u,v)=\left("
    r"(R+a\cos v)\cos u,\ "
    r"(R+a\cos v)\sin u,\ "
    r"a\sin v"
    r"\right)"
)

st.write("Vectores tangentes:")

st.latex(r"\vec X_u = " + sp.latex(Xu))
st.latex(r"\vec X_v = " + sp.latex(Xv))


# Tensor métrico
st.subheader("3. Tensor métrico calculado")

st.write("El tensor métrico se obtiene con:")

st.latex(r"g_{ij}=\vec X_i\cdot \vec X_j")

st.latex(r"g_{ij}=" + sp.latex(g))

st.write("Métrica inversa:")

st.latex(r"g^{ij}=" + sp.latex(g_inv))


# Diferencial y gradiente
df = calcular_diferencial(f)
grad = calcular_gradiente(df)

st.subheader("4. Diferencial total")

st.latex(
    "df="
    + latex_expr(df[0])
    + r"\,du + "
    + latex_expr(df[1])
    + r"\,dv"
)

st.subheader("5. Conversión covector → vector usando la métrica")

st.write("El diferencial es un covector:")

st.latex(r"df = \partial_i f\,dq^i")

st.write("El gradiente se obtiene levantando índices:")

st.latex(r"(\nabla f)^i = g^{ij}\partial_j f")

st.subheader("6. Gradiente sobre el toro")

st.latex(r"\nabla f = " + sp.latex(grad))


# Derivada direccional
st.subheader("7. Derivada direccional")

col1, col2 = st.columns(2)

with col1:
    wu = st.number_input("Componente direccional wᵘ", value=1.0)
    wv = st.number_input("Componente direccional wᵛ", value=1.0)

with col2:
    u0 = st.number_input("u₀", value=1.0)
    v0 = st.number_input("v₀", value=1.0)

D, w_unit, norma = derivada_direccional(f, wu, wv)

st.write("La norma del vector direccional se calcula con la métrica:")

st.latex(r"\|w\|^2 = g_{ij}w^i w^j")

st.latex(r"\|w\| = " + latex_expr(norma))

st.write("Vector unitario:")

st.latex(r"\hat w = " + sp.latex(w_unit))

st.write("Derivada direccional:")

st.latex(r"D_{\hat w}f = df(\hat w)=" + latex_expr(D))

D_val = float(
    D.subs({
        u: u0,
        v: v0,
        R: Rval,
        a: aval
    })
)

st.success(f"Derivada direccional en (u₀,v₀)=({u0},{v0}): {D_val:.8f}")


# Transformación inversa
st.subheader("8. Transformación aproximada desde R³ hacia coordenadas del toro")

st.write("Para un punto sobre el toro:")

st.latex(r"u = \operatorname{atan2}(y,x)")

st.latex(r"v = \operatorname{atan2}\left(z,\sqrt{x^2+y^2}-R\right)")

x0, y0, z0 = torus_xyz(u0, v0, Rval, aval)
uinv, vinv = xyz_to_torus(x0, y0, z0, Rval)

tabla_inv = pd.DataFrame({
    "Cantidad": ["x", "y", "z", "u recuperado", "v recuperado"],
    "Valor": [x0, y0, z0, uinv, vinv]
})

st.dataframe(tabla_inv)


# Verificación numérica
st.subheader("9. Verificación numérica")

h = st.number_input("Paso h", value=0.0001, format="%.6f")

f_num = sp.lambdify((u, v, R, a), f, "numpy")

wu_num = float(w_unit[0].subs({u: u0, v: v0, R: Rval, a: aval}))
wv_num = float(w_unit[1].subs({u: u0, v: v0, R: Rval, a: aval}))

D_num = (
    f_num(u0 + h * wu_num, v0 + h * wv_num, Rval, aval)
    - f_num(u0, v0, Rval, aval)
) / h

tabla_verificacion = pd.DataFrame({
    "Método": [
        "Derivada direccional simbólica",
        "Diferencia finita",
        "Error absoluto"
    ],
    "Valor": [
        D_val,
        D_num,
        abs(D_val - D_num)
    ]
})

st.dataframe(tabla_verificacion)


# Visualización 3D

st.subheader("10. Visualización 3D")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(graficar_toro(f, Rval, aval))

with col2:
    st.pyplot(graficar_trayectoria_toro(Rval, aval, u0, v0, wu_num, wv_num))


# Tabla final
st.subheader("11. Tabla comparativa final")

tabla_final = pd.DataFrame({
    "Cantidad": [
        "Función potencial",
        "Diferencial total",
        "Tensor métrico",
        "Métrica inversa",
        "Gradiente",
        "Derivada direccional"
    ],
    "Resultado": [
        str(f),
        str(df),
        str(g),
        str(g_inv),
        str(grad),
        str(D)
    ]
})

st.dataframe(tabla_final)