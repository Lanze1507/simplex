"""
Simplex Web - Metodo Simplex con interfaz en el navegador (sin dependencias).

    python simplex_web.py                # abre http://127.0.0.1:8000
    python simplex_web.py --puerto 9000 --no-abrir
    python simplex_web.py --probar       # verifica los ejemplos y termina

Soporta: maximizar/minimizar, restricciones <=, >= y =, lado derecho negativo,
Dos Fases, aritmetica exacta (fracciones), regla anticiclado de Bland,
deteccion de infactible / no acotado / optimos multiples / degeneracion,
precios sombra con rangos, costos reducidos y grafica para 2 variables.
"""
import argparse
import json
import sys
import webbrowser
from fractions import Fraction as F
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_VARS, MAX_REST, MAX_ITER = 20, 50, 2000


# ----------------------------------------------------------------- utilidades
def num(t):
    t = str(t).strip().replace(",", ".")
    if t == "":
        return F(0)
    if len(t) > 24 or "e" in t.lower():
        raise ValueError(f"Valor no admitido: «{t[:24]}». Use decimales o fracciones (3/2).")
    try:
        return F(t)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Valor inválido: «{t}».")


def fr(x):
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def P(x):
    """Par [fraccion, decimal] para que la interfaz alterne la vista."""
    return None if x is None else [fr(x), f"{float(x):.6g}"]


def lineal(M, r):
    """Resuelve M x = r con fracciones (Gauss-Jordan)."""
    n = len(M)
    A = [list(f) + [r[i]] for i, f in enumerate(M)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0)
        A[c], A[p] = A[p], A[c]
        A[c] = [v / A[c][c] for v in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return [A[i][n] for i in range(n)]


# ----------------------------------------------------------------- tablero
class Tab:
    def __init__(s, T, cols, basis):
        s.T, s.cols, s.basis, s.out = T, cols, basis, []

    def zfila(s, cv):
        """Fila Z = -c + sum(c_B * fila_basica): deja la base con costo reducido 0."""
        z = [-v for v in cv] + [F(0)]
        for i, bn in enumerate(s.basis):
            cb = cv[s.cols.index(bn)]
            if cb:
                z = [a + cb * b for a, b in zip(z, s.T[i])]
        return z

    def snap(s, z, fase, piv=None, nota=""):
        s.out.append({"fase": fase, "cols": list(s.cols), "base": list(s.basis),
                      "t": [[P(v) for v in r] for r in s.T], "z": [P(v) for v in z],
                      "piv": piv, "nota": nota})

    def pivotar(s, z, i, j):
        p = s.T[i][j]
        antes = list(s.T[i])
        s.T[i] = [v / p for v in s.T[i]]
        operaciones = [{
            "fila": s.basis[i], "indice": i, "tipo": "normalizar", "factor": P(p),
            "formula": f"F{i + 1} ({s.basis[i]}) = F{i + 1} anterior / ({fr(p)})",
            "antes": [P(v) for v in antes], "despues": [P(v) for v in s.T[i]],
        }]
        for k in range(len(s.T)):
            if k != i and s.T[k][j] != 0:
                f = s.T[k][j]
                s.T[k] = [a - f * b for a, b in zip(s.T[k], s.T[i])]
                operaciones.append({
                    "fila": s.basis[k], "indice": k, "tipo": "eliminar", "factor": P(f),
                    "formula": (f"F{k + 1} ({s.basis[k]}) = F{k + 1} anterior "
                                f"- ({fr(f)}) × F{i + 1} normalizada"),
                })
        if z[j] != 0:
            f = z[j]
            antes_z = list(z)
            z[:] = [a - f * b for a, b in zip(z, s.T[i])]
            operaciones.append({
                "fila": "Z", "indice": None, "tipo": "eliminar", "factor": P(f),
                "formula": f"Z = Z anterior - ({fr(f)}) × F{i + 1} normalizada",
                "antes": [P(v) for v in antes_z], "despues": [P(v) for v in z],
            })
        s.basis[i] = s.cols[j]
        return operaciones

    def correr(s, z, fase, permitidas):
        degen = pasos = 0
        while True:
            neg = [j for j in permitidas if z[j] < 0]
            if not neg:
                s.snap(z, fase, nota="Óptima: la fila Z no tiene coeficientes negativos.")
                return "optimo"
            # Dantzig; tras 5 pivotes degenerados seguidos se usa Bland (anticiclado)
            regla = "Bland" if degen >= 5 else "Dantzig"
            j = min(neg) if degen >= 5 else min(neg, key=lambda j: (z[j], j))
            q = [s.T[i][-1] / s.T[i][j] if s.T[i][j] > 0 else None for i in range(len(s.T))]
            el = [i for i, v in enumerate(q) if v is not None]
            if not el:
                s.snap(z, fase, {"i": None, "j": j, "entra": s.cols[j], "sale": None,
                                 "q": [None] * len(q), "pivote": None, "zc": P(z[j]), "min": None,
                                 "regla": regla, "operaciones": []},
                       "No acotado: la columna de entrada no tiene coeficientes positivos.")
                return "no_acotado"
            mn = min(q[i] for i in el)
            i = min((i for i in el if q[i] == mn), key=lambda i: s.cols.index(s.basis[i]))
            piv = {"i": i, "j": j, "entra": s.cols[j], "sale": s.basis[i],
                   "q": [P(v) for v in q], "pivote": P(s.T[i][j]),
                   "zc": P(z[j]), "min": P(mn), "regla": regla}
            s.snap(z, fase, piv)
            piv["operaciones"] = s.pivotar(z, i, j)
            degen = degen + 1 if mn == 0 else 0
            pasos += 1
            if pasos > MAX_ITER:
                raise ValueError(f"Se superó el límite de {MAX_ITER} iteraciones.")


# ----------------------------------------------------------------- geometria
def geometria_2d(d, resultado):
    """Vertices reales y dimension de la region, con aritmetica exacta.

    Las variables son no negativas, igual que en el solver. Las esquinas de
    la ventana de dibujo nunca son vertices del modelo. La acotacion de la
    region se calcula independientemente de la de la funcion objetivo.
    """
    c = [num(v) for v in d["c"]]
    if len(c) != 2:
        return None
    restricciones = [(num(r["a"][0]), num(r["a"][1]), r["op"], num(r["b"]))
                     for r in d["restricciones"]]

    def cumple(x, y):
        if x < 0 or y < 0:
            return False
        for a, b, op, rhs in restricciones:
            v = a * x + b * y
            if (op == "<=" and v > rhs or op == ">=" and v < rhs
                    or op == "=" and v != rhs):
                return False
        return True

    rectas = [(a, b, rhs) for a, b, _, rhs in restricciones if a or b]
    rectas += [(F(1), F(0), F(0)), (F(0), F(1), F(0))]
    puntos = set()
    for i, (a, b, rhs) in enumerate(rectas):
        for u, v, termino in rectas[i + 1:]:
            det = a * v - b * u
            if det:
                x, y = (rhs * v - b * termino) / det, (a * termino - rhs * u) / det
                if cumple(x, y):
                    puntos.add((x, y))
    # Todo poliedro no vacio contenido en el primer cuadrante tiene vertices.
    if not puntos:
        return {"vertices": [], "region_no_acotada": False, "dimension": -1}

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    ordenados = sorted(puntos)
    if len(ordenados) > 1:
        inferior, superior = [], []
        for cadena, recorrido in ((inferior, ordenados), (superior, reversed(ordenados))):
            for p in recorrido:
                while len(cadena) >= 2 and cruz(cadena[-2], cadena[-1], p) <= 0:
                    cadena.pop()
                cadena.append(p)
        vertices = inferior[:-1] + superior[:-1]
    else:
        vertices = ordenados

    # Una direccion de recesion no nula del primer cuadrante se normaliza
    # como (t, 1-t), 0 <= t <= 1. Las restricciones homogeneas reducen este
    # intervalo; sus extremos son suficientes para obtener su dimension.
    lo, hi = F(0), F(1)
    for a, b, op, _ in restricciones:
        desigualdades = [(a - b, -b)]
        if op == ">=":
            desigualdades = [(b - a, b)]
        elif op == "=":
            desigualdades.append((b - a, b))
        for coef, rhs in desigualdades:
            if coef > 0:
                hi = min(hi, rhs / coef)
            elif coef < 0:
                lo = max(lo, rhs / coef)
            elif rhs < 0:
                lo, hi = F(1), F(0)
    no_acotada = lo <= hi
    direcciones = [(lo, 1 - lo), (hi, 1 - hi)] if no_acotada else []
    base = vertices[0]
    vectores = [(x - base[0], y - base[1]) for x, y in vertices[1:]] + direcciones
    vectores = [v for v in vectores if v != (0, 0)]
    dimension = 0
    if vectores:
        dimension = 2 if any(cruz((0, 0), vectores[0], v) != 0 for v in vectores[1:]) else 1
    z_optimo = F(resultado["Z"][0]) if resultado.get("estado") == "optimo" else None
    return {
        "vertices": [{"x": P(x), "y": P(y), "z": P(c[0] * x + c[1] * y),
                      "optimo": z_optimo is not None and c[0] * x + c[1] * y == z_optimo}
                     for x, y in vertices],
        "region_no_acotada": no_acotada,
        "dimension": dimension,
    }


def punto_optimo_alterno(tab, z, n):
    """Busca otro punto optimo con una sola optimizacion, aun con degeneracion.

    En la cara optima, toda columna de costo reducido estricto permanece en
    cero. Un punto distinto de esta solucion basica debe tener alguna de sus
    variables no basicas positivas. Maximizar su suma prueba si existe tal
    punto; pivotar a otra base con el mismo punto no cuenta como multiplicidad.
    """
    candidatas = [j for j, cn in enumerate(tab.cols) if cn not in tab.basis and z[j] == 0]
    if not candidatas:
        return None
    auxiliar = Tab([list(r) for r in tab.T], list(tab.cols), list(tab.basis))
    objetivo = [F(int(j in candidatas)) for j in range(len(tab.cols))]
    za = auxiliar.zfila(objetivo)
    estado = auxiliar.correr(za, 0, [j for j in range(len(tab.cols)) if z[j] == 0])
    if estado == "optimo" and za[-1] == 0:
        return None
    valores = {cn: F(0) for cn in auxiliar.cols}
    for i, cn in enumerate(auxiliar.basis):
        valores[cn] = auxiliar.T[i][-1]
    if estado == "no_acotado":
        # Un paso unitario sobre el rayo es otro punto factible de la cara.
        j = auxiliar.out[-1]["piv"]["j"]
        valores[auxiliar.cols[j]] = F(1)
        for i, cn in enumerate(auxiliar.basis):
            valores[cn] -= auxiliar.T[i][j]
    return [valores[f"X{j + 1}"] for j in range(n)]


# ----------------------------------------------------------------- resolver
def resolver(d):
    sentido = d.get("sentido", "max")
    if sentido not in ("max", "min"):
        raise ValueError("El sentido debe ser max o min.")
    sg = 1 if sentido == "max" else -1
    c = [num(v) for v in d["c"]]
    n, filas = len(c), d["restricciones"]
    if not 1 <= n <= MAX_VARS or not 1 <= len(filas) <= MAX_REST:
        raise ValueError(f"Use entre 1 y {MAX_VARS} variables y entre 1 y {MAX_REST} restricciones.")

    A, b, ops, flip, orig = [], [], [], [], []
    for r in filas:
        a, rhs, op = [num(v) for v in r["a"]], num(r["b"]), r["op"]
        if len(a) != n or op not in ("<=", ">=", "="):
            raise ValueError("Restricción mal formada.")
        orig.append((a, rhs, op))
        f = 1
        if rhs < 0:  # se multiplica por -1 para tener b >= 0
            a, rhs, op, f = [-v for v in a], -rhs, {"<=": ">=", ">=": "<=", "=": "="}[op], -1
        A.append(a), b.append(rhs), ops.append(op), flip.append(f)
    m = len(A)

    cols = [f"X{j + 1}" for j in range(n)]
    extra = [(f"S{i + 1}", i, 1) for i in range(m) if ops[i] == "<="]
    extra += [(f"E{i + 1}", i, -1) for i in range(m) if ops[i] == ">="]
    arts = [(f"R{i + 1}", i, 1) for i in range(m) if ops[i] != "<="]
    cols += [e[0] for e in extra + arts]
    T = [A[i] + [F(0)] * (len(cols) - n) + [b[i]] for i in range(m)]
    for nombre, i, cf in extra + arts:
        T[i][cols.index(nombre)] = F(cf)
    basis = [None] * m
    for nombre, i, cf in extra + arts:
        if cf == 1:
            basis[i] = nombre
    M0 = [list(r) for r in T]
    cols0 = list(cols)
    base_inicial = list(basis)
    tipos = {"X": "decision", "S": "holgura", "E": "exceso", "R": "artificial"}
    estandar = {
        "sentido": sentido, "c": [P(v) for v in c], "columnas": list(cols0),
        "filas": [{"a": [P(v) for v in fila[:-1]], "b": P(fila[-1]),
                   "original": i, "invertida": flip[i] == -1} for i, fila in enumerate(M0)],
        "variables": [{"nombre": cn, "tipo": tipos[cn[0]]} for cn in cols0],
        "base_inicial": base_inicial,
        "objetivo_fase1": [P(F(-1) if cn.startswith("R") else F(0)) for cn in cols0] if arts else None,
        "objetivo_interno": [P(sg * c[j] if j < n else F(0)) for j in range(len(cols0))],
    }
    tab = Tab(T, cols, basis)

    def salida(estado, mensaje, extra=None):
        out = {"estado": estado, "mensaje": mensaje, "tablas": tab.out,
               "fases": 2 if arts else 1, "estandar": estandar}
        out.update(extra or {})
        if n == 2:
            out["grafica"] = geometria_2d(d, out)
        return out

    fase = 1 if arts else 0
    if arts:
        z = tab.zfila([F(-1) if cn.startswith("R") else F(0) for cn in cols])
        tab.correr(z, 1, [j for j, cn in enumerate(cols) if not cn.startswith("R")])
        if z[-1] < 0:
            return salida("infactible", "No existe solución factible: las restricciones se contradicen "
                          f"(la suma de artificiales no baja de {fr(-z[-1])}).")
        redundante = False
        for i in reversed(range(len(tab.T))):
            if tab.basis[i].startswith("R"):
                j = next((j for j, cn in enumerate(tab.cols)
                          if not cn.startswith("R") and tab.T[i][j] != 0), None)
                if j is None:
                    del tab.T[i], tab.basis[i]
                    redundante = True
                else:
                    tab.pivotar(z, i, j)
        keep = [j for j, cn in enumerate(tab.cols) if not cn.startswith("R")]
        tab.T = [[r[j] for j in keep] + [r[-1]] for r in tab.T]
        tab.cols = [tab.cols[j] for j in keep]
        fase = 2
    else:
        redundante = False
    cv = [sg * c[j] if j < n else F(0) for j in range(len(tab.cols))]
    z = tab.zfila(cv)
    estado = tab.correr(z, fase, list(range(len(tab.cols))))
    if estado == "no_acotado":
        return salida("no_acotado", "La función objetivo mejora indefinidamente: "
                      "ninguna restricción limita la variable que entra.")

    # ---- solucion
    val = {cn: F(0) for cn in tab.cols}
    for i, bn in enumerate(tab.basis):
        val[bn] = tab.T[i][-1]
    x = [val[f"X{j + 1}"] for j in range(n)]
    Z = sg * z[-1]
    rest = []
    for i, (a, rhs, op) in enumerate(orig):
        act = sum(p * q for p, q in zip(a, x))
        rest.append({"act": P(act), "b": P(rhs), "op": op, "holgura": P(abs(rhs - act)),
                     "activa": act == rhs})
    alternos = [cn for j, cn in enumerate(tab.cols) if cn not in tab.basis and z[j] == 0]
    punto_alterno = punto_optimo_alterno(tab, z, n)
    degenerada = any(fila[-1] == 0 for fila in tab.T)
    rangos_objetivo = []
    for j in range(n):
        nombre = f"X{j + 1}"
        fila_basica = tab.basis.index(nombre) if nombre in tab.basis else None
        lo, hi = None, None
        for k, cn in enumerate(tab.cols):
            if cn in tab.basis:
                continue
            # z_k(delta) = z_k + pendiente * delta >= 0 para el objetivo
            # interno de maximizacion; delta cambia el coeficiente original.
            pendiente = sg * ((tab.T[fila_basica][k] if fila_basica is not None else F(0)) - int(k == j))
            if pendiente > 0:
                limite = -z[k] / pendiente
                lo = limite if lo is None else max(lo, limite)
            elif pendiente < 0:
                limite = -z[k] / pendiente
                hi = limite if hi is None else min(hi, limite)
        rangos_objetivo.append({
            "n": nombre, "actual": P(c[j]), "lo": P(None if lo is None else c[j] + lo),
            "hi": P(None if hi is None else c[j] + hi),
            "aumento": P(hi), "disminucion": P(None if lo is None else -lo),
        })
    nota_sensibilidad = ("Los rangos son inclusivos y cambian un coeficiente a la vez, manteniendo los demás "
                         "datos y la base final. Los rangos de c conservan su optimalidad; los del lado "
                         "derecho conservan su factibilidad. Fuera de un rango se debe resolver de nuevo.")
    if degenerada:
        nota_sensibilidad += (" La base es degenerada: otra base puede representar el mismo punto; "
                              "un rango de esta base no describe necesariamente todos los cambios "
                              "para los que ese punto sigue siendo óptimo.")
    if redundante:
        nota_sensibilidad += (" Se eliminaron ecuaciones redundantes en la Fase 1. No se presentan "
                              "precios sombra ni rangos del lado derecho de las filas originales: "
                              "sus perturbaciones pueden romper esa dependencia y los multiplicadores "
                              "no quedan determinados por una base cuadrada original. Los costos "
                              "reducidos y los rangos de c sí corresponden al sistema reducido.")
    extras = {
        "Z": P(Z), "x": [[f"X{j + 1}", *P(x[j])] for j in range(n)], "restr": rest,
        "alternos": alternos, "multiples": punto_alterno is not None,
        "punto_alterno": None if punto_alterno is None else
                         [[f"X{j + 1}", *P(punto_alterno[j])] for j in range(n)],
        "degenerada": degenerada, "base_final": list(tab.basis),
        "rangos_objetivo": rangos_objetivo, "nota_sensibilidad": nota_sensibilidad,
        "costos": [{"n": f"X{j + 1}", "basica": f"X{j + 1}" in tab.basis,
                    "v": P(-sg * z[j])} for j in range(n)],
        "duales": None,
    }
    if not redundante:  # precios sombra y rangos con B^-1
        idx = [cols0.index(bn) for bn in tab.basis]
        B = [[M0[i][k] for k in idx] for i in range(m)]
        y = lineal([[B[i][k] for i in range(m)] for k in range(m)],
                   [cv[tab.cols.index(bn)] for bn in tab.basis])
        xb = [tab.T[k][-1] for k in range(m)]
        duales = []
        for i in range(m):
            dv = lineal(B, [F(int(k == i)) for k in range(m)])
            lo = max((-xb[k] / dv[k] for k in range(m) if dv[k] > 0), default=None)
            hi = min((xb[k] / -dv[k] for k in range(m) if dv[k] < 0), default=None)
            bo = orig[i][1]
            if flip[i] == 1:
                rng = [None if lo is None else bo + lo, None if hi is None else bo + hi]
            else:
                rng = [None if hi is None else bo - hi, None if lo is None else bo - lo]
            duales.append({"precio": P(sg * flip[i] * y[i]), "lo": P(rng[0]), "hi": P(rng[1])})
        extras["duales"] = duales
    return salida("optimo", "Solución óptima encontrada.", extras)


# ----------------------------------------------------------------- validacion independiente
def validar_modelo(d, resultado=None):
    """Contrasta el resultado con HiGHS y comprueba el certificado exacto.

    SciPy usa flotantes; las tablas, pivotes y la solucion propia se revisan
    con Fraction. No se exige el mismo vector cuando hay optimos multiples.
    """
    from math import isclose, isfinite

    r = resolver(d) if resultado is None else resultado
    sentido = d.get("sentido", "max")
    if sentido not in ("max", "min"):
        raise ValueError("El sentido debe ser max o min.")
    c = [num(v) for v in d["c"]]
    n = len(c)
    filas = [([num(v) for v in f["a"]], num(f["b"]), f["op"])
             for f in d["restricciones"]]
    if not 1 <= n <= MAX_VARS or not 1 <= len(filas) <= MAX_REST:
        raise ValueError("Cantidad de variables o restricciones no admitida.")
    if any(len(a) != n or op not in ("<=", ">=", "=") for a, b, op in filas):
        raise ValueError("Restriccion mal formada.")
    sg = 1 if sentido == "max" else -1
    cert = {"factible": None, "objetivo_consistente": None, "tabla_optima": None,
            "tablas_consistentes": True, "pivotes_validos": True,
            "estado_certificado": False, "tablas_revisadas": 0,
            "pivotes_revisados": 0, "errores": []}
    cmp = {"disponible": False, "metodo": "scipy.optimize.linprog (HiGHS)",
           "estado_simplex": r.get("estado"), "estado_scipy": None,
           "coincide_estado": None, "z_simplex": r.get("Z"), "z_scipy": None,
           "coincide_objetivo": None, "x_scipy": None, "factible_scipy": None,
           "tolerancia": {"absoluta": 1e-7, "relativa": 1e-7}, "mensaje": ""}
    salida = {"estado": "error", "ok": None, "mensaje": "",
              "certificado": cert, "comparacion": cmp}

    def fallo(campo, mensaje):
        cert[campo] = False
        if len(cert["errores"]) < 20:
            cert["errores"].append(mensaje)

    def valor(par):
        return None if par is None else F(par[0])

    def cumple(x):
        return len(x) == n and all(v >= 0 for v in x) and all(
            (sum(ai * xi for ai, xi in zip(a, x)) <= b if op == "<=" else
             sum(ai * xi for ai, xi in zip(a, x)) >= b if op == ">=" else
             sum(ai * xi for ai, xi in zip(a, x)) == b) for a, b, op in filas)

    def espacio_filas(matriz):
        """Forma reducida canonica: permite eliminar filas redundantes."""
        a = [list(f) for f in matriz]
        i = 0
        for j in range(len(a[0]) if a else 0):
            k = next((k for k in range(i, len(a)) if a[k][j]), None)
            if k is None:
                continue
            a[i], a[k] = a[k], a[i]
            div = a[i][j]
            a[i] = [v / div for v in a[i]]
            for k in range(len(a)):
                if k != i and a[k][j]:
                    mul = a[k][j]
                    a[k] = [v - mul * w for v, w in zip(a[k], a[i])]
            i += 1
            if i == len(a):
                break
        return [f for f in a if any(f)]

    x = None
    try:
        if r.get("estado") == "optimo":
            valores = {v[0]: F(v[1]) for v in r["x"]}
            x = [valores[f"X{j + 1}"] for j in range(n)]
            cert["factible"] = cumple(x)
            cert["objetivo_consistente"] = sum(a * b for a, b in zip(c, x)) == valor(r["Z"])
            if not cert["factible"]:
                fallo("factible", "La solucion no satisface exactamente todas las restricciones.")
            if not cert["objetivo_consistente"]:
                fallo("objetivo_consistente", "Z no coincide con la funcion objetivo evaluada en la solucion.")

        normal = [([-v for v in a], -b, {"<=": ">=", ">=": "<=", "=": "="}[op])
                  if b < 0 else (a, b, op) for a, b, op in filas]
        extras = [(f"S{i + 1}", i, F(1)) for i, (_, _, op) in enumerate(normal) if op == "<="]
        extras += [(f"E{i + 1}", i, F(-1)) for i, (_, _, op) in enumerate(normal) if op == ">="]
        artes = [(f"R{i + 1}", i, F(1)) for i, (_, _, op) in enumerate(normal) if op != "<="]
        originales = [f"X{j + 1}" for j in range(n)] + [v[0] for v in extras + artes]
        inicial = [list(a) + [F(0)] * (len(originales) - n) + [b] for a, b, _ in normal]
        base_inicial = [None] * len(filas)
        for nombre, i, coef in extras + artes:
            inicial[i][originales.index(nombre)] = coef
            if coef == 1:
                base_inicial[i] = nombre
        tablas = r.get("tablas", [])
        if not tablas:
            raise ValueError("No hay tablas que permitan comprobar el procedimiento.")
        leidas = []
        for t in tablas:
            leidas.append((t["cols"], t["base"], [[valor(v) for v in f] for f in t["t"]],
                           [valor(v) for v in t["z"]], t["fase"]))
        if leidas[0][:3] != (originales, base_inicial, inicial) or leidas[0][4] != (1 if artes else 0):
            fallo("tablas_consistentes", "La tabla inicial no corresponde al modelo normalizado.")

        for k, (cols, base, T, z, fase) in enumerate(leidas):
            cert["tablas_revisadas"] += 1
            if (len(base) != len(T) or len(set(base)) != len(base) or
                    len(set(cols)) != len(cols) or len(z) != len(cols) + 1 or
                    any(len(f) != len(cols) + 1 for f in T) or any(b not in cols for b in base)):
                raise ValueError(f"Estructura invalida en la tabla {k + 1}.")
            if any(f[-1] < 0 for f in T) or any(
                    T[i][cols.index(b)] != int(i == h) for h, b in enumerate(base) for i in range(len(T))):
                fallo("tablas_consistentes", f"La base de la tabla {k + 1} no es factible o canonica.")
            costos = [F(-1) if cn.startswith("R") else F(0) for cn in cols] if fase == 1 else [
                sg * c[int(cn[1:]) - 1] if cn.startswith("X") else F(0) for cn in cols]
            z_esperada = [-v for v in costos] + [F(0)]
            for i, bn in enumerate(base):
                z_esperada = [v + costos[cols.index(bn)] * w for v, w in zip(z_esperada, T[i])]
            if z != z_esperada:
                fallo("tablas_consistentes", f"La fila Z de la tabla {k + 1} no corresponde a su base.")
            p = tablas[k].get("piv")
            if p:
                j, i = p["j"], p["i"]
                if not isinstance(j, int) or not 0 <= j < len(cols):
                    raise ValueError(f"Columna de pivote invalida en la tabla {k + 1}.")
                cocientes = [f[-1] / f[j] if f[j] > 0 else None for f in T]
                validos = [v for v in cocientes if v is not None]
                correcto = (z[j] < 0 and p["entra"] == cols[j] and valor(p["zc"]) == z[j]
                            and [valor(q) for q in p["q"]] == cocientes)
                if i is None:
                    correcto &= not validos and p["sale"] is None and p["pivote"] is None and p["min"] is None
                    correcto &= k == len(tablas) - 1 and r.get("estado") == "no_acotado"
                else:
                    cert["pivotes_revisados"] += 1
                    if not isinstance(i, int) or not 0 <= i < len(T):
                        raise ValueError(f"Fila de pivote invalida en la tabla {k + 1}.")
                    correcto &= bool(validos) and T[i][j] > 0
                    correcto &= p["sale"] == base[i] and valor(p["pivote"]) == T[i][j]
                    correcto &= bool(validos) and cocientes[i] == min(validos) == valor(p["min"])
                    if correcto:
                        fila = [v / T[i][j] for v in T[i]]
                        siguiente = [fila if h == i else [v - f[j] * w for v, w in zip(f, fila)]
                                     for h, f in enumerate(T)]
                        z_sig = [v - z[j] * w for v, w in zip(z, fila)]
                        base_sig = list(base)
                        base_sig[i] = cols[j]
                        if k + 1 == len(leidas) or leidas[k + 1] != (cols, base_sig, siguiente, z_sig, fase):
                            fallo("tablas_consistentes", f"La tabla posterior al pivote {k + 1} no coincide con Gauss-Jordan.")
                if not correcto:
                    fallo("pivotes_validos", f"El pivote o los cocientes de la tabla {k + 1} son incorrectos.")
            elif k + 1 < len(leidas):
                cn, bn, tn, zn, fn = leidas[k + 1]
                conservar = [i for i, cnm in enumerate(cols) if not cnm.startswith("R")]
                origen = [[f[i] for i in conservar] + [f[-1]] for f in T]
                if not (fase == 1 and fn == 2 and z[-1] == 0 and
                        cn == [cols[i] for i in conservar] and espacio_filas(origen) == espacio_filas(tn)):
                    fallo("tablas_consistentes", "El paso de Fase 1 a Fase 2 no conserva las restricciones.")

        cols, base, T, z, fase = leidas[-1]
        final_p = tablas[-1].get("piv")
        if r.get("estado") == "optimo":
            cert["tabla_optima"] = (fase in (0, 2) and not final_p and
                                     all(v >= 0 for v in z[:-1]) and sg * z[-1] == valor(r["Z"]))
            valores_finales = {cn: F(0) for cn in cols}
            valores_finales.update({bn: T[i][-1] for i, bn in enumerate(base)})
            cert["tabla_optima"] &= x == [valores_finales[f"X{j + 1}"] for j in range(n)]
            cert["estado_certificado"] = cert["tabla_optima"]
            if not cert["tabla_optima"]:
                fallo("tabla_optima", "La tabla final no certifica el optimo informado.")
        elif r.get("estado") == "infactible":
            cert["estado_certificado"] = (fase == 1 and not final_p and z[-1] < 0 and
                                          all(z[j] >= 0 for j, cn in enumerate(cols) if not cn.startswith("R")))
        elif r.get("estado") == "no_acotado":
            cert["estado_certificado"] = bool(fase in (0, 2) and final_p and final_p["i"] is None
                                               and cert["pivotes_validos"])
        if not cert["estado_certificado"]:
            fallo("estado_certificado", "Las tablas no certifican el estado final informado.")
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        cert["pivotes_validos"] = False
        fallo("tablas_consistentes", "No se pudo comprobar el certificado: " + str(exc))

    try:
        from scipy.optimize import linprog
    except ImportError:
        salida.update(estado="no_disponible", mensaje="La comprobacion externa no esta disponible: falta SciPy en este Python.")
        cmp["mensaje"] = "Instala SciPy en el mismo entorno de Python y vuelve a validar. El certificado exacto se muestra por separado."
        return salida

    cmp["disponible"] = True
    # linprog minimiza y solo admite desigualdades <=: negar ambos lados de >=.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    aub, bub, aeq, beq = [], [], [], []
    for a, b, op in filas:
        if op == "=":
            aeq.append([float(v) for v in a])
            beq.append(float(b))
        else:
            signo = -1 if op == ">=" else 1
            aub.append([float(signo * v) for v in a])
            bub.append(float(signo * b))
    try:
        externo = linprog([-sg * float(v) for v in c], A_ub=aub or None, b_ub=bub or None,
                          A_eq=aeq or None, b_eq=beq or None, bounds=(0, None), method="highs",
                          options={"time_limit": 15, "primal_feasibility_tolerance": 1e-7,
                                   "dual_feasibility_tolerance": 1e-7})
        cmp["codigo_scipy"] = int(externo.status)
        cmp["estado_scipy"] = {0: "optimo", 2: "infactible", 3: "no_acotado"}.get(externo.status, "indeterminado")
        cmp["mensaje"] = {0: "HiGHS encontro una solucion optima.", 1: "HiGHS alcanzo su limite de tiempo o iteraciones.",
                          2: "HiGHS detecto restricciones incompatibles.", 3: "HiGHS detecto un objetivo no acotado.",
                          4: "HiGHS no pudo concluir por un problema numerico."}.get(externo.status, "HiGHS no pudo concluir.")
        if cmp["estado_scipy"] == "indeterminado":
            salida["mensaje"] = "No se pudo completar la comparacion independiente. " + cmp["mensaje"]
            return salida
        cmp["coincide_estado"] = cmp["estado_scipy"] == r.get("estado")
        if externo.status == 0:
            xs = [float(v) for v in externo.x]
            zs = -sg * float(externo.fun)
            if not all(isfinite(v) for v in xs + [zs]):
                raise ValueError("HiGHS devolvio valores no finitos.")
            cmp["x_scipy"], cmp["z_scipy"] = xs, zs
            atol, rtol = cmp["tolerancia"]["absoluta"], cmp["tolerancia"]["relativa"]
            factible_scipy = all(v >= -atol for v in xs)
            for a, b, op in filas:
                actividad = sum(float(ai) * xi for ai, xi in zip(a, xs))
                tolerancia = atol + rtol * max(abs(float(b)), abs(actividad))
                diferencia = actividad - float(b)
                factible_scipy &= (diferencia <= tolerancia if op == "<=" else
                                   diferencia >= -tolerancia if op == ">=" else abs(diferencia) <= tolerancia)
            cmp["factible_scipy"] = bool(factible_scipy)
            if r.get("estado") == "optimo":
                cmp["coincide_objetivo"] = isclose(float(valor(r["Z"])), zs, rel_tol=rtol, abs_tol=atol)
        certificado_ok = all(v is not False for clave, v in cert.items() if clave != "errores") and not cert["errores"]
        ok = bool(certificado_ok and cmp["coincide_estado"] and cmp["coincide_objetivo"] is not False
                  and cmp["factible_scipy"] is not False)
        salida.update(estado="correcto" if ok else "discrepancia", ok=ok,
                      mensaje="Resultado verificado: el certificado exacto y HiGHS coinciden." if ok else
                      "Se encontro una diferencia. Revisa el certificado y la comparacion antes de usar el resultado.")
        return salida
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError, RuntimeError) as exc:
        cmp["mensaje"] = "No se pudo completar HiGHS: " + str(exc)
        salida["mensaje"] = "La validacion externa no pudo concluir; no se ha confirmado el resultado."
        return salida


# ----------------------------------------------------------------- pruebas
PRUEBAS = [
    ("max", [3, 5], [([1, 0], "<=", 4), ([0, 2], "<=", 12), ([3, 2], "<=", 18)], "optimo", 36),
    ("max", [5, 4], [([6, 4], "<=", 24), ([1, 2], "<=", 6)], "optimo", 21),
    ("min", ["0.12", "0.15"], [([60, 60], ">=", 300), ([12, 6], ">=", 36), ([10, 30], ">=", 90)],
     "optimo", F(66, 100)),
    ("max", [1, 1], [([1, 1], "<=", 2), ([1, 1], ">=", 5)], "infactible", None),
    ("max", [1, 1], [([1, -1], "<=", 1)], "no_acotado", None),
    ("min", [1, 1], [([-1, -1], "<=", -4), ([1, 0], "=", 1)], "optimo", 4),
    ("max", [1500, 1800], [([1, 1], "<=", 48), ([6, 8], "<=", 360), ([200, 150], "<=", 8000)],
     "optimo", 82800),
]


def probar_geometria():
    # nombre, sentido, objetivo, restricciones, vertices, dimension,
    # region no acotada, vertices optimos
    casos = [
        ("siembra", "max", [1500, 1800],
         [([1, 1], "<=", 48), ([6, 8], "<=", 360), ([200, 150], "<=", 8000)],
         {(0, 0), (40, 0), (16, 32), (12, 36), (0, 45)}, 2, False, {(12, 36)}),
        ("fracciones y optimos multiples", "max", [1, 1], [([3, 3], "<=", 2)],
         {(0, 0), (F(2, 3), 0), (0, F(2, 3))}, 2, False, {(F(2, 3), 0), (0, F(2, 3))}),
        ("region ilimitada con minimo finito", "min", [1, 1], [([1, 1], ">=", 2)],
         {(2, 0), (0, 2)}, 2, True, {(2, 0), (0, 2)}),
        ("objetivo no acotado", "max", [1, 1], [([1, -1], "<=", 1)],
         {(0, 0), (1, 0)}, 2, True, set()),
        ("igualdad y lado derecho negativo", "min", [1, 1],
         [([-1, -1], "<=", -4), ([1, 0], "=", 1)],
         {(1, 3)}, 1, True, {(1, 3)}),
        ("segmento", "max", [2, 1], [([1, 1], "=", 3)],
         {(3, 0), (0, 3)}, 1, False, {(3, 0)}),
        ("punto degenerado", "max", [2, 1], [([1, 1], "<=", 0)],
         {(0, 0)}, 0, False, {(0, 0)}),
        ("rayo sobre un eje", "min", [0, 1], [([1, 0], "<=", 0)],
         {(0, 0)}, 1, True, {(0, 0)}),
        ("restriccion nula redundante", "min", [1, 1], [([0, 0], "=", 0)],
         {(0, 0)}, 2, True, {(0, 0)}),
        ("restriccion nula contradictoria", "max", [1, 1], [([0, 0], "<=", -1)],
         set(), -1, False, set()),
        ("restricciones incompatibles", "max", [1, 1],
         [([1, 1], "<=", 2), ([1, 1], ">=", 5)], set(), -1, False, set()),
        ("rectangulo sin esquinas artificiales", "min", [1, 1],
         [([1, 0], ">=", 2), ([1, 0], "<=", 4), ([0, 1], ">=", 3), ([0, 1], "<=", 5)],
         {(2, 3), (4, 3), (4, 5), (2, 5)}, 2, False, {(2, 3)}),
    ]
    fallos = 0
    for nombre, sentido, c, rs, esperados, dimension, no_acotada, optimos in casos:
        d = {"sentido": sentido, "c": c,
             "restricciones": [{"a": a, "op": op, "b": b} for a, op, b in rs]}
        r = resolver(d)
        g = r["grafica"]
        puntos = [(F(v["x"][0]), F(v["y"][0])) for v in g["vertices"]]
        mejores = {p for p, v in zip(puntos, g["vertices"]) if v["optimo"]}
        valores_ok = all(F(v["z"][0]) == num(c[0]) * p[0] + num(c[1]) * p[1]
                         for p, v in zip(puntos, g["vertices"]))
        orden_ok = True
        for i in range(len(puntos)) if len(puntos) >= 3 else []:
            a, b, p = puntos[i - 2], puntos[i - 1], puntos[i]
            orden_ok &= (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0
        ok = (set(puntos) == esperados and len(puntos) == len(esperados)
              and g["dimension"] == dimension and g["region_no_acotada"] == no_acotada
              and mejores == optimos and valores_ok and orden_ok)
        fallos += not ok
        print(f"Geometria {nombre}: {'OK' if ok else 'FALLO'}")
    return fallos


def probar():
    fallos = 0
    for k, (s, c, rs, est, z) in enumerate(PRUEBAS, 1):
        r = resolver({"sentido": s, "c": c, "restricciones": [{"a": a, "op": o, "b": b} for a, o, b in rs]})
        ok = r["estado"] == est and (z is None or F(r["Z"][0]) == F(z))
        fallos += not ok
        print(f"Caso {k}: {r['estado']:<10} Z={r.get('Z', ['-'])[0]:<6} {'OK' if ok else 'FALLO'}")
    return fallos + probar_geometria()


# ----------------------------------------------------------------- servidor
class Handler(BaseHTTPRequestHandler):
    def _enviar(s, codigo, cuerpo, tipo):
        s.send_response(codigo)
        s.send_header("Content-Type", tipo)
        s.send_header("Content-Length", str(len(cuerpo)))
        s.send_header("X-Content-Type-Options", "nosniff")
        s.end_headers()
        s.wfile.write(cuerpo)

    def do_GET(s):
        if s.path in ("/", "/index.html"):
            s._enviar(200, HTML.encode(), "text/html; charset=utf-8")
        else:
            s._enviar(404, b"No encontrado", "text/plain")

    def do_POST(s):
        if s.path not in ("/api/resolver", "/api/validar"):
            return s._enviar(404, b"No encontrado", "text/plain")
        try:
            n = int(s.headers.get("Content-Length", 0))
            if n > 200_000:
                raise ValueError("Solicitud demasiado grande.")
            modelo = json.loads(s.rfile.read(n))
            operacion = validar_modelo if s.path == "/api/validar" else resolver
            out, codigo = operacion(modelo), 200
        except (ValueError, KeyError, TypeError, StopIteration) as e:
            out, codigo = {"error": str(e) or "Datos del modelo inválidos."}, 400
        s._enviar(codigo, json.dumps(out).encode(), "application/json")

    def log_message(s, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8000)
    ap.add_argument("--no-abrir", action="store_true")
    ap.add_argument("--probar", action="store_true")
    a = ap.parse_args()
    if a.probar:
        return 1 if probar() else 0
    for puerto in range(a.puerto, a.puerto + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", puerto), Handler)
            break
        except OSError:
            continue
    else:
        sys.exit("No hay puertos libres.")
    url = f"http://127.0.0.1:{puerto}"
    print(f"Simplex Web en {url}  (Ctrl+C para salir)")
    if not a.no_abrir:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    return 0


# ----------------------------------------------------------------- interfaz
HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simplex</title>
<style>
:root{--bg:#F3F4F6;--sf:#fff;--ink:#14171C;--mut:#667085;--ln:#E2E5EA;--ac:#0B7285;--acs:#DDF1F5;--ok:#2B8A3E;--er:#C92A2A;--wn:#A66A00;color-scheme:light dark}
@media(prefers-color-scheme:dark){:root{--bg:#0E1115;--sf:#161A20;--ink:#E8EAEE;--mut:#8A93A2;--ln:#262B33;--ac:#4CC3D6;--acs:#12303A;--ok:#69DB7C;--er:#FF8787;--wn:#F0C36D}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
header{max-width:1240px;margin:0 auto;padding:28px 24px 8px;display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:22px;font-weight:650;letter-spacing:-.02em;margin:0}
header p{margin:0;color:var(--mut)}
.wrap{max-width:1240px;margin:0 auto;padding:16px 24px 48px;display:grid;grid-template-columns:minmax(340px,430px) 1fr;gap:20px;align-items:start}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--sf);border:1px solid var(--ln);border-radius:14px;padding:20px;min-width:0}
h2{font-size:13px;font-weight:600;color:var(--mut);margin:20px 0 8px}
h2:first-child{margin-top:0}
.eq{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.cf{display:inline-flex;align-items:center;gap:3px}
.cf i{font-style:normal;color:var(--mut);font-size:13px}
input,select{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--ln);border-radius:8px;padding:6px 8px;width:62px;text-align:right}
select{width:auto;text-align:left}
input:focus,select:focus,button:focus-visible{outline:2px solid var(--ac);outline-offset:1px}
input.bad{border-color:var(--er)}
button{font:inherit;color:inherit;background:transparent;border:1px solid var(--ln);border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{background:var(--bg)}
button.x{border:0;color:var(--mut);padding:4px 8px}
button.main{background:var(--ac);border-color:var(--ac);color:#fff;font-weight:600;padding:10px 18px;width:100%;margin-top:14px}
@media(prefers-color-scheme:dark){button.main{color:#06222A}}
button.main:hover{filter:brightness(1.08)}
.seg{display:inline-flex;border:1px solid var(--ln);border-radius:10px;padding:3px;gap:2px}
.seg button{border:0;padding:5px 14px}
.seg button.on{background:var(--acs);color:var(--ac);font-weight:600}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--ln);margin:-6px 0 18px;overflow-x:auto}
.tabs button{border:0;border-bottom:2px solid transparent;border-radius:0;padding:8px 12px;color:var(--mut);white-space:nowrap}
.tabs button.on{color:var(--ink);border-color:var(--ac);font-weight:600}
.banner{padding:12px 14px;border-radius:10px;font-weight:600;margin-bottom:16px;background:var(--acs);color:var(--ac)}
.banner.bad{background:color-mix(in srgb,var(--er) 14%,transparent);color:var(--er)}
.big{font-size:40px;font-weight:650;letter-spacing:-.03em;line-height:1.1}
.mut{color:var(--mut)}
.note{margin:8px 0;padding-left:12px;border-left:2px solid var(--wn);color:var(--ink)}
table{border-collapse:collapse;width:100%}
.tw{overflow-x:auto}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--ln);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--mut)}
.tb td.pc,.tb th.pc{background:var(--acs)}
.tb tr.pr td,.tb tr.pr th{background:var(--acs)}
.tb td.pv{background:var(--ac);color:#fff;font-weight:700}
.tb tr.zr td,.tb tr.zr th{border-top:2px solid var(--ink);font-weight:600}
.stepbar{display:flex;gap:10px;align-items:center;margin-bottom:14px}
.stepbar input[type=range]{flex:1;width:auto;padding:0;accent-color:var(--ac)}
.cap{margin:0 0 12px;min-height:44px}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;background:var(--bg);color:var(--mut);margin-right:6px}
.empty{color:var(--mut);padding:36px 0;text-align:center}
.tools{display:flex;justify-content:flex-end;gap:6px;flex-wrap:wrap;margin:-8px 0 18px;font-size:12px}
.example{padding:12px 14px;background:var(--acs);border-radius:10px;margin:16px 0;font-size:13px}
.example p{margin:6px 0 0}
.graph-button{width:100%;margin-top:8px;color:var(--ac);border-color:var(--ac)}
.method-title{font-size:19px;color:var(--ink);margin:0 0 6px}
.model-eq{font-size:16px;font-weight:600;margin:8px 0 16px}
.graph-wrap{background:#fff;border:1px solid var(--ln);border-radius:10px;overflow:hidden}
.graph-wrap svg{display:block;width:100%;height:auto}
.opt-row{background:var(--acs);font-weight:600}
.tabs{flex-wrap:wrap;overflow:visible}
textarea{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--ln);border-radius:8px;padding:8px;resize:vertical;width:100%}
.context label{display:block;font-size:13px;margin:10px 0 4px;color:var(--mut)}
.context input:not([type=checkbox]){width:100%;text-align:left}
.context input[type=checkbox]{width:auto}
details{border:1px solid var(--ln);border-radius:10px;padding:12px;margin:14px 0}
summary{cursor:pointer;font-weight:600}
.math{font-family:Consolas,monospace;line-height:1.8;overflow-wrap:anywhere}
.resource-name{font-size:12px;color:var(--mut);margin:12px 0 5px}
.status-note{font-size:12px;color:var(--wn);margin:8px 0}
.steps-list{padding-left:22px}.steps-list li{margin:8px 0}
.check-list{display:grid;gap:8px;margin:16px 0}.check-list div{padding:10px;border:1px solid var(--ln);border-radius:8px}
.scenario{background:var(--bg);border-radius:10px;padding:14px;margin:18px 0}
.scenario select{max-width:100%}.scenario input{width:140px}
.compact-table th,.compact-table td{padding:7px;font-size:13px}
.interpretation{line-height:1.7}.interpretation li{margin:8px 0}
.graph-actions{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}
.stepbar b{min-width:100px;font-size:12px;text-align:right}
@media(max-width:520px){.wrap{padding:12px 10px 28px}.card{padding:14px}header{padding:20px 12px 4px}.tabs button{padding:8px}.stepbar{flex-wrap:wrap}.stepbar input[type=range]{min-width:100px}}
.hint{font-size:13px;color:var(--mut);margin-top:12px}
@media(prefers-reduced-motion:no-preference){.card{transition:border-color .2s}}
</style></head><body>
<header><h1>Simplex · Análisis y decisiones</h1><p>Modelos multivariables, método gráfico y sensibilidad.</p></header>
<div class="wrap">
<section class="card" id="modelo" aria-label="Modelo"></section>
<section class="card" id="salida" aria-live="polite"><div class="empty">Define el modelo y pulsa Resolver.<br>También puedes cargar un ejemplo.</div></section>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const EJ={
"Taller · 3 productos (Fase 2)":{s:'max',c:['27','22','37'],r:[[['2','1','3'],'<=','85'],[['1','2','1'],'<=','65'],[['1','1','1'],'<=','45']],meta:{titulo:'Plan de producción de un taller',descripcion:'Caso didáctico con datos supuestos. Decide cuántas mesas, sillas y estantes producir. La ganancia por unidad es $27, $22 y $37; hay 85 horas de carpintería, 65 horas de acabado y 45 unidades de embalaje. Cambia los datos para representar tu caso real.',variables:['Mesas (unidades)','Sillas (unidades)','Estantes (unidades)'],restricciones:['Carpintería (horas)','Acabado (horas)','Embalaje (unidades)'],moneda:true}},
"Siembra · maíz y frijol":{s:'max',c:['1500','1800'],r:[[['1','1'],'<=','48'],[['6','8'],'<=','360'],[['200','150'],'<=','8000']],meta:{titulo:'Plan óptimo de siembra',variables:['Maíz (hectáreas)','Frijol (hectáreas)'],restricciones:['Terreno','Mano de obra','Capital'],descripcion:'Dispones de 48 hectáreas, 360 horas de trabajo y $8,000 de capital. Cada hectárea de maíz genera $1,500 y cada hectárea de frijol, $1,800 de ganancia.',moneda:true}},
"Producción · máx":{s:'max',c:['3','5'],r:[[['1','0'],'<=','4'],[['0','2'],'<=','12'],[['3','2'],'<=','18']]},
"Producción 2 · máx":{s:'max',c:['5','4'],r:[[['6','4'],'<=','24'],[['1','2'],'<=','6']]},
"Dieta · mín (Dos Fases)":{s:'min',c:['0.12','0.15'],r:[[['60','60'],'>=','300'],[['12','6'],'>=','36'],[['10','30'],'>=','90']]},
"Óptimos múltiples":{s:'max',c:['1','1'],r:[[['1','1'],'<=','4'],[['1','0'],'<=','3']]},
"Infactible":{s:'max',c:['1','1'],r:[[['1','1'],'<=','2'],[['1','1'],'>=','5']]},
"No acotado":{s:'max',c:['1','1'],r:[[['1','-1'],'<=','1']]}};
let S={s:'max',c:['3','5'],r:[]},R=null,MD=null,tab=0,paso=0,dec=false,solicitud=0,VALID=null,validando=false,ESCENARIO=null,simulando=false;
function metadata(m,n,k){m=m&&typeof m=='object'?m:{};return {titulo:String(m.titulo||'Modelo de programación lineal').slice(0,160),descripcion:String(m.descripcion||'').slice(0,2000),variables:Array.from({length:n},(_,i)=>String(m.variables?.[i]||'Variable x'+(i+1)).slice(0,100)),restricciones:Array.from({length:k},(_,i)=>String(m.restricciones?.[i]||'Restricción '+(i+1)).slice(0,100)),moneda:m.moneda===true}}
function cargar(e){S={s:e.s,c:[...e.c],r:e.r.map(([a,o,b])=>({a:[...a],op:o,b})),meta:metadata(e.meta,e.c.length,e.r.length)};form()}
function pendiente(){const el=$('#estado-modelo');if(el)el.textContent=MD&&JSON.stringify(MD)!==JSON.stringify(S)?'Hay cambios pendientes. Resuelve para actualizar los resultados.':''}
function form(){
 const n=S.c.length;
 let h=`<div class="seg" role="group" aria-label="Objetivo"><button data-s="max" class="${S.s=='max'?'on':''}">Maximizar</button><button data-s="min" class="${S.s=='min'?'on':''}">Minimizar</button></div>
 <div class="example"><strong>${esc(S.meta.titulo)}</strong><p>${esc(S.meta.descripcion)}</p><p>${n} variables · ${S.r.length} restricciones. ${n>=3?'Modelo multivariable.':'La gráfica está disponible para dos variables.'}</p></div>
 <details class="context"><summary>Descripción y nombres del modelo</summary>
 <label for="titulo">Nombre del caso</label><input id="titulo" data-meta="titulo" maxlength="160" value="${esc(S.meta.titulo)}">
 <label for="descripcion">Contexto, unidades y supuestos</label><textarea id="descripcion" data-meta="descripcion" maxlength="2000" rows="3">${esc(S.meta.descripcion)}</textarea>
 ${S.meta.variables.map((v,i)=>`<label for="vn${i}">x${i+1}: nombre y unidad</label><input id="vn${i}" data-vn="${i}" maxlength="100" value="${esc(v)}">`).join('')}
 ${S.meta.restricciones.map((v,i)=>`<label for="rn${i}">R${i+1}: recurso o requerimiento</label><input id="rn${i}" data-rn="${i}" maxlength="100" value="${esc(v)}">`).join('')}
 <label><input type="checkbox" data-money="1" ${S.meta.moneda?'checked':''}> Mostrar Z como importe monetario ($)</label></details>
 <h2>Función objetivo</h2><div class="eq"><span>Z =</span>${S.c.map((v,j)=>cf(`data-c="${j}"`,v,j)).join('')}</div>
 <h2>Restricciones</h2>`+S.r.map((r,i)=>`<div class="resource-name">R${i+1} · ${esc(S.meta.restricciones[i])}</div><div class="eq">${r.a.map((v,j)=>cf(`data-a="${i},${j}"`,v,j)).join('')}
 <select data-o="${i}" aria-label="Tipo">${['<=','>=','='].map(o=>`<option ${r.op==o?'selected':''}>${o}</option>`).join('')}</select>
 <input data-b="${i}" value="${esc(r.b)}" inputmode="decimal" aria-label="Lado derecho"><button class="x" data-del="${i}" aria-label="Quitar restricción">×</button></div>`).join('')+
 `<div class="row"><button data-act="addr" ${S.r.length>=50?'disabled':''}>+ Restricción</button><button data-act="addv" ${n>=20?'disabled':''}>+ Variable</button><button data-act="delv" ${n<2?'disabled':''}>− Variable</button></div>
 <h2>Ejemplos</h2><select id="ej" style="width:100%;text-align:left"><option value="">Cargar un ejemplo…</option>${Object.keys(EJ).map(k=>`<option>${k}</option>`).join('')}</select>
 <button class="main" data-act="go">Resolver con Simplex</button>${n==2?'<button class="graph-button" data-act="graph">Resolver y ver método gráfico</button>':''}<p id="estado-modelo" class="status-note"></p><p class="hint">Variables continuas ≥ 0. Hasta 20 variables y 50 restricciones. Acepta decimales y fracciones (3/2). Ctrl + Enter resuelve.</p><div class="row"><button data-act="save">Guardar modelo JSON</button><button data-act="load">Cargar modelo JSON</button><input type="file" id="importar" accept=".json,application/json" hidden></div><p id="import-error" class="status-note"></p>`;
 $('#modelo').innerHTML=h;
 pendiente();
}
const cf=(at,v,j)=>`<span class="cf"><input ${at} value="${esc(v)}" inputmode="decimal" aria-label="Coeficiente de x${j+1}"><i>x<sub>${j+1}</sub></i></span>`;
$('#modelo').addEventListener('input',e=>{const d=e.target.dataset;e.target.classList.remove('bad');
 if(d.c!==undefined)S.c[d.c]=e.target.value;
 else if(d.a!==undefined){const[i,j]=d.a.split(',');S.r[i].a[j]=e.target.value}
 else if(d.b!==undefined)S.r[d.b].b=e.target.value;
 else if(d.meta!==undefined)S.meta[d.meta]=e.target.value;
 else if(d.vn!==undefined)S.meta.variables[d.vn]=e.target.value;
 else if(d.rn!==undefined)S.meta.restricciones[d.rn]=e.target.value;
 pendiente()});
$('#modelo').addEventListener('change',e=>{const d=e.target.dataset;
 if(d.o!==undefined)S.r[d.o].op=e.target.value;
 if(d.money)S.meta.moneda=e.target.checked;
 if(e.target.id=='importar')importarModelo(e.target.files?.[0]);
 if(e.target.id=='ej'&&EJ[e.target.value]){cargar(EJ[e.target.value]);resolver(S.c.length==2?2:0)}
 pendiente()});
$('#modelo').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const d=b.dataset;
 if(d.s){S.s=d.s;form()}
 else if(d.del!==undefined&&S.r.length>1){S.r.splice(d.del,1);S.meta.restricciones.splice(d.del,1);form()}
 else if(d.act=='addr'&&S.r.length<50){S.r.push({a:S.c.map(()=> '0'),op:'<=',b:'0'});S.meta.restricciones.push('Restricción '+S.r.length);form()}
 else if(d.act=='addv'&&S.c.length<20){S.c.push('0');S.r.forEach(r=>r.a.push('0'));S.meta.variables.push('Variable x'+S.c.length);form()}
 else if(d.act=='delv'&&S.c.length>1){S.c.pop();S.r.forEach(r=>r.a.pop());S.meta.variables.pop();form()}
 else if(d.act=='save')guardarJSON({version:2,modelo:S},'modelo-simplex.json')
 else if(d.act=='load')$('#importar').click()
 else if(d.act=='go')resolver()
 else if(d.act=='graph')resolver(2)});
document.addEventListener('keydown',e=>{
 if(e.ctrlKey&&e.key=='Enter')resolver();
 if(R&&tab==1&&R.tablas.length&&!/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)){
  if(e.key=='ArrowRight')ir(paso+1);if(e.key=='ArrowLeft')ir(paso-1)}});
const NUM=/^\s*-?(\d+([.,]\d*)?|[.,]\d+)(\/\d+)?\s*$|^\s*$/;
async function resolver(destino=0){
 const id=++solicitud;
 let ok=true;document.querySelectorAll('#modelo input[data-c],#modelo input[data-a],#modelo input[data-b]').forEach(i=>{const bad=!NUM.test(i.value);i.classList.toggle('bad',bad);if(bad)ok=false});
 if(!ok){$('#salida').innerHTML='<div class="banner bad">Hay valores inválidos (marcados en rojo).</div>';return}
 const modelo=JSON.parse(JSON.stringify(S));
 try{
  const r=await fetch('/api/resolver',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentido:modelo.s,c:modelo.c,restricciones:modelo.r})});
  const j=await r.json();
  if(id!==solicitud)return;
  if(!r.ok){$('#salida').innerHTML=`<div class="banner bad">${esc(j.error)}</div>`;return}
  MD=modelo;R=j;VALID=null;validando=false;ESCENARIO=null;simulando=false;tab=destino;paso=0;pintar();pendiente()
 }catch(err){if(id===solicitud)$('#salida').innerHTML='<div class="banner bad">No se pudo contactar con el servidor.</div>'}
}
const V=p=>p==null?'∞':(dec?p[1]:p[0]);
const nm=c=>`${c[0].toLowerCase()}<sub>${c.slice(1)}</sub>`;
const nombreX=(i,m=MD)=>esc(m.meta?.variables?.[i]||'Variable x'+(i+1));
const nombreR=(i,m=MD)=>'R'+(i+1)+' · '+esc(m.meta?.restricciones?.[i]||'Restricción '+(i+1));
const limite=(p,inferior=false)=>p?V(p):(inferior?'−∞':'+∞');
const estadoTexto=s=>({optimo:'Óptimo',infactible:'Infactible',no_acotado:'No acotado'}[s]||s||'Sin resultado');
function guardarJSON(datos,nombre){const url=URL.createObjectURL(new Blob([JSON.stringify(datos,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=nombre;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
async function importarModelo(file){if(!file)return;try{if(file.size>10000000)throw Error('El archivo supera 10 MB.');const data=JSON.parse(await file.text()),m=data.modelo||data;
 if(!['max','min'].includes(m.s)||!Array.isArray(m.c)||m.c.length<1||m.c.length>20||!Array.isArray(m.r)||m.r.length<1||m.r.length>50)throw Error('El JSON no contiene un modelo válido.');
 const numero=v=>{if(!['number','string'].includes(typeof v)||!NUM.test(String(v))||!Number.isFinite(nf(v)))throw Error('Hay un coeficiente inválido.');return String(v)};
 const c=m.c.map(numero),r=m.r.map(f=>{if(!Array.isArray(f.a)||f.a.length!==c.length||!['<=','>=','='].includes(f.op))throw Error('Hay restricciones mal formadas.');return {a:f.a.map(numero),op:f.op,b:numero(f.b)}});
 S={s:m.s,c,r,meta:metadata(m.meta,c.length,r.length)};form();await resolver();
 }catch(e){$('#import-error').textContent='No se pudo cargar: '+e.message}}
function expresion(columnas,coef){let terms=[];coef.forEach((v,i)=>{const exact=Array.isArray(v)?v[0]:String(v),f=nf(exact);if(!f)return;const texto=Array.isArray(v)?V(v):exact,abs=String(texto).replace(/^-/,'');terms.push((f<0?' − ':terms.length?' + ':'')+(Math.abs(f)==1?'':esc(abs)+'·')+nm(columnas[i]))});return terms.join('').trim()||'0'}
function pintar(){
 const T=[[4,'Modelo estándar'],[0,'Resultado'],[1,'Iteraciones'],[2,'Método gráfico'],[3,'Sensibilidad'],[5,'Validación'],[6,'Decisiones']];
 const bad=R.estado!='optimo';
 let h=`<div class="tabs" role="tablist">${T.map(([i,t])=>`<button role="tab" aria-selected="${tab==i}" class="${tab==i?'on':''}" data-t="${i}">${t}</button>`).join('')}</div>
 <div class="tools"><button data-dec="1">${dec?'Ver fracciones':'Ver decimales'}</button><button data-json="1">Descargar JSON</button></div>`;
 h+=[vRes,vIter,vGraf,vSens,vModelo,vValidacion,vDecisiones][tab]();
 $('#salida').innerHTML=h;
 const sl=$('#sl');if(sl)sl.oninput=()=>ir(+sl.value);
 const iso=$('#iso');if(iso)iso.oninput=()=>dibujaIso(+iso.value);
 const sc=$('#sc-campo');if(sc)sc.onchange=()=>{const[t,i]=sc.value.split(':');$('#sc-valor').value=t=='b'?MD.r[+i].b:MD.c[+i]};
}
$('#salida').addEventListener('click',e=>{const b=e.target.closest('button');if(!b||!R)return;const d=b.dataset;
 if(d.t!==undefined){tab=+d.t;pintar()}
 if(d.dec){dec=!dec;pintar()}
 if(d.step)ir(paso+ +d.step);
 if(d.goto!==undefined)ir(+d.goto);
 if(d.validar)validar();
 if(d.simular)simular();
 if(d.svg)descargarGrafica();
 if(d.opt&&G){const iso=$('#iso');if(iso)iso.value=G.Z;dibujaIso(G.Z)}
 if(d.json)guardarJSON({version:2,modelo:MD,resultado:R,validacion:VALID,escenario:ESCENARIO},'simplex-resultados.json')});
function vRes(){
 if(R.estado!='optimo')return `<div class="banner bad">${esc(R.mensaje)}</div><p class="mut">Revisa la pestaña Iteraciones para ver dónde se detectó.</p>`;
 const mx=MD.s=='max';
 let h=`<div class="mut">${mx?'Valor máximo':'Valor mínimo'} de Z</div><div class="big">${esc(valorZ(R.Z))}</div>
 <h2>Plan óptimo</h2><div class="tw"><table><tr><th>Variable y significado</th><th>Cantidad</th></tr>${R.x.map((x,i)=>`<tr><td>${nm(x[0])} · ${nombreX(i)}</td><td>${dec?x[2]:x[1]}</td></tr>`).join('')}</table></div>
 <h2>Uso de recursos y cumplimiento</h2><div class="tw"><table><tr><th>Restricción</th><th>Valor alcanzado</th><th>Límite</th><th>Holgura / exceso</th><th>Estado</th></tr>
 ${R.restr.map((r,i)=>`<tr><td>${nombreR(i)}</td><td>${V(r.act)}</td><td>${esc(r.op)} ${V(r.b)}</td><td>${V(r.holgura)} ${r.op=='<='?'de holgura':r.op=='>='?'de exceso':''}</td><td>${r.activa?'<span class="tag" style="color:var(--ac)">activa</span>':'<span class="tag">no activa</span>'}</td></tr>`).join('')}</table></div><p class="hint">Una restricción activa se cumple con igualdad. Su precio sombra indica el efecto marginal de cambiarla; estar activa no implica necesariamente un beneficio al ampliarla.</p>`;
 if(R.multiples)h+=`<p class="note">Hay <b>óptimos múltiples</b>: existe otro plan con el mismo valor de Z.${R.punto_alterno?' Ejemplo: '+R.punto_alterno.map(x=>nm(x[0])+' = '+(dec?x[2]:x[1])).join(', ')+'.':''}</p>`;
 if(R.degenerada)h+=`<p class="note">Solución <b>degenerada</b>: una variable básica vale 0. Los precios sombra pueden no ser únicos.</p>`;
 return h;
}
function ir(k){paso=Math.max(0,Math.min(R.tablas.length-1,k));pintar()}
function vIter(){
 const ts=R.tablas;if(!ts.length)return '<div class="empty">Sin iteraciones.</div>';
 const t=ts[paso],p=t.piv;
 const fase=R.fases==2?`<span class="tag">Fase ${t.fase}</span>`:'';
 let cap=p?(p.i==null?`Entra <b>${nm(p.entra)}</b>, pero ninguna fila tiene coeficiente positivo. ${esc(t.nota)}`
  :`Entra <b>${nm(p.entra)}</b> (coeficiente ${V(p.zc)} en Z) · sale <b>${nm(p.sale)}</b> (cociente mínimo ${V(p.min)}) · pivote ${V(p.pivote)}`):esc(t.nota);
 const iniFase=paso>0&&ts[paso-1].fase!=t.fase?' Se retiran las variables artificiales y comienza la Fase 2.':'';
 let h=`<div class="row"><button data-goto="0">Tabla inicial</button>${ts.length>1?'<button data-goto="1">Paso 1</button>':''}${ts.length>2?'<button data-goto="2">Paso 2</button>':''}<button data-goto="${ts.length-1}">${R.estado=='optimo'?'Tabla óptima':'Tabla final'}</button></div><p class="hint">${MD.s=='min'?'La minimización se calcula maximizando −Z. ':''}${R.fases==2?'La Fase 1 busca factibilidad maximizando −Σ artificiales; la Fase 2 optimiza el objetivo.':'Se parte de una base factible con variables de holgura.'}</p>
 <div class="stepbar"><button data-step="-1" ${paso==0?'disabled':''}>←</button><input type="range" id="sl" min="0" max="${ts.length-1}" value="${paso}" aria-label="Iteración"><button data-step="1" ${paso==ts.length-1?'disabled':''}>→</button></div>
 <p class="cap">${fase}<span class="tag">${paso==0?'Inicial':paso==ts.length-1?'Final':'Tabla '+paso}</span>${cap}${iniFase}</p>
 <div class="tw"><table class="tb"><tr><th>Base</th>${t.cols.map((c,j)=>`<th class="${p&&p.j==j?'pc':''}">${nm(c)}</th>`).join('')}<th>b</th>${p&&p.i!=null?'<th>Cociente</th>':''}</tr>`;
 t.t.forEach((r,i)=>{h+=`<tr class="${p&&p.i==i?'pr':''}"><th>${nm(t.base[i])}</th>${r.map((v,j)=>`<td class="${p&&p.j==j?'pc':''} ${p&&p.i==i&&p.j==j?'pv':''}">${V(v)}</td>`).join('')}${p&&p.i!=null?`<td>${p.q[i]?V(p.q[i]):'—'}</td>`:''}</tr>`});
 h+=`<tr class="zr"><th>Z</th>${t.z.map(v=>`<td>${V(v)}</td>`).join('')}${p&&p.i!=null?'<td></td>':''}</tr></table></div><p class="hint">Usa ← → del teclado para avanzar.</p>`;
 if(p&&p.i!=null){h+=`<h2>Cómo se obtiene la siguiente tabla</h2><ol class="steps-list"><li>Variable entrante: ${nm(p.entra)}. ${p.regla=='Bland'?'Regla de Bland: primera columna elegible, para evitar ciclos.':'Regla de Dantzig: coeficiente más negativo de la fila Z interna.'}</li><li>Variable saliente: ${nm(p.sale)}. Solo se consideran coeficientes positivos de la columna entrante: ${t.t.map((r,i)=>p.q[i]?`${nm(t.base[i])}: ${V(r[r.length-1])} / ${V(r[p.j])} = ${V(p.q[i])}`:`${nm(t.base[i])}: se descarta (coeficiente ≤ 0)`).join('; ')}.</li><li>Se divide la fila pivote entre ${V(p.pivote)} y se elimina esa columna en las demás filas.</li></ol>`;
 if(p.operaciones)h+=`<details open><summary>Operaciones de fila con valores exactos</summary><ol class="steps-list math">${p.operaciones.map(o=>`<li>${esc(o.formula)}${o.antes?`<details><summary>Ver fila antes y después</summary><div class="tw"><table class="compact-table"><tr><th></th>${t.cols.map(c=>`<th>${nm(c)}</th>`).join('')}<th>b</th></tr><tr><th>Antes</th>${o.antes.map(v=>`<td>${V(v)}</td>`).join('')}</tr><tr><th>Después</th>${o.despues.map(v=>`<td>${V(v)}</td>`).join('')}</tr></table></div></details>`:''}</li>`).join('')}</ol></details>`}
 return h;
}
function vSens(){
 if(R.estado!='optimo')return '<div class="empty">Disponible cuando hay solución óptima.</div>';
 let h='<p class="note">Estos intervalos permiten cambiar <b>un solo dato a la vez</b>, manteniendo los demás fijos y conservando esta base óptima. En los extremos puede haber degeneración u otras bases óptimas.</p>';
 if(R.duales)h+=`<h2>Precios sombra</h2><p class="mut" style="margin-top:0">Cambio de Z por cada unidad extra de lado derecho, válido dentro del rango.</p>
 <div class="tw"><table><tr><th>Restricción</th><th>RHS actual</th><th>Precio sombra</th><th>Rango de factibilidad</th></tr>${R.duales.map((d,i)=>`<tr><td>${nombreR(i)}</td><td>${esc(MD.r[i].b)}</td><td>${V(d.precio)}</td><td>${limite(d.lo,true)} a ${limite(d.hi)}</td></tr>`).join('')}</table></div><p class="hint">Dentro del intervalo: ΔZ = precio sombra × ΔRHS. El signo indica si Z aumenta o disminuye; en minimización, una reducción de Z es favorable. Un cambio de una unidad solo está cubierto si permanece dentro del rango.</p>`;
 else h+='<p class="note">'+esc(R.nota_sensibilidad||'Hay restricciones redundantes; no se dispone de rangos RHS ni precios sombra para esta base.')+'</p>';
 if(R.rangos_objetivo)h+=`<h2>Rangos de optimalidad de la función objetivo</h2><div class="tw"><table><tr><th>Coeficiente</th><th>Actual</th><th>Intervalo permitido</th><th>Disminución</th><th>Aumento</th></tr>${R.rangos_objetivo.map((d,i)=>`<tr><td>${nm(d.n)} · ${nombreX(i)}</td><td>${V(d.actual)}</td><td>${limite(d.lo,true)} a ${limite(d.hi)}</td><td>${limite(d.disminucion)}</td><td>${limite(d.aumento)}</td></tr>`).join('')}</table></div><p class="hint">En este intervalo, el plan básico sigue siendo óptimo. Z cambia según la cantidad de la variable cuyo coeficiente se modifica.</p>`;
 h+=`<h2>Costos reducidos</h2><p class="mut" style="margin-top:0">Cambio de Z al forzar una unidad de una variable no básica.</p>
 <div class="tw"><table><tr><th>Variable</th><th>Costo reducido</th><th>Situación</th></tr>${R.costos.map(c=>`<tr><td>${nm(c.n)}</td><td>${V(c.v)}</td><td>${c.basica?'Básica':'No básica (valor 0)'}</td></tr>`).join('')}</table></div><p class="hint">Para una variable no básica, el costo reducido describe el cambio marginal en Z al introducirla compensando las básicas, mientras la base sea factible. Si es cero, se debe verificar si hay un plan alternativo distinto.</p>`;
 h+=vistaEscenario();
 return h;
}
function vModelo(){const e=R.estandar;if(!e)return '<div class="empty">Vuelve a resolver para obtener la forma estándar.</div>';
 let h=`<h2 class="method-title">${esc(MD.meta?.titulo||'Modelo matemático')}</h2><p>${esc(MD.meta?.descripcion||'Define el contexto del caso en el panel del modelo.')}</p><p class="banner">${MD.c.length} variables de decisión · ${MD.r.length} restricciones${MD.c.length>=3?' · Modelo multivariable':''}</p>
 <h2>Variables de decisión</h2><ul>${MD.c.map((c,i)=>`<li>${nm('X'+(i+1))}: ${nombreX(i)}; no negativa y continua.</li>`).join('')}</ul>
 <h2>Formulación original</h2><div class="math">${MD.s=='max'?'Maximizar':'Minimizar'} Z = ${expresion(MD.c.map((_,i)=>'X'+(i+1)),MD.c)}<br>${MD.r.map((r,i)=>`${nombreR(i)}: ${expresion(MD.c.map((_,j)=>'X'+(j+1)),r.a)} ${esc(r.op)} ${esc(r.b)}`).join('<br>')}<br>${MD.c.map((_,i)=>nm('X'+(i+1))).join(', ')} ≥ 0</div>
 <h2>Forma estándar y preparación de la base</h2><p class="hint">Se convierten las desigualdades en igualdades. Si el lado derecho es negativo, se multiplica la fila por −1 y se invierte la desigualdad. Las artificiales son auxiliares de la Fase 1.</p><div class="math">${e.filas.map(f=>`R${f.original+1}: ${expresion(e.columnas,f.a)} = ${V(f.b)}${f.invertida?' <span class="tag">fila multiplicada por −1</span>':''}`).join('<br>')}</div>
 <div class="tw"><table><tr><th>Variable</th><th>Tipo</th></tr>${e.variables.map(v=>`<tr><td>${nm(v.nombre)}</td><td>${{decision:'Decisión',holgura:'Holgura (+)',exceso:'Exceso (−)',artificial:'Artificial (+), solo Fase 1'}[v.tipo]||esc(v.tipo)}</td></tr>`).join('')}</table></div><p>Todas las variables de esta formulación son ≥ 0.</p>
 <h2>Objetivo interno y base inicial</h2><div class="math">Maximizar ${MD.s=='min'?'−Z':'Z'} = ${expresion(e.columnas,e.objetivo_interno)}${e.objetivo_fase1?'<br>Fase 1: maximizar W = '+expresion(e.columnas,e.objetivo_fase1):''}</div><p>Base inicial: ${e.base_inicial.map(nm).join(', ')}.</p><p class="hint">La fila Z de la tabla incorpora los costos de las variables básicas para dejar sus coeficientes en cero.</p><button data-t="1">Ver tabla inicial e iteraciones</button>`;
 return h;
}
function vistaEscenario(){const anterior=ESCENARIO,seleccion=anterior?.campo||'b:0';return `<div class="scenario"><h2>¿Qué pasa si cambia un dato?</h2><p class="hint">Se resuelve de nuevo el modelo con un cambio. Los resultados principales conservan el modelo original.</p><label for="sc-campo">Dato a cambiar</label><div class="row"><select id="sc-campo">${MD.r.map((r,i)=>`<option value="b:${i}" ${seleccion=='b:'+i?'selected':''}>RHS ${nombreR(i)}</option>`).join('')}${MD.c.map((c,i)=>`<option value="c:${i}" ${seleccion=='c:'+i?'selected':''}>Coeficiente x${i+1} · ${nombreX(i)}</option>`).join('')}</select><input id="sc-valor" aria-label="Nuevo valor" value="${esc(anterior?.valor??MD.r[0].b)}" inputmode="decimal"><button data-simular="1" ${simulando?'disabled':''}>${simulando?'Calculando…':'Comparar escenario'}</button></div>
 <div id="sc-resultado">${anterior?resultadoEscenario(anterior):''}</div></div>`}
function resultadoEscenario(e){if(e.error)return `<p class="note">${esc(e.error)}</p>`;const rr=e.resultado;let h=`<p><b>Escenario: ${estadoTexto(rr.estado)}.</b> ${esc(rr.mensaje)}</p><p class="hint">Cambio: ${e.campo.startsWith('b:')?nombreR(+e.campo.split(':')[1]):nombreX(+e.campo.split(':')[1])}; valor anterior ${esc(e.anterior)} → ${esc(e.valor)}. ${e.dentro===null?'No hay intervalo disponible para este dato.':e.dentro?'El valor está dentro del intervalo de la base original.':'El valor está fuera del intervalo de la base original; el efecto se obtiene al resolver de nuevo.'}</p>`;
 if(rr.estado=='optimo'){const dz=nf(rr.Z[0])-nf(R.Z[0]);h+=`<div class="tw"><table><tr><th></th><th>Original</th><th>Escenario</th></tr><tr><th>Z</th><td>${esc(valorZ(R.Z))}</td><td>${esc(valorZ(rr.Z))}</td></tr>${R.x.map((x,i)=>`<tr><td>${nombreX(i)}</td><td>${dec?x[2]:x[1]}</td><td>${dec?rr.x[i][2]:rr.x[i][1]}</td></tr>`).join('')}</table></div><p>Variación de Z ≈ ${corto(dz)}. ${Math.abs(dz)<1e-9?'El valor objetivo se mantiene.':(MD.s=='max'?dz>0:dz<0)?'El objetivo mejora.':'El objetivo empeora.'} ${e.misma_base?'El solver conserva la misma base.':'El solver eligió una base distinta.'}</p>`;
 if(e.prediccion!==null)h+=`<p class="hint">Predicción por sensibilidad: Z ≈ ${corto(e.prediccion)}. La tabla muestra el resultado recalculado con fracciones exactas.</p>`}
 return h}
async function simular(){if(simulando)return;const campo=$('#sc-campo').value,valor=$('#sc-valor').value,[tipo,k]=campo.split(':'),i=+k,base=MD,original=R;
 if(!NUM.test(valor)||!Number.isFinite(nf(valor))){ESCENARIO={campo,valor,error:'Ingresa un número o fracción válida.'};pintar();return}
 const mod=JSON.parse(JSON.stringify(base)),anterior=tipo=='b'?mod.r[i].b:mod.c[i];if(tipo=='b')mod.r[i].b=valor;else mod.c[i]=valor;
 simulando=true;ESCENARIO={campo,valor,error:'Calculando el nuevo modelo…'};pintar();
 try{const response=await fetch('/api/resolver',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentido:mod.s,c:mod.c,restricciones:mod.r})}),rr=await response.json();if(MD!==base||R!==original)return;if(!response.ok)throw Error(rr.error||'No se pudo resolver.');
 const rango=tipo=='b'?original.duales?.[i]:original.rangos_objetivo?.[i],v=nf(valor),dentro=rango?(rango.lo===null||v>=nf(rango.lo[0]))&&(rango.hi===null||v<=nf(rango.hi[0])):null,delta=v-nf(anterior),pendiente=tipo=='b'?rango?.precio?.[0]:original.x[i][1];
 ESCENARIO={campo,valor,anterior,dentro,prediccion:dentro&&pendiente!==undefined?nf(original.Z[0])+nf(pendiente)*delta:null,misma_base:JSON.stringify([...(original.base_final||[])].sort())===JSON.stringify([...(rr.base_final||[])].sort()),modelo:mod,resultado:rr};
 }catch(e){if(MD!==base||R!==original)return;ESCENARIO={campo,valor,error:e.message}}
 if(MD===base&&R===original){simulando=false;pintar()}}
function vValidacion(){let h='<h2 class="method-title">Validación del resultado</h2><p>Compara el Simplex con fracciones exactas con <b>SciPy / HiGHS</b>, un optimizador independiente. También revisa factibilidad, objetivo, tablas y pivotes.</p><button data-validar="1" '+(validando?'disabled':'')+'>'+(validando?'Verificando…':'Validar con SciPy / HiGHS')+'</button>';
 if(!VALID)return h+'<p class="hint">La validación usa el modelo que produjo estos resultados.</p>';
 h+=`<div class="banner ${VALID.ok===false?'bad':''}" style="margin-top:16px">${esc(VALID.mensaje)}</div>`;
 if(VALID.certificado){const c=VALID.certificado,labels={factible:'La solución cumple todas las restricciones',objetivo_consistente:'Z coincide con la evaluación del plan',tabla_optima:'La tabla final satisface el criterio de optimalidad',tablas_consistentes:'Las tablas son consistentes',pivotes_validos:'Las operaciones de pivote son correctas'};h+=`<div class="check-list">${Object.entries(labels).map(([k,label])=>`<div>${c[k]===true?'✓':c[k]===false?'✕':'—'} ${label}${c[k]===null?' (no aplica)':''}</div>`).join('')}</div><p class="hint">${c.tablas_revisadas??0} tablas y ${c.pivotes_revisados??0} pivotes revisados.</p>`;if(c.errores?.length)h+='<ul>'+c.errores.map(e=>'<li>'+esc(e)+'</li>').join('')+'</ul>'}
 const c=VALID.comparacion;if(c){h+=`<h2>Comparación independiente</h2><div class="tw"><table><tr><th></th><th>Simplex exacto</th><th>SciPy / HiGHS</th></tr><tr><th>Estado</th><td>${estadoTexto(c.estado_simplex)}</td><td>${estadoTexto(c.estado_scipy)}</td></tr><tr><th>Z</th><td>${c.z_simplex?V(c.z_simplex):'—'}</td><td>${c.z_scipy==null?'—':corto(c.z_scipy)}</td></tr></table></div><p>${esc(c.mensaje||'')}</p>`;
 if(Array.isArray(c.x_scipy))h+=`<p class="math">Plan de SciPy: ${c.x_scipy.map((v,i)=>nm('X'+(i+1))+' = '+corto(v)).join('; ')}.</p>`;
 h+='<p class="hint">Se compara el estado y el objetivo con tolerancia numérica absoluta y relativa de 10⁻⁷. Si hay óptimos múltiples, ambos métodos pueden devolver planes distintos con la misma Z.</p>'}
 return h}
async function validar(){if(validando)return;const base=MD,original=R;validando=true;pintar();try{const response=await fetch('/api/validar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentido:base.s,c:base.c,restricciones:base.r})}),j=await response.json();if(MD!==base||R!==original)return;if(!response.ok)throw Error(j.error||'No se pudo validar.');VALID=j}catch(e){if(MD!==base||R!==original)return;VALID={ok:null,mensaje:e.message}}if(MD===base&&R===original){validando=false;pintar()}}
function vDecisiones(){let h='<h2 class="method-title">Interpretación y decisiones</h2>';
 if(R.estado!='optimo')return h+`<div class="banner bad">${esc(R.mensaje)}</div><p>${R.estado=='infactible'?'Revisa si los mínimos exigidos superan la capacidad, si hay igualdades contradictorias o si las unidades son coherentes. No existe un plan que cumpla el modelo actual.':'Revisa si faltan límites de capacidad, demanda o presupuesto. El modelo permite mejorar el objetivo indefinidamente y no proporciona un plan óptimo finito.'}</p><button data-t="5">Validar el diagnóstico</button>`;
 h+=`<div class="banner">El mejor valor de ${MD.s=='max'?'ganancia o beneficio':'costo'} según el modelo es ${esc(valorZ(R.Z))}.</div><div class="interpretation"><h2>Plan recomendado</h2><ul>${R.x.map((x,i)=>`<li><b>${nombreX(i)}</b>: ${dec?x[2]:x[1]}.${nf(x[1])===0?' No se asigna cantidad a esta actividad en el plan obtenido.':''}</li>`).join('')}</ul><h2>Recursos y requerimientos</h2><ul>${R.restr.map((r,i)=>`<li><b>${nombreR(i)}</b>: ${r.op=='<='?`se usan ${V(r.act)} de ${V(r.b)}; sobran ${V(r.holgura)}.`:r.op=='>='?`se alcanza ${V(r.act)} frente al mínimo ${V(r.b)}; el exceso es ${V(r.holgura)}.`:`se cumple exactamente ${V(r.b)}.`} ${r.activa?'La restricción está activa.':'La restricción no está activa.'}</li>`).join('')}</ul>`;
 if(R.duales){h+='<h2>Qué recursos conviene revisar</h2><ul>'+R.duales.map((d,i)=>{const p=nf(d.precio[0]);let mensaje=p===0?'Un cambio pequeño dentro del intervalo no cambia Z.':`Por cada unidad de aumento del RHS, Z ${p>0?'aumenta':'disminuye'} en ${corto(Math.abs(p))}, mientras el nuevo RHS esté entre ${limite(d.lo,true)} y ${limite(d.hi)}.`;
 if(MD.s=='max'&&MD.r[i].op=='<='&&p>0)mensaje+=` Su valor marginal es ${V(d.precio)} por unidad: compáralo con el costo de ampliar el recurso. Solo conviene si el beneficio adicional supera ese costo y el cambio permanece en el rango.`;
 return `<li><b>${nombreR(i)}</b>: ${mensaje}</li>`}).join('')+'</ul><p class="hint">Los precios sombra son marginales y dependen de esta base. Una restricción activa con precio cero no justifica por sí sola comprar más recurso. Usa el simulador para evaluar un cambio concreto.</p>'}
 else h+='<p class="note">No se generan recomendaciones marginales de recursos porque esta base tiene restricciones redundantes.</p>';
 if(R.multiples)h+='<p class="note">Existen planes alternativos con la misma Z. Puedes elegir entre ellos usando criterios que no estén en el modelo, como facilidad operativa o riesgo.</p>';
 h+='<h2>Supuestos y límites del modelo</h2><ul><li><b>Proporcionalidad:</b> costos, ganancias y consumos por unidad son constantes; no se incluyen descuentos ni economías de escala.</li><li><b>Aditividad:</b> los aportes de las actividades se suman; no se modelan interacciones entre productos.</li><li><b>Certidumbre:</b> capacidades y coeficientes se toman como conocidos. Si cambian, compara escenarios.</li><li><b>Divisibilidad:</b> las variables son continuas. Si representan unidades indivisibles, una solución fraccionaria requiere programación entera; redondearla puede violar restricciones.</li><li><b>Alcance:</b> la recomendación solo considera las restricciones ingresadas. Demanda, calidad, riesgo u otros límites deben incorporarse cuando correspondan.</li></ul><button data-t="3">Analizar un cambio en Sensibilidad</button></div>';
 return h}
/* ---------- gráfica (2 variables) ---------- */
const nf=t=>{t=String(t).replace(',','.').trim();if(t.includes('/')){const[a,b]=t.split('/');return a/b}return t===''?0:+t};
const corto=n=>Number(n.toPrecision(6)).toLocaleString('es-GT',{maximumSignificantDigits:6});
const dinero=n=>'$'+Number(n).toLocaleString('en-US',{maximumFractionDigits:2});
const valorZ=p=>MD.meta?.moneda&&(!p[0].includes('/')||dec)?dinero(nf(p[0])):V(p);
const formula=a=>a.map((v,i)=>{const n=nf(v);if(!n)return '';return `${n<0?' − ':i&&a.slice(0,i).some(v=>nf(v))?' + ':''}${Math.abs(n)==1?'':esc(String(v).trim().replace(/^-/,'')||'0')+'·'}x${i?'₂':'₁'}`}).join('').trim()||'0';
const nombreVertice=i=>i<26?String.fromCharCode(65+i):'V'+(i+1);
let G=null;
function unicos(ps,L=1){const eps=1e-9*(Math.abs(L)||1);return ps.filter((p,i)=>!ps.slice(0,i).some(q=>Math.hypot(p[0]-q[0],p[1]-q[1])<eps))}
function clip(poly,a,b,s){const scale=Math.max(Math.abs(a[0]),Math.abs(a[1]),Math.abs(b))||1,f=p=>s*(b-a[0]*p[0]-a[1]*p[1])/scale,o=[];
 poly.forEach((p,k)=>{const q=poly[(k+1)%poly.length],fp=f(p),fq=f(q);
  const ip=fp>=-1e-9,iq=fq>=-1e-9;if(ip)o.push(p);
  if(ip!==iq){const t=Math.max(0,Math.min(1,fp/(fp-fq)));o.push([p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])])}});return unicos(o)}
function segmento(a,b,L){const c=[],eps=L*1e-9;
 if(a[1]){[0,L].forEach(x=>{const y=(b-a[0]*x)/a[1];if(y>=-eps&&y<=L+eps)c.push([x,Math.max(0,Math.min(L,y))])})}
 if(a[0]){[0,L].forEach(y=>{const x=(b-a[1]*y)/a[0];if(x>=-eps&&x<=L+eps)c.push([Math.max(0,Math.min(L,x)),y])})}
 const ps=unicos(c,L);return ps.length>=2?[ps[0],ps[1]]:null}
function escala(v){const paso=Math.pow(10,Math.floor(Math.log10(v))),q=v/paso;return (q<=1?1:q<=2?2:q<=2.5?2.5:q<=5?5:q<=7.5?7.5:10)*paso}
function vGraf(){
 if(MD.c.length!=2)return '<div class="empty">La gráfica está disponible con exactamente 2 variables.</div>';
 if(!R.grafica)return '<div class="banner bad">Reinicia el archivo de Python y vuelve a resolver para cargar el método gráfico.</div>';
 const rs=MD.r.map(r=>({a:r.a.map(nf),op:r.op,b:nf(r.b)})),c=MD.c.map(nf);
 const opt=R.estado=='optimo'?R.x.map(x=>nf(x[1])):null,geo=R.grafica,vs=geo.vertices;
 let extent=0;if(!vs.length||geo.region_no_acotada)rs.forEach(r=>r.a.forEach(a=>{if(a&&r.b/a>0)extent=Math.max(extent,r.b/a)}));
 vs.forEach(v=>{extent=Math.max(extent,nf(v.x[0]),nf(v.y[0]))});if(opt)extent=Math.max(extent,...opt);
 const L=escala((extent||4)*1.08),colors=['#2563eb','#15803d','#dc2626','#d97706','#0891b2','#7c3aed','#be185d'];
 let poly=[[0,0],[1,0],[1,1],[0,1]];
 rs.forEach(r=>{const a=r.a.map(v=>v*L);if(r.op!='>=')poly=clip(poly,a,r.b,1);if(r.op!='<=')poly=clip(poly,a,r.b,-1)});
 poly=poly.map(p=>p.map(v=>v*L));
 if(R.estado=='infactible')poly=[];
 G={L,c,rs,poly,opt,Z:opt?nf(R.Z[0]):0};
 const W=700,H=580+rs.length*23,left=72,right=670,top=68,bottom=438;
 const X=x=>left+x/L*(right-left),Y=y=>bottom-y/L*(bottom-top);G.X=X;G.Y=Y;
 const titulo=MD.meta?.titulo||'Región factible y función objetivo';
 let s=`<svg id="grafica" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="graf-titulo graf-desc" style="font-family:Segoe UI,Arial,sans-serif;font-size:13px;background:#fff;color:#334155">
 <title id="graf-titulo">${esc(titulo)}</title><desc id="graf-desc">Restricciones, región factible y vértices. ${opt?'Óptimo ('+R.x.map(x=>x[1]).join(', ')+'), Z = '+R.Z[0]:esc(R.mensaje)}</desc>
 <rect width="${W}" height="${H}" fill="#ffffff"/>
 <text x="${left}" y="27" fill="#0f172a" font-size="18" font-weight="600">${esc(titulo)}</text>
 <text x="${left}" y="49" fill="#475569">${MD.s=='max'?'Maximizar':'Minimizar'} Z = ${formula(MD.c)}</text>`;
 for(let k=0;k<=5;k++){const v=L*k/5,t=+v.toPrecision(3);
  s+=`<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(L)}" stroke="#e2e8f0"/><line x1="${X(0)}" y1="${Y(v)}" x2="${X(L)}" y2="${Y(v)}" stroke="#e2e8f0"/>
  <text x="${X(v)}" y="${bottom+22}" text-anchor="middle" fill="#475569">${corto(t)}</text><text x="${left-10}" y="${Y(v)+4}" text-anchor="end" fill="#475569">${corto(t)}</text>`}
 s+=`<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="none" stroke="#94a3b8"/>
 <text x="${(left+right)/2}" y="${bottom+51}" text-anchor="middle" fill="#0f172a">${esc(MD.meta?.variables[0]||'Variable x₁')} (x₁ ≥ 0)</text>
 <text transform="translate(18 ${(top+bottom)/2}) rotate(-90)" text-anchor="middle" fill="#0f172a">${esc(MD.meta?.variables[1]||'Variable x₂')} (x₂ ≥ 0)</text>`;
 if(poly.length>2&&geo.dimension==2)s+=`<polygon points="${poly.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill="#0b7285" fill-opacity=".16" stroke="#0b7285" stroke-width="1.5"/>`;
 rs.forEach((r,i)=>{const sg=segmento(r.a,r.b,L);if(!sg)return;
  const label=sg[0][0]>sg[1][0]?sg[0]:sg[1];
  s+=`<line x1="${X(sg[0][0])}" y1="${Y(sg[0][1])}" x2="${X(sg[1][0])}" y2="${Y(sg[1][1])}" stroke="${colors[i%colors.length]}" stroke-width="2" ${i>=colors.length?'stroke-dasharray="8 4"':''}><title>R${i+1}: ${formula(MD.r[i].a)} ${esc(r.op)} ${esc(MD.r[i].b)}</title></line><text x="${Math.max(left+10,Math.min(right-25,X(label[0])-8))}" y="${Math.max(top+14,Y(label[1])-7)}" fill="${colors[i%colors.length]}" font-weight="600">R${i+1}</text>`});
 if(geo.dimension==1&&poly.length>=2){const ends=poly.reduce((best,p)=>{for(const q of poly)if(Math.hypot(p[0]-q[0],p[1]-q[1])>best.d)best={d:Math.hypot(p[0]-q[0],p[1]-q[1]),p,q};return best},{d:-1});s+=`<line x1="${X(ends.p[0])}" y1="${Y(ends.p[1])}" x2="${X(ends.q[0])}" y2="${Y(ends.q[1])}" stroke="#0b7285" stroke-width="6" opacity=".65"/>`}
 s+=`<g id="isog"></g>`;
 vs.forEach((v,i)=>{const x=nf(v.x[0]),y=nf(v.y[0]);
  s+=`<circle cx="${X(x)}" cy="${Y(y)}" r="${v.optimo?6:4}" fill="${v.optimo?'#a21caf':'#fff'}" stroke="${v.optimo?'#a21caf':'#0b7285'}" stroke-width="2"><title>${nombreVertice(i)}: (${v.x[0]}, ${v.y[0]}) · Z = ${v.z[0]}${v.optimo?' · Óptimo':''}</title></circle><text x="${X(x)+8}" y="${Y(y)-8}" fill="#334155" font-weight="600">${nombreVertice(i)}</text>`});
 if(opt){s+=`<circle cx="${X(opt[0])}" cy="${Y(opt[1])}" r="7" fill="#a21caf" stroke="#fff" stroke-width="2"/>
 <text x="${Math.min(right-195,X(opt[0])+14)}" y="${Math.max(top+20,Y(opt[1])-26)}" fill="#86198f" font-weight="700" style="paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round">Óptimo (${esc(V([R.x[0][1],R.x[0][2]]))}, ${esc(V([R.x[1][1],R.x[1][2]]))})</text>`}
 rs.forEach((r,i)=>{const y=529+i*23;s+=`<line x1="${left}" y1="${y-4}" x2="${left+25}" y2="${y-4}" stroke="${colors[i%colors.length]}" stroke-width="3"/>
 <text x="${left+35}" y="${y}" fill="#334155">R${i+1}${MD.meta?' · '+esc(MD.meta.restricciones[i]):''}: ${formula(MD.r[i].a)} ${r.op=='<='?'≤':r.op=='>='?'≥':'='} ${esc(MD.r[i].b)}</text>`});
 const ly=529+rs.length*23;
 s+=`<rect x="${left}" y="${ly-11}" width="24" height="12" fill="#0b7285" fill-opacity=".22"/><text x="${left+35}" y="${ly}" fill="#334155">Región factible${geo.dimension==1?' (segmento o semirrecta)':geo.dimension==0?' (punto)':''}</text>`;
 if(opt)s+=`<circle cx="${left+12}" cy="${ly+19}" r="5" fill="#a21caf"/><text x="${left+35}" y="${ly+23}" fill="#86198f">Óptimo: Z = ${esc(valorZ(R.Z))}</text>`;
 s+='</svg>';
 const zEsquinas=[0,c[0]*L,c[1]*L,(c[0]+c[1])*L],zmin=Math.min(...zEsquinas,G.Z),zmax=Math.max(...zEsquinas,G.Z),amplitud=zmax-zmin||1;
 setTimeout(()=>dibujaIso(G.Z),0);
 let h=`<h2 class="method-title">Método gráfico</h2><p class="mut">1. Traza las restricciones. 2. Encuentra la región común. 3. Compara Z en sus vértices.</p>`;
 if(opt)h+=`<div class="banner">${MD.s=='max'?'Máximo':'Mínimo'}: Z = ${esc(valorZ(R.Z))} · x₁ = ${esc(V([R.x[0][1],R.x[0][2]]))}, x₂ = ${esc(V([R.x[1][1],R.x[1][2]]))}</div>`;
 else h+=`<div class="banner bad">${esc(R.mensaje)}</div>`;
 h+=`<div class="graph-wrap">${s}</div><div class="graph-actions"><span class="hint">x₁, x₂ ≥ 0 · Pasa el cursor por los vértices.</span><button data-svg="1">Descargar gráfica SVG</button></div>`;
 if(c.some(v=>v!==0))h+=`<div class="stepbar"><label for="iso">Línea de Z</label><input type="range" id="iso" aria-label="Valor de la función objetivo" min="${zmin}" max="${zmax==zmin?zmin+1:zmax}" step="${amplitud/400}" value="${G.Z}"><b id="isov"></b>${opt?'<button data-opt="1">Volver al óptimo</button>':''}</div><p class="hint">La línea discontinua une puntos con la misma Z. Muévela para ver cómo cambia la ganancia o el costo.</p>`;
 if(geo.region_no_acotada)h+='<p class="note">La región factible se extiende fuera de la ventana. Los bordes del dibujo no son restricciones. '+(opt?'Aunque la región no está acotada, la función objetivo sí tiene un óptimo.':'Comparar solo los vértices no demuestra un óptimo: también hay que revisar las direcciones en que la región se extiende.')+'</p>';
 if(geo.dimension==1)h+='<p class="hint">La región factible está sobre una recta; no tiene área.</p>';
 if(geo.dimension==0)h+='<p class="hint">La región factible contiene un único punto.</p>';
 h+='<h2>Evaluación de los vértices factibles</h2>';
 if(vs.length){h+=`<div class="tw"><table><thead><tr><th>Punto</th><th>x₁</th><th>x₂</th><th>Z</th><th>Evaluación</th></tr></thead><tbody>${vs.map((v,i)=>`<tr class="${v.optimo?'opt-row':''}"><td>${nombreVertice(i)}</td><td>${V(v.x)}</td><td>${V(v.y)}</td><td>${esc(valorZ(v.z))}</td><td>${v.optimo?'Óptimo':'Factible'}</td></tr>`).join('')}</tbody></table></div>`;
 if(opt)h+=`<p class="hint">Se elige el ${MD.s=='max'?'mayor':'menor'} valor de Z. Las coordenadas se calculan con fracciones exactas; la gráfica usa una aproximación visual.</p>`;
 if(vs.filter(v=>v.optimo).length>1)h+='<p class="note">Hay varios vértices óptimos. Los segmentos que los unen también son óptimos.</p>';
 }else h+='<p class="mut">No hay vértices factibles: las restricciones no tienen una solución común.</p>';
 return h;
}
function dibujaIso(z){const g=$('#isog');if(!g||!G)return;const{c,L,X,Y}=G;const sg=segmento(c,z,L);
 g.innerHTML=sg?`<line x1="${X(sg[0][0])}" y1="${Y(sg[0][1])}" x2="${X(sg[1][0])}" y2="${Y(sg[1][1])}" stroke="#a21caf" stroke-width="2" stroke-dasharray="7 5"/>`:'';
 const v=$('#isov');if(v)v.textContent='Z = '+(MD.meta?.moneda?dinero(z):corto(z))}
function descargarGrafica(){const svg=$('#grafica');if(!svg)return;const blob=new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='metodo-grafico.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
cargar(EJ["Taller · 3 productos (Fase 2)"]);resolver(0);
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
