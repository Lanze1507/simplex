"""
Prototipo 1.0: Calculadora del Metodo Simplex tabular.
Curso: Investigacion de Operaciones.

Alcance: max Z = c*x, sujeto a A*x <= b, b >= 0, x >= 0.
No implementa minimizacion, restricciones >= o =, variables libres,
Gran M, Dos Fases, variables artificiales ni interfaz grafica.

Uso:
    python simplex.py                         # Dos ejemplos y su validacion
    python simplex.py --interactivo            # Ingresar un modelo
    python simplex.py --reporte resultado.txt  # Guardar evidencia en texto
    python simplex.py --info                   # Alcance y entorno real

Dependencias: python -m pip install numpy pandas
Informe complementario: simplex_proyecto.md.
Los ejemplos incluidos requieren confirmar su correspondencia con la clase.
"""

import argparse
import io
import math
import platform
import sys
from contextlib import redirect_stdout
from fractions import Fraction
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(
        "Falta una dependencia: "
        f"{error.name}.\nInstalala con el mismo Python que ejecuta el programa:\n"
        f'"{sys.executable}" -m pip install numpy pandas'
    ) from error

TOLERANCIA = 1e-9


class ProblemaNoAcotado(ValueError):
    """Una variable puede aumentar Z sin que una restriccion la acote."""


class LimiteSimplex(RuntimeError):
    """El algoritmo se detuvo sin certificar un optimo."""


def numero(valor):
    """Formatea solo la salida; los calculos conservan su precision."""
    return f"{float(valor):.8g}"


def expresion(coeficientes, nombres):
    """Construye una expresion matematica legible sin omitir signos."""
    partes = []
    for coeficiente, nombre in zip(coeficientes, nombres):
        if coeficiente == 0:
            continue
        termino = f"{numero(abs(coeficiente))}*{nombre}"
        if not partes:
            partes.append(("-" if coeficiente < 0 else "") + termino)
        else:
            partes.append((" - " if coeficiente < 0 else " + ") + termino)
    return "".join(partes) or "0"


class SimplexTabular:
    """
    Traduce el pivoteo a operaciones sobre una matriz NumPy de tipo float.

    tabla: matriz bidimensional con filas [A | I | b] y [-c | 0 | 0].
    base: lista con los nombres de las variables basicas, una por restriccion.
    nombres_columnas: X1..Xn, S1..Sm, b.
    Pandas se utiliza exclusivamente para presentar las tablas.

    Entrada: coeficiente mas negativo de la fila Z.
    Salida: minimo cociente no negativo con denominador positivo.
    Empates exactos: primera columna/fila en el orden de la tabla.
    Se detecta repeticion de base y se limita el numero de iteraciones.
    """

    def __init__(self, c, A, b, max_iteraciones=1000):
        try:
            self.c = np.array(c, dtype=float)
            self.A = np.array(A, dtype=float)
            self.b = np.array(b, dtype=float)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(
                "c, A y b deben contener numeros y filas del mismo largo."
            ) from error
        self._validar_datos()
        if (isinstance(max_iteraciones, bool)
                or not isinstance(max_iteraciones, (int, np.integer))
                or max_iteraciones < 1):
            raise ValueError("max_iteraciones debe ser un entero positivo.")
        self.max_iteraciones = int(max_iteraciones)
        self.num_vars = len(self.c)
        self.num_restricciones = len(self.b)
        self.iteracion = 0
        self._construir_tabla_inicial()

    def _validar_datos(self):
        """Rechaza entradas incompatibles con la base inicial de holguras."""
        if self.c.ndim != 1 or self.c.size == 0:
            raise ValueError("c debe ser un vector con al menos una variable.")
        if self.b.ndim != 1 or self.b.size == 0:
            raise ValueError("b debe ser un vector con al menos una restriccion.")
        if self.A.ndim != 2 or self.A.shape != (self.b.size, self.c.size):
            raise ValueError(
                "A debe tener una fila por restriccion y una columna "
                "por variable; sus dimensiones deben coincidir con b y c."
            )
        if not all(np.isfinite(datos).all()
                   for datos in (self.c, self.A, self.b)):
            raise ValueError("Todos los coeficientes deben ser finitos (sin NaN/inf).")
        if np.any(self.b < 0):
            raise ValueError(
                "Este prototipo requiere b >= 0. Un b negativo puede necesitar "
                "otra base inicial; no se resuelve con esta version."
            )

    def _construir_tabla_inicial(self):
        """Agrega S >= 0: A*x + I*S = b, Z - c*x = 0."""
        m, n = self.num_restricciones, self.num_vars
        self.tabla = np.zeros((m + 1, n + m + 1), dtype=float)
        self.tabla[:m, :n] = self.A
        self.tabla[:m, n:n + m] = np.eye(m)
        self.tabla[:m, -1] = self.b
        self.tabla[-1, :n] = -self.c
        self.nombres_columnas = (
            [f"X{i + 1}" for i in range(n)]
            + [f"S{i + 1}" for i in range(m)] + ["b"]
        )
        self.base = [f"S{i + 1}" for i in range(m)]

    def mostrar_modelo(self):
        """Expone el modelo original y su conversion a forma estandar."""
        variables = self.nombres_columnas[:self.num_vars]
        objetivo = expresion(self.c, variables)
        print("\nMODELO MATEMATICO")
        print(f"Maximizar Z = {objetivo}")
        for i, (fila, lado) in enumerate(zip(self.A, self.b), start=1):
            print(f"  R{i}: {expresion(fila, variables)} <= {numero(lado)}")
        print("  " + ", ".join(variables) + " >= 0")
        print("\nFORMA ESTANDAR")
        for i, (fila, lado) in enumerate(zip(self.A, self.b), start=1):
            print(f"  R{i}: {expresion(fila, variables)} + S{i} = {numero(lado)}")
        print(f"  Z - ({objetivo}) = 0")
        print("  Todas las variables de decision y de holgura son >= 0.")

    def es_optima(self):
        """Optima si la fila Z no tiene coeficientes < -TOLERANCIA."""
        return bool(np.all(self.tabla[-1, :-1] >= -TOLERANCIA))

    def encontrar_columna_pivote(self):
        """Selecciona el coeficiente mas negativo; en empate, el primero."""
        return int(np.argmin(self.tabla[-1, :-1]))

    def calcular_cocientes(self, col_pivote):
        """None marca una fila no elegible y evita dividir por cero."""
        return [
            float(self.tabla[i, -1] / self.tabla[i, col_pivote])
            if self.tabla[i, col_pivote] > TOLERANCIA else None
            for i in range(self.num_restricciones)
        ]

    def encontrar_fila_pivote(self, col_pivote):
        """Selecciona la primera fila con el menor cociente; acepta cero."""
        cocientes = self.calcular_cocientes(col_pivote)
        elegibles = [i for i, valor in enumerate(cocientes) if valor is not None]
        if not elegibles:
            raise ProblemaNoAcotado(
                "Problema no acotado: no hay coeficientes positivos "
                "en la columna pivote dentro de la tolerancia utilizada."
            )
        return min(elegibles, key=lambda i: cocientes[i])

    def actualizar_tabla(self, fila_pivote, col_pivote):
        """Normaliza el pivote y elimina su columna mediante Gauss-Jordan."""
        pivote = self.tabla[fila_pivote, col_pivote]
        if abs(pivote) <= TOLERANCIA:
            raise ArithmeticError("Pivote nulo o demasiado pequeno para dividir.")
        # La copia evita dejar una tabla parcial si falla una operacion.
        nueva = self.tabla.copy()
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            nueva[fila_pivote, :] /= pivote
            for i in range(nueva.shape[0]):
                if i != fila_pivote:
                    factor = nueva[i, col_pivote]
                    nueva[i, :] -= factor * nueva[fila_pivote, :]
        if not np.isfinite(nueva).all():
            raise ArithmeticError("Error numerico: la tabla contiene NaN o infinito.")
        self.tabla = nueva
        self.base[fila_pivote] = self.nombres_columnas[col_pivote]

    def obtener_dataframe(self):
        """Copia la tabla a un DataFrame para visualizarla en consola."""
        df = pd.DataFrame(self.tabla.copy(), columns=self.nombres_columnas)
        df.insert(0, "Base", self.base + ["Z"])
        return df

    def mostrar_tabla(self):
        print(self.obtener_dataframe().to_string(index=False, float_format=numero))

    def _mostrar_pivoteo(self, fila, columna):
        """Imprime razones, pivote y operaciones que se ejecutaran."""
        print(f"\n--- Iteracion {self.iteracion + 1} ---")
        print(f"Entra {self.nombres_columnas[columna]}: coeficiente "
              f"{numero(self.tabla[-1, columna])} en la fila Z.")
        print("Prueba del cociente minimo (solo denominadores positivos):")
        for i, cociente in enumerate(self.calcular_cocientes(columna)):
            prefijo = f"  F{i + 1} ({self.base[i]}): "
            if cociente is None:
                print(prefijo + "no elegible; coeficiente <= tolerancia.")
            else:
                print(prefijo + f"{numero(self.tabla[i, -1])} / "
                      f"{numero(self.tabla[i, columna])} = {numero(cociente)}")
        print(f"Sale {self.base[fila]}; fila pivote F{fila + 1}.")
        print(f"Elemento pivote: {numero(self.tabla[fila, columna])}")
        print("Operaciones de Gauss-Jordan:")
        print(f"  F{fila + 1} <- F{fila + 1} / "
              f"({numero(self.tabla[fila, columna])})")
        for i in range(self.tabla.shape[0]):
            if i != fila:
                etiqueta = "Z" if i == self.num_restricciones else f"F{i + 1}"
                print(f"  {etiqueta} <- {etiqueta} - "
                      f"({numero(self.tabla[i, columna])}) * F{fila + 1} normalizada")

    def resolver(self, mostrar_iteraciones=True):
        """Ejecuta el ciclo y solo devuelve una solucion tras ser optima."""
        bases_vistas = set()
        if mostrar_iteraciones:
            print(f"\nTABLA INICIAL (iteracion {self.iteracion})")
            self.mostrar_tabla()
        while not self.es_optima():
            if self.iteracion >= self.max_iteraciones:
                raise LimiteSimplex(
                    f"Limite de {self.max_iteraciones} iteraciones alcanzado. "
                    "No se ha certificado una solucion optima."
                )
            base_actual = tuple(self.base)
            if base_actual in bases_vistas:
                raise LimiteSimplex(
                    "Se detecto una base repetida (posible ciclo por degeneracion). "
                    "No se ha certificado una solucion optima. "
                    "Una regla anticiclado completa queda para una fase futura."
                )
            bases_vistas.add(base_actual)
            columna = self.encontrar_columna_pivote()
            fila = self.encontrar_fila_pivote(columna)
            if mostrar_iteraciones:
                self._mostrar_pivoteo(fila, columna)
            self.actualizar_tabla(fila, columna)
            self.iteracion += 1
            if mostrar_iteraciones:
                self.mostrar_tabla()
        solucion = self.obtener_solucion()
        if not all(self.verificar_solucion(solucion).values()):
            raise ArithmeticError(
                "La tabla termino, pero fallo la comprobacion numerica "
                "de factibilidad u objetivo. Revise la escala de los datos."
            )
        if mostrar_iteraciones:
            print("\nTABLA FINAL")
            self.mostrar_tabla()
            print(f"Criterio de parada: fila Z sin valores < -{TOLERANCIA:g}.")
        return solucion

    def obtener_solucion(self):
        """Extrae Z y X; las variables no basicas valen cero."""
        solucion = {f"X{i + 1}": 0.0 for i in range(self.num_vars)}
        for i, variable in enumerate(self.base):
            if variable in solucion:
                solucion[variable] = float(self.tabla[i, -1])
        return {"Z": float(self.tabla[-1, -1]), **solucion}

    def verificar_solucion(self, solucion):
        """Comprueba x >= 0, A*x <= b y Z = c*x sobre el modelo original."""
        x = np.array([solucion[f"X{i + 1}"] for i in range(self.num_vars)])
        actividad = self.A @ x
        escala = np.maximum(1.0, np.maximum(np.abs(self.b), np.abs(actividad)))
        return {
            "Variables no negativas": bool(np.all(x >= -1e-7)),
            "Restricciones satisfechas": bool(np.all(actividad <= self.b + 1e-7 * escala)),
            "Z coincide con c*x": math.isclose(
                solucion["Z"], float(self.c @ x), rel_tol=1e-7, abs_tol=1e-7
            ),
        }


# Confirmar que estos ejemplos sean los ejercicios resueltos en clase.
# Los valores esperados son independientes de la salida del algoritmo.
CASOS_PRUEBA = [
    {
        "nombre": "CASO 1: Maximizar Z = 3X1 + 5X2",
        "c": [3, 5], "A": [[1, 0], [0, 2], [3, 2]], "b": [4, 12, 18],
        "esperado": {"Z": 36.0, "X1": 2.0, "X2": 6.0},
        "certificado_dual": [0, 1.5, 1],
    },
    {
        "nombre": "CASO 2: Maximizar Z = 5X1 + 4X2",
        "c": [5, 4], "A": [[6, 4], [1, 2]], "b": [24, 6],
        "esperado": {"Z": 21.0, "X1": 3.0, "X2": 1.5},
        "certificado_dual": [0.75, 0.5],
    },
]


def resolver_caso(nombre, c, A, b, esperado=None, certificado_dual=None,
                  max_iteraciones=1000):
    """Presenta modelo, tablas, interpretacion y contraste con la teoria."""
    print("\n" + "=" * 76)
    print(nombre)
    print("=" * 76)
    modelo = SimplexTabular(c, A, b, max_iteraciones=max_iteraciones)
    modelo.mostrar_modelo()
    solucion = modelo.resolver()
    print("\nSOLUCION OPTIMA")
    for variable, valor in solucion.items():
        print(f"  {variable} = {numero(valor)}")
    print(f"Iteraciones realizadas: {modelo.iteracion}")
    x = np.array([solucion[f"X{i + 1}"] for i in range(modelo.num_vars)])
    print("\nINTERPRETACION")
    print(f"El valor maximo de la funcion objetivo es {numero(solucion['Z'])}.")
    for i, holgura in enumerate(modelo.b - modelo.A @ x, start=1):
        estado = "activa" if abs(holgura) <= 1e-7 else "con holgura"
        print(f"  R{i}: holgura S{i} = {numero(holgura)} ({estado}).")
    comprobaciones = modelo.verificar_solucion(solucion)
    print("\nVALIDACION")
    for descripcion, cumple in comprobaciones.items():
        print(f"  {descripcion}: {'OK' if cumple else 'FALLO'}")
    if esperado is not None:
        print("  Comparacion con los valores teoricos de referencia:")
        for variable, referencia in esperado.items():
            cumple = math.isclose(
                solucion[variable], referencia, rel_tol=1e-7, abs_tol=1e-7
            )
            comprobaciones[f"Teoria {variable}"] = cumple
            print(f"    {variable}: calculado={numero(solucion[variable])}, "
                  f"esperado={numero(referencia)} -> {'OK' if cumple else 'FALLO'}")
    if certificado_dual is not None:
        # y >= 0 y A^T*y >= c implican c*x <= b^T*y para todo x factible.
        # Alcanzar esa cota superior verifica la optimalidad de los ejemplos.
        y = np.array(certificado_dual, dtype=float)
        certificado_valido = (
            y.shape == modelo.b.shape and np.isfinite(y).all()
            and np.all(y >= 0)
            and np.all(modelo.A.T @ y >= modelo.c - 1e-7)
        )
        cota = float(modelo.b @ y) if certificado_valido else math.nan
        cumple = bool(certificado_valido and math.isclose(
            solucion["Z"], cota, rel_tol=1e-7, abs_tol=1e-7
        ))
        comprobaciones["Cota teorica"] = cumple
        print(f"  Cota superior por suma ponderada de restricciones: "
              f"{numero(cota)} -> {'OK' if cumple else 'FALLO'}")
        print("  Pesos: " + ", ".join(numero(v) for v in y))
    if not all(comprobaciones.values()):
        raise ArithmeticError("La validacion del caso no coincide con la referencia.")
    print("RESULTADO: VALIDACION CORRECTA")
    return solucion


def leer_entero(mensaje):
    while True:
        try:
            valor = int(input(mensaje).strip())
            if valor < 1:
                raise ValueError
            return valor
        except ValueError:
            print("Ingrese un numero entero positivo.")


def leer_vector(mensaje, cantidad, no_negativo=False):
    """Lee numeros separados por espacios; acepta decimales y fracciones."""
    while True:
        try:
            tokens = input(mensaje).split()
            if len(tokens) != cantidad:
                raise ValueError(f"Se necesitan exactamente {cantidad} valores.")
            valores = [float(Fraction(token.replace(",", "."))) for token in tokens]
            if not all(math.isfinite(valor) for valor in valores):
                raise ValueError("Los valores deben ser finitos.")
            if no_negativo and any(valor < 0 for valor in valores):
                raise ValueError("El lado derecho debe ser no negativo.")
            return valores
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            print(f"Entrada invalida: {error}")
            print("Use espacios entre numeros; ejemplos: 3 1.5 -2 o 3 3/2 -2.")


def modo_interactivo(max_iteraciones=1000):
    print("\nINGRESO DE UN MODELO")
    print("Solo MAXIMIZACION, restricciones <=, b >= 0 y variables >= 0.")
    print("Separe los coeficientes con espacios. Se aceptan decimales y fracciones.")
    n = leer_entero("Numero de variables de decision: ")
    m = leer_entero("Numero de restricciones: ")
    c = leer_vector(f"Coeficientes de Z ({n} valores): ", n)
    A, b = [], []
    for i in range(m):
        A.append(leer_vector(f"R{i + 1}: coeficientes ({n} valores): ", n))
        b.append(leer_vector(f"R{i + 1}: lado derecho b (>= 0): ", 1, True)[0])
    return resolver_caso("MODELO INGRESADO POR EL USUARIO", c, A, b,
                        max_iteraciones=max_iteraciones)


def mostrar_info():
    print("PROTOTIPO 1.0 - METODO SIMPLEX TABULAR")
    print("Curso: Investigacion de Operaciones")
    print(f"Entorno: Python {platform.python_version()}, "
          f"NumPy {np.__version__}, Pandas {pd.__version__}")
    print("Incluye: maximizacion, <=, b >= 0, x >= 0, holguras y tablas por iteracion.")
    print("Excluye: minimizacion, >=, =, variables libres, artificiales, "
          "Gran M, Dos Fases e interfaz grafica.")
    print("Estructuras: matriz NumPy, listas de nombres y DataFrame para impresion.")
    print("Flujo: validar -> forma estandar -> tabla inicial -> optimalidad ->")
    print("       columna pivote -> cociente minimo -> Gauss-Jordan -> repetir.")
    print(f"Precision: aritmetica flotante; tolerancia de pivoteo {TOLERANCIA:g}.")
    print("Empates exactos: primera columna/fila. Ciclos y limite detienen el calculo.")
    print("No sustituye un solucionador numerico para modelos mal escalados.")
    print("Informe: simplex_proyecto.md. Ejemplos de clase: pendientes de confirmar.")


class SalidaDuplicada:
    """Muestra en consola y guarda el mismo texto para el anexo de pruebas."""

    def __init__(self, consola, registro):
        self.consola = consola
        self.registro = registro

    def write(self, texto):
        self.consola.write(texto)
        self.registro.write(texto)
        return len(texto)

    def flush(self):
        self.consola.flush()
        self.registro.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--interactivo", action="store_true", help="Ingresar un modelo por consola.")
    modo.add_argument("--info", action="store_true", help="Mostrar alcance, logica y versiones.")
    parser.add_argument("--reporte", metavar="ARCHIVO.txt",
                        help="Guardar la salida completa como evidencia en un archivo de texto.")
    parser.add_argument("--max-iteraciones", type=int, default=1000,
                        help="Limite de iteraciones por modelo (por defecto: 1000).")
    args = parser.parse_args(argv)
    if args.max_iteraciones < 1:
        parser.error("--max-iteraciones debe ser positivo.")
    ruta = Path(args.reporte).expanduser().resolve() if args.reporte else None
    if ruta is not None and ruta.suffix.lower() != ".txt":
        parser.error("El reporte debe tener extension .txt para proteger el codigo fuente.")
    registro = io.StringIO()
    codigo_salida = 0
    with redirect_stdout(SalidaDuplicada(sys.stdout, registro)):
        try:
            mostrar_info()
            if args.interactivo:
                modo_interactivo(args.max_iteraciones)
            elif not args.info:
                print("\nENTRADA: coeficientes definidos en CASOS_PRUEBA dentro del codigo.")
                for caso in CASOS_PRUEBA:
                    resolver_caso(**caso, max_iteraciones=args.max_iteraciones)
                print("\nRESUMEN: 2 de 2 ejemplos validados con su solucion teorica.")
                print("Confirmar con el equipo si son los ejercicios resueltos en clase.")
        except (ValueError, RuntimeError, ArithmeticError) as error:
            print(f"\nCALCULO DETENIDO: {error}")
            codigo_salida = 1
        except (EOFError, KeyboardInterrupt):
            print("\nEntrada cancelada. No se completo el modelo.")
            codigo_salida = 1
    if ruta is not None:
        try:
            ruta.write_text(registro.getvalue(), encoding="utf-8")
            print(f"\nReporte guardado en: {ruta}")
        except OSError as error:
            print(f"\nNo se pudo guardar el reporte: {error}", file=sys.stderr)
            codigo_salida = 1
    return codigo_salida


if __name__ == "__main__":
    raise SystemExit(main())
