
# importar librerías necesarias
from math import *
import numpy as np
import copy
import numpy.polynomial as poly
import scipy.interpolate as scip
import sympy as sp
import copy
import scipy.integrate as scin
import numpy.linalg as la
import scipy.optimize as scop
import scipy.linalg as sla
from random import randint
from scipy.interpolate import CubicSpline

def euclides(D,d):
    '''
    Esta función calcula el máximo común divisor de dos enteros dados utilizando
    el algoritmo de Euclides implementado en la función gcd de Python.

    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    D: primer número entero 
    d: segundo número entero 
    
    RETORNO
    -------
    c: máximo común divisor de D y d
    '''
    
    # Validación de tipos: verificamos que ambos parámetros sean enteros
    # Si no lo son, se lanza un AssertionError con un mensaje descriptivo
    assert type(D)==int, 'Debe introducir un número entero'
    assert type(d)==int, 'Debe introducir un número entero'

    # Calculamos el máximo común divisor utilizando la función gcd de Python
    # Esta función implementa el algoritmo de Euclides de forma eficiente
    c = gcd(D, d)

    return c

def raiznsima(a,n,niter,tol):
    '''
    Esta función calcula la raíz enésima de un número 'a' dado utilizando el
    método de Newton-Raphson. El método converge iterativamente hacia la solución
    de la ecuación x^n = a.

    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    a: número del cual se quiere calcular la raíz enésima
    n: índice de la raíz (debe ser entero positivo)
    niter: número máximo de iteraciones permitidas
    tol: tolerancia o error máximo permitido para detener las iteraciones
    
    RETORNO
    -------
    x: aproximación de la raíz enésima de a
    '''
    # Valor inicial: comenzamos con x = a como primera aproximación
    x = a
    
    # Iteración del método de Newton-Raphson para resolver x^n = a
    for i in range(niter):
        # Fórmula de Newton-Raphson específica para f(x) = x^n - a
        # xk+1 = x*(1-1/n) + a/(n*x^(n-1))
        # Esta fórmula se deriva de: x - f(x)/f'(x)
        xkm1 = x*(1-1/n) + a/(n*x**(n-1))
        
        # Criterio de parada: si la diferencia entre iteraciones consecutivas
        # es menor que la tolerancia, hemos alcanzado convergencia
        if abs(x-xkm1) < tol: 
            break  # Finaliza el bucle entero anticipadamente
        
        # Actualizamos x para la siguiente iteración
        x = xkm1
    
    return x


def exponencial(x,n):
    '''
    Esta función evalúa el desarrollo de Taylor de la función exponencial
    utilizando dos fórmulas distintas y compara su convergencia.
    
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    x: punto donde queremos evaluar la función exponencial
    n: número de términos del desarrollo de Taylor
    
    RETORNO
    -------
    f1: lista con las sumas parciales del desarrollo directo de e^x
    f2: lista con las sumas parciales usando la fórmula alternativa
    '''
    
    # PRIMERA FÓRMULA: Desarrollo directo de e^x = Σ(x^k / k!)
    # Esta es la serie de Taylor estándar de la exponencial
    t1 = 0  # Suma acumulada
    f1 = []  # Lista para almacenar sumas parciales

    for k in range(n+1):
        # Término k-ésimo: x^k / k!
        xk = (x**k) / factorial(k)
        # Suma acumulada hasta el término k
        tk = t1 + xk
        # Guardamos esta aproximación
        f1.append(tk)
        # Actualizamos la suma para la siguiente iteración
        t1 = tk
        
    # SEGUNDA FÓRMULA: Utilizando e^x = 1 / e^(-x)
    # Primero calculamos e^(-x) usando su serie de Taylor
    # y luego tomamos el recíproco
    t2 = 0  # Suma acumulada para e^(-x)
    f2 = []  # Lista para almacenar aproximaciones de e^x
    
    for k in range(n+1):
        # Término k-ésimo de e^(-x): (-x)^k / k! = (-1)^k * x^k / k!
        xk = (((-1)*x)**k) / factorial(k)
        # Suma parcial de e^(-x)
        tk = t2 + xk
        # Invertimos para obtener e^x = 1 / e^(-x)
        tk = 1 / tk
        # Guardamos esta aproximación
        f2.append(tk)
        # Actualizamos la suma para la siguiente iteración
        t2 = tk

    return f1, f2



def base_lagrange(x):
    '''
    Esta función construye la base de polinomios de Lagrange para un conjunto
    de nodos dados. Cada polinomio L_k(x) satisface:
    - L_k(x_k) = 1
    - L_k(x_j) = 0 para j ≠ k
    
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    x: nodos de interpolación (puede ser lista, tupla o array de numpy)
    
    RETORNO
    -------
    L: lista de polinomios de Lagrange, uno por cada nodo
    '''
    # Lista para almacenar los polinomios de Lagrange
    L = []
    suma = 0

    # * Preparar datos de entrada: convertir a lista
    # Aceptamos varios tipos de entrada para mayor flexibilidad
    
    if isinstance(x, tuple):
        x = list(x)
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, list):
        pass  # Ya está en el formato correcto
    else:
        raise ValueError("Tipo de dato no soportado")
    
    # Mensaje de depuración para verificar la entrada procesada
    print(f"tipo: {type(x)}, valor: {x} \n")
    
    # * Construcción de los polinomios de Lagrange
    # Para cada nodo x_i, construimos el polinomio L_i(x)
    
    for i in range(len(x)):
        # Creamos una copia del soporte completo
        soporte = x.copy()
        # Extraemos el nodo actual x_k del soporte
        x_k = soporte.pop(i)
        
        # Construimos el polinomio con raíces en todos los nodos excepto x_k
        # P(x) = (x - x_0)(x - x_1)...(x - x_{k-1})(x - x_{k+1})...(x - x_n)
        P = poly.Polynomial.fromroots(soporte)
        
        # Normalizamos dividing por P(x_k) para que L_k(x_k) = 1
        # L_k(x) = P(x) / P(x_k)
        L_k = P / P(x_k)
        
        # Añadimos este polinomio de Lagrange a la lista
        L.append(L_k)

    # * Retorno: lista con todos los polinomios de la base de Lagrange
    return L


def itp_Tchebishev(fun,ntps,a,b):
    '''
    Esta función construye un polinomio interpolante de una función dada
    utilizando nodos de Chebyshev. Los nodos de Chebyshev minimizan el
    error de interpolación (fenómeno de Runge).
    
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    fun: función a interpolar
    ntps: número de nodos de Chebyshev a utilizar
    a, b: extremos del intervalo de interpolación [a, b]
    
    RETORNO
    -------
    P_itp: polinomio interpolante en el intervalo [a, b]
    '''
    # * Cálculo de nodos de Chebyshev en el intervalo canónico [-1, 1]
    # Los nodos de Chebyshev son los ceros del polinomio de Chebyshev T_n(x)
    # Fórmula: x_k = cos((2k+1)π / (2n)) para k = 0, 1, ..., n-1
    xk = [np.cos((2*k+1)*np.pi/(2*ntps)) for k in range(ntps)]

    # * Transformación afín de nodos de [-1, 1] a [a, b]
    # Fórmula de cambio de variable: x_nuevo = (b-a)/2 * x_viejo + (a+b)/2
    xk_ab = [0.5*(b-a)*x + 0.5*(a+b) for x in xk]

    # * Construcción de la base de polinomios de Lagrange
    # Usando los nodos de Chebyshev transformados al intervalo [a, b]
    lagrange = base_lagrange(xk_ab)

    # * Construcción del polinomio interpolante
    # El polinomio interpolante se expresa como: P(x) = Σ f(x_i) * L_i(x)
    # donde L_i(x) son los polinomios de Lagrange y f(x_i) los valores de la función
    for i in range(len(xk_ab)):
        if i == 0:
            # Inicializamos con el primer término
            P_itp = fun(xk_ab[i]) * lagrange[i]
        else:
            # Sumamos los términos restantes
            P_itp += fun(xk_ab[i]) * lagrange[i]

    return P_itp

def itp_parametrica(data, bc_type="natural", u=None):
    """
    Calcula el interpolante paramétrico mediante splines cúbicas.
    
    Parámetros:
    -----------
    data : array-like, shape (n, 2) o (n, 3)
        Puntos a interpolar. Cada fila es un punto (x, y) o (x, y, z)
    bc_type : str, opcional (default='natural')
        Tipo de condiciones de frontera: 'natural', 'clamped', 'periodic'
    u : array-like, opcional (default=None)
        Valores del parámetro en cada punto. Si es None, se calcula
        proporcional a la distancia acumulada entre puntos.
    
    Retorna:
    --------
    funitp : callable
        Función vectorial de interpolación que acepta valores del parámetro
    param : array
        Valores del parámetro en los puntos dados
    """
    
    # Convertir data a numpy array
    data = np.array(data)
    n_points = data.shape[0]
    n_dims = data.shape[1]
    
    # Calcular parámetro u si no se proporciona
    if u is None:
        # Calcular distancias acumuladas entre puntos
        distances = np.zeros(n_points)
        for i in range(1, n_points):
            distances[i] = distances[i-1] + np.linalg.norm(data[i] - data[i-1])
        
        # Normalizar a [0, 1]
        if distances[-1] > 0:
            u = distances / distances[-1]
        else:
            u = np.linspace(0, 1, n_points)
    else:
        u = np.array(u)
    
    # Crear splines cúbicas para cada coordenada
    splines = []
    for dim in range(n_dims):
        spline = CubicSpline(u, data[:, dim], bc_type=bc_type)
        splines.append(spline)
    
    # Función interpolante vectorial
    def funitp(u_vals):
        """
        Evalúa la función interpolante en los valores del parámetro dados.
        
        Parámetros:
        -----------
        u_vals : float o array-like
            Valores del parámetro donde evaluar la interpolación
        
        Retorna:
        --------
        result : array
            Puntos interpolados. Si u_vals es escalar, retorna array de tamaño (n_dims,)
            Si u_vals es array, retorna array de tamaño (len(u_vals), n_dims)
        """
        u_vals = np.atleast_1d(u_vals)
        result = np.zeros((len(u_vals), n_dims))
        
        for dim in range(n_dims):
            result[:, dim] = splines[dim](u_vals)
        
        # Si la entrada fue escalar, retornar resultado escalar
        if len(u_vals) == 1:
            return result[0]
        return result
    
    return funitp, u


def dncoef_base(soporte, puntos, orden):
    '''
    Esta función calcula los coeficientes de una regla de derivación numérica
    utilizando la base de polinomios de Lagrange. Los coeficientes obtenidos
    permiten aproximar derivadas de funciones mediante combinaciones lineales
    de valores de la función en los nodos del soporte.
    
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    soporte: nodos que definen los polinomios de la base de Lagrange
    puntos: punto(s) donde se evalúan los coeficientes de derivación
    orden: orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
    
    RETORNO
    -------
    coef: coeficientes de la regla de derivación numérica
    '''
    
    # Validación de entrada: el orden debe ser un entero positivo
    if not isinstance(orden, int) or orden <= 0:
        raise ValueError('El orden de la derivada debe ser un número entero positivo')
    
    # * Paso 1: Construcción de la base de Lagrange
    # Construimos los polinomios de Lagrange L_i(x) para el soporte dado
    Pol_Lg = base_lagrange(soporte)
    
    # * Paso 2: Derivación de los polinomios de Lagrange
    # Para obtener los coeficientes de derivación numérica, derivamos
    # cada polinomio de Lagrange 'orden' veces
    Derivadas = []
    
    for L in Pol_Lg:
        # Calculamos la derivada de orden 'orden' del polinomio L_i(x)
        dif = L.deriv(orden)
        Derivadas.append(dif)
    
    # Convertimos a array de numpy para facilitar operaciones
    Derivadas = np.array(Derivadas)
    # Nota: También se podría usar list comprehension:
    # Derivadas = [L.deriv(orden) for L in Pol_Lg]
    
    # * Paso 3: Evaluación de las derivadas en los puntos dados
    # Los coeficientes son los valores de d^n L_i(x) / dx^n evaluados en los puntos
    
    if isinstance(puntos, (int, float)):
        # Caso 1: Un solo punto
        # Evaluamos todas las derivadas en ese punto
        coef = []
        for i in range(len(Derivadas)):
            coef.append(Derivadas[i](puntos))
        
    elif isinstance(puntos, (list, tuple, np.ndarray)):
        # Caso 2: Múltiples puntos
        # Evaluamos las derivadas en cada punto
        
        coef_0 = []
        for x in puntos:
            # Para cada punto, evaluamos todas las derivadas de Lagrange
            coef_0 = [d(x) for d in Derivadas]

        # Redondeamos los coeficientes a 2 decimales para presentación
        coef = [round(float(x), 2) for x in coef_0]

    return coef



def deriva2(fun, puntos, h):
    '''
    Esta función calcula la derivada segunda de una función en punto(s) dado(s)
    utilizando cuatro métodos de diferencias finitas distintos. Permite comparar
    la precisión y convergencia de diferentes esquemas numéricos.
    
    Los métodos implementados son:
    1. Diferencias hacia adelante (orden h^2)
    2. Diferencias centradas (orden h^2)
    3. Diferencias hacia adelante con 4 puntos (orden h^2)
    4. Diferencias centradas con 5 puntos (orden h^4)

    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    fun: función de la cual se quiere calcular la derivada segunda
    puntos: punto(s) donde calcular la derivada (puede ser escalar o vector)
    h: paso de discretización (puede ser escalar o vector)
    
    RETORNO
    -------
    der2: lista de listas con los resultados de cada método
    formato: [[método1], [método2], [método3], [método4]]
    '''
    if isinstance(puntos, (list,tuple, np.ndarray)) and isinstance(h,(int,float)):
        # CASO 1: Múltiples puntos con un solo valor de h
        # Calculamos la derivada segunda en varios puntos usando el mismo paso h
        
        # Estructura de salida: lista de listas con resultados de cada método
        # der2 = [[método 1], [método 2], [método 3], [método 4]]
        der2 = []

        # Listas para almacenar resultados de cada método
        m1 = []  # Diferencias hacia adelante
        m2 = []  # Diferencias centradas (3 puntos)
        m3 = []  # Diferencias hacia adelante (4 puntos)
        m4 = []  # Diferencias centradas (5 puntos, alta precisión)
        
        for x in puntos:
            # Método 1: Diferencias hacia adelante con 3 puntos
            # f''(x) ≈ [f(x) - 2f(x+h) + f(x+2h)] / h^2
            # Error: O(h^2)
            f1 = (fun(x) - 2*fun(x+h) + fun(x+2*h)) / (h**2)
            m1.append(f1)
            
            # Método 2: Diferencias centradas con 3 puntos
            # f''(x) ≈ [f(x-h) - 2f(x) + f(x+h)] / h^2
            # Error: O(h^2), más preciso que el método 1
            f2 = (fun(x-h) - 2*fun(x) + fun(x+h)) / (h**2)
            m2.append(f2)
            
            # Método 3: Diferencias hacia adelante con 4 puntos
            # f''(x) ≈ [2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)] / h^2
            # Error: O(h^2)
            f3 = (2*fun(x) - 5*fun(x+h) + 4*fun(x+2*h) - fun(x+3*h)) / (h**2)
            m3.append(f3)
            
            # Método 4: Diferencias centradas con 5 puntos (alta precisión)
            # f''(x) ≈ [-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)] / (12h^2)
            # Error: O(h^4), el más preciso de todos
            f4 = (-fun(x-2*h) + 16*fun(x-h) - 30*fun(x) + 16*fun(x+h) - fun(x+2*h)) / (12*h**2)
            m4.append(f4)
        
        # Organizamos los resultados por método
        der2.append(m1)
        der2.append(m2)
        der2.append(m3)
        der2.append(m4)

    elif isinstance(puntos, (int,float)) and isinstance(h,(list,tuple, np.ndarray)):
        # CASO 2: Un solo punto con múltiples valores de h
        # Útil para estudiar la convergencia del método al variar el paso h
        
        # Estructura de salida: lista de listas con resultados de cada método
        # para diferentes valores de h
        der2 = []

        # Listas para almacenar resultados de cada método
        m1 = []  # Diferencias hacia adelante
        m2 = []  # Diferencias centradas (3 puntos)
        m3 = []  # Diferencias hacia adelante (4 puntos)
        m4 = []  # Diferencias centradas (5 puntos, alta precisión)
        
        for i in h:
            # Aplicamos los mismos cuatro métodos pero con diferentes valores de h
            
            # Método 1: Diferencias hacia adelante
            h1 = (fun(puntos) - 2*fun(puntos+i) + fun(puntos+2*i)) / (i**2)
            m1.append(h1)
            
            # Método 2: Diferencias centradas (3 puntos)
            h2 = (fun(puntos-i) - 2*fun(puntos) + fun(puntos+i)) / (i**2)
            m2.append(h2)
            
            # Método 3: Diferencias hacia adelante (4 puntos)
            h3 = (2*fun(puntos) - 5*fun(puntos+i) + 4*fun(puntos+2*i) - fun(puntos+3*i)) / (i**2)
            m3.append(h3)
            
            # Método 4: Diferencias centradas (5 puntos, alta precisión)
            h4 = (-fun(puntos-2*i) + 16*fun(puntos-i) - 30*fun(puntos) + 16*fun(puntos+i) - fun(puntos+2*i)) / (12*i**2)
            m4.append(h4)
        
        # Organizamos los resultados por método
        der2.append(m1)
        der2.append(m2)
        der2.append(m3)
        der2.append(m4)

    elif isinstance(puntos, (int,float)) and isinstance(h,(int,float)):
        # CASO 3: Un solo punto y un solo valor de h
        # Calcula la derivada segunda en un punto específico con un paso h dado
        
        # Estructura de salida: lista simple con un resultado por método
        # der2 = [método 1, método 2, método 3, método 4]
        der2 = []

        # Método 1: Diferencias hacia adelante
        f1 = (fun(puntos) - 2*fun(puntos+h) + fun(puntos+2*h)) / (h**2)
        der2.append(f1)
        
        # Método 2: Diferencias centradas (3 puntos)
        f2 = (fun(puntos-h) - 2*fun(puntos) + fun(puntos+h)) / (h**2)
        der2.append(f2)
        
        # Método 3: Diferencias hacia adelante (4 puntos)
        f3 = (2*fun(puntos) - 5*fun(puntos+h) + 4*fun(puntos+2*h) - fun(puntos+3*h)) / (h**2)
        der2.append(f3)
        
        # Método 4: Diferencias centradas (5 puntos, alta precisión)
        f4 = (-fun(puntos-2*h) + 16*fun(puntos-h) - 30*fun(puntos) + 16*fun(puntos+h) - fun(puntos+2*h)) / (12*h**2)
        der2.append(f4)

    else:
        # Control de errores: no se permite que puntos y h sean ambos vectores
        raise ValueError('Puntos y h no pueden ser vectores simultáneamente')
    
    # Redondeo de resultados a 2 decimales para mejor presentación
    # Verificar si der2 es una lista de listas o lista simple
    if isinstance(der2[0], list):
        # Casos 1 y 2: lista de listas
        der2 = [[round(float(x), 2) for x in lista] for lista in der2]
    else:
        # Caso 3: lista simple
        der2 = [round(float(x), 2) for x in der2]
    
    return der2



def incoef_base(soporte,a,b):
    '''
    Esta función calcula los coeficientes (pesos) de una regla de integración
    numérica basada en la cuadratura de Newton-Cotes. Los coeficientes obtenidos
    se pueden usar para aproximar integrales mediante:
    ∫_a^b f(x)dx ≈ Σ w_i * f(x_i)
    donde w_i son los coeficientes y x_i los nodos del soporte.
    
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    soporte: nodos de integración que definen los polinomios de Lagrange
    a, b: límites de integración [a, b]
    
    RETORNO
    -------
    Integrales: lista de coeficientes (pesos) w_i de la regla de integración
    '''
    
    # * Paso 1: Construcción de la base de Lagrange
    # Los polinomios de Lagrange L_i(x) forman una base para el espacio
    # de polinomios de grado ≤ n-1 (donde n es el número de nodos)
    Pol_Lg = base_lagrange(soporte)
    
    # * Paso 2: Integración de los polinomios de Lagrange
    # Los coeficientes de integración son w_i = ∫_a^b L_i(x) dx
    # Estos coeficientes dependen solo de los nodos y del intervalo,
    # no de la función a integrar
    Integrales = []

    for L in Pol_Lg:
        # Calculamos la primitiva de L_i(x) con límite inferior en a
        # int_a_x(x) = ∫_a^x L_i(t) dt
        int_a_x = L.integ(lbnd=a)
        
        # Evaluamos la primitiva en b para obtener ∫_a^b L_i(x) dx
        int_a_b = int_a_x(b)
        
        # Guardamos este coeficiente
        Integrales.append(int_a_b)

    return Integrales


def in_romberg(fun,a,b,nivel=10,tol=1e-6):
    '''
    Esta función calcula la integral definida de una función utilizando el método
    de Romberg, que combina la regla del trapecio con extrapolación de Richardson
    para obtener alta precisión. El método construye una tabla triangular donde
    cada entrada mejora la aproximación mediante eliminación sucesiva de términos
    de error.
    
    El método itera hasta alcanzar la tolerancia especificada o el nivel máximo.

    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    fun: función a integrar
    a, b: límites de integración [a, b]
    nivel: número máximo de refinamientos (por defecto 10)
    tol: tolerancia para el criterio de parada (por defecto 1e-6)
    
    RETORNO
    -------
    N[n][n]: mejor aproximación de la integral obtenida
    delta: estimación del error en la última iteración
    N: tabla completa de Romberg con todas las aproximaciones
    '''
    # 1. Crear matriz N
    N = np.zeros((nivel, nivel))

    # 2. Inicialización
    h = b - a
    N[0][0] = (fun(a) + fun(b)) * (h / 2)
    p = 1  # Puntos a añadir

    # Variable para almacenar el error de la diagonal
    last_delta = 0.0

    # 3. Iteración principal
    for n in range(1, nivel):
        # a) Modificar h
        h = h / 2

        # b) Calcular p puntos equiespaciados (los nuevos impares)
        puntos = [a + (2 * i - 1) * h for i in range(1, p + 1)]

        # c) Trapecio compuesto (fórmula recursiva)
        # Nota: Usamos sum() generador para eficiencia
        N[n][0] = 0.5 * N[n-1][0] + h * sum(fun(x) for x in puntos)
        
        # d) Extrapolación de Richardson
        q = 1
        for j in range(1, n + 1):
            # 1) Modificar q (4^j)
            q = q * 4
            
            # 2) Calcular corrección delta
            delta = (1 / (q - 1)) * (N[n][j-1] - N[n-1][j-1])
            
            # 3) Obtener nuevo valor de la columna
            N[n][j] = N[n][j-1] + delta
            
            # Si estamos en la diagonal, guardamos este delta para comparar
            if j == n:
                last_delta = delta

        # e) Criterio de parada
        # Se verifica solo si la corrección de la diagonal cumple la tolerancia
        if abs(last_delta) < tol:
            return N[n][n], last_delta, N
        else:
            # Si no converge, simplemente duplicamos p y DEJAMOS que el bucle siga
            p = 2 * p

    # Si terminamos todos los niveles sin llegar a la tolerancia:
    return N[nivel-1][nivel-1], last_delta, N


def paracaidista(y0, v0, m, cx, At, apertura=1500, rovar=False):
    '''
    Determina el tiempo y la velocidad a la que toma tierra un paracaidista.
    
    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    y0 : posición inicial del salto (m)
    v0 : velocidad inicial del salto (m/s)
    m : masa del paracaidista equipado (kg)
    cx : iterable con coeficientes de arrastre [antes_apertura, después_apertura]
    At : área transversal (m²)
    apertura : altura a la que se abre el paracaídas (m), por defecto 1500
    rovar : valor lógico que indica si la densidad es variable o no, por defecto False
    
    RESULTADO
    ---------
    Lista con cuatro valores: [v_max, v_impacto, t_apertura, t_total]
        - v_max: velocidad máxima alcanzada (m/s)
        - v_impacto: velocidad de impacto en el suelo (m/s)
        - t_apertura: tiempo hasta que abre el paracaídas (s)
        - t_total: tiempo total de vuelo (s)
    '''
    
    # Constantes físicas
    g = 9.81  # Aceleración de la gravedad (m/s²)
    rho_0 = 1.225  # Densidad del aire a nivel del mar (kg/m³)
    
    # Tiempos máximos estimados para cada fase
    t_fin1 = 100  # Tiempo máximo fase 1 (caída libre)
    t_fin2 = 500  # Tiempo máximo fase 2 (con paracaídas)
    
    # Condiciones iniciales
    ci = [y0, v0]
    
    # * Sistema de ecuaciones diferenciales
    def sedo(t, Y, m, cx_val, At, rovar):
        '''
        Sistema de EDOs para el movimiento del paracaidista
        Y[0] = y(t): altura
        Y[1] = v(t): velocidad
        '''
        # Densidad del aire (constante o variable)
        rho = rho_0
        if rovar:
            rho = rho_0 * np.exp(-Y[0]/8243)
        
        # Coeficiente de arrastre
        kw = cx_val * rho * At / 2
        
        # Sistema de ecuaciones: [dy/dt, dv/dt]
        dY = np.array([
            Y[1],  # dy/dt = v
            -g - (kw * Y[1] * abs(Y[1])) / m  # dv/dt = -g - (arrastre)/m
        ])
        return dY
    
    # * FASE 1: Caída libre hasta la apertura del paracaídas
    
    # Evento: apertura del paracaídas en la altura especificada
    def abreParaca(t, Y, m, cx_val, At, rovar):
        return Y[0] - apertura
    abreParaca.terminal = True  # Detiene la integración
    abreParaca.direction = -1   # Solo cuando Y[0] decrece
    
    # Resolver fase 1 con cx[0] (antes de la apertura)
    sol1 = sp.integrate.solve_ivp(
        sedo, 
        [0, t_fin1], 
        ci, 
        args=[m, cx[0], At, rovar], 
        events=abreParaca,
        dense_output=True
    )
    
    # * FASE 2: Descenso con paracaídas hasta el suelo
    
    # Evento: impacto con el suelo
    def impactoSuelo(t, Y, m, cx_val, At, rovar):
        return Y[0]
    impactoSuelo.terminal = True  # Detiene la integración
    impactoSuelo.direction = -1   # Solo cuando Y[0] decrece
    
    # Condiciones iniciales de la fase 2: estado final de la fase 1
    ci2 = [sol1.y[0, -1], sol1.y[1, -1]]
    t_inicio2 = sol1.t[-1]
    
    # Resolver fase 2 con cx[1] (después de la apertura)
    sol2 = sp.integrate.solve_ivp(
        sedo,
        [t_inicio2, t_fin2],
        ci2,
        args=[m, cx[1], At, rovar],
        events=impactoSuelo,
        dense_output=True
    )
    
    # * Cálculo de resultados
    
    # Velocidad máxima (en valor absoluto) en toda la trayectoria
    v_max = max(np.max(np.abs(sol1.y[1])), np.max(np.abs(sol2.y[1])))
    
    # Velocidad de impacto (en el último punto de sol2)
    v_impacto = abs(sol2.y[1, -1])
    
    # Tiempo de apertura del paracaídas
    t_apertura = sol1.t[-1]
    
    # Tiempo total de vuelo
    t_total = sol2.t[-1]
    
    return [v_max, v_impacto, t_apertura, t_total]

def disparo(F, ab, cc, mi=[0, 1], niter=100, xtol=1e-6, ftol=1e-9, **opt):
    '''
    Resuelve un problema de valores en la frontera (P.V.F.) transformándolo en
    un problema de valores iniciales (P.V.I.) mediante el método de disparo.
    
    PARÁMETROS:
    -----------
    F : función (vectorial) que define el SEDO
    ab : intervalo de resolución [a, b]
    cc : condiciones de contorno (Dirichlet) en los extremos [y(a), y(b)]
    mi : valores de la pendiente en las dos primeras iteraciones (default: [0, 1])
    niter : número máximo de iteraciones (default: 100)
    xtol : error admisible en la pendiente (default: 1e-6)
    ftol : error admisible en la solución (default: 1e-9)
    **opt : opciones para solve_ivp (method, dense_output, events)
            Por defecto: method='RK45', dense_output=False, events=None
    
    RESULTADOS:
    -----------
    Estructura de tipo solve_ivp con la solución del problema
    '''
    
    # Configuración de opciones por defecto
    opc = {"method": "RK45", "dense_output": False, "events": None}
    
    # Actualizar opciones con las proporcionadas
    for clave, valor in opt.items():
        clave_lower = clave.lower()
        if clave_lower in ["method", "dense_output", "events"]:
            opc[clave_lower] = valor
        else:
            print(f"Opción '{clave}' inválida")
    
    # 1. Resolución con m1 y obtención de w_n(m1)
    s1 = sp.integrate.solve_ivp(
        F, ab, [cc[0], mi[0]], 
        method=opc['method'], 
        dense_output=opc['dense_output'], 
        events=opc['events']
    )
    w_m1 = s1.y[0][-1]  # w_n(m1)
    
    # 2. Resolución con m2 y obtención de w_n(m2)
    s2 = sp.integrate.solve_ivp(
        F, ab, [cc[0], mi[1]], 
        method=opc['method'], 
        dense_output=opc['dense_output'], 
        events=opc['events']
    )
    w_m2 = s2.y[0][-1]  # w_n(m2)
    
    # Inicialización de variables
    m_k = mi[0]
    m_k1 = mi[1]
    w_k = w_m1
    w_k1 = w_m2
    
    # 3. Bucle para k = 1, ..., niter
    for k in range(1, niter + 1):
        # a) Cálculo de la nueva pendiente mk+2
        m_k2 = m_k1 + (cc[1] - w_k1) * ((m_k1 - m_k) / (w_k1 - w_k))
        
        # b) Resolución con mk+2 y obtención de w_n(mk+2)
        s_k2 = sp.integrate.solve_ivp(
            F, ab, [cc[0], m_k2], 
            method=opc['method'], 
            dense_output=opc['dense_output'], 
            events=opc['events']
        )
        w_k2 = s_k2.y[0][-1]  # w_n(mk+2)
        
        # c) Si |mk+2 - mk+1| < xtol, devolver sk+2 y finalizar
        if abs(m_k2 - m_k1) < xtol:
            return s_k2
        
        # d) Si |w_n(mk+2) - β| < ftol, devolver sk+2 y finalizar
        if abs(w_k2 - cc[1]) < ftol:
            return s_k2
        
        # Actualizar variables para la siguiente iteración
        m_k, m_k1 = m_k1, m_k2
        w_k, w_k1 = w_k1, w_k2
    
    # Si se alcanza el máximo de iteraciones
    return s_k2



def ensolver(funx, a, b, meth='rf', maxiter=128, Ex=1e-9, ex=1e-6, EF=1e-12):
    '''
    Resuelve ecuaciones no lineales mediante métodos de intervalo (bisección/Regula Falsi).
    
    CONDICIONES PREVIAS (Teorema de Bolzano):
    - La función debe ser continua en [a, b]
    - La función debe cambiar de signo en los extremos: f(a) * f(b) < 0
    
    PROCESO:
    En cada iteración se reduce el intervalo de búsqueda hasta cumplir el criterio de parada.
    
    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    funx: función cuyo cero se desea determinar
    a, b: extremos del intervalo inicial [a, b]
    meth: método de resolución
          - 'di': Dicotomía (bisección)
          - 'rf': Regula Falsi (posición falsa) [por defecto]
          - 'fm': Regula Falsi modificada
    maxiter: número máximo de iteraciones permitidas [128 por defecto]
    Ex: tolerancia absoluta del intervalo [1e-9 por defecto]
    ex: tolerancia relativa del intervalo [1e-6 por defecto]
    EF: tolerancia para el valor de la función [1e-12 por defecto]
    
    RETORNO
    -------
    r: cero aproximado de la función (None si hay error)
    info: código de terminación
          -2: método desconocido
          -1: no verifica el teorema de Bolzano
           0: tolerancia absoluta o relativa del intervalo alcanzada
           1: tolerancia de la función alcanzada
           2: número máximo de iteraciones alcanzado
    suc: lista con la sucesión de aproximaciones {xn}
    '''
    
    # ========================================================================
    # PASO 1: VALIDACIÓN INICIAL Y ASIGNACIONES
    # ========================================================================
    
    # Verificar que el método sea válido
    metodos_validos = ['di', 'rf', 'fm']
    if meth not in metodos_validos:
        return None, -2, []
    
    # 1. Asignar fa ← f(a), fb ← f(b), [fxn ← 0]*
    fa = funx(a)
    fb = funx(b)
    fxn = 0  # Solo para Regula Falsi modificada (indicado con [...]*)
    
    # a) Si |fa| < εF, finalizar r = a
    if abs(fa) < EF:
        return a, 1, [a]
    
    # b) Si |fb| < εF, finalizar r = b
    if abs(fb) < EF:
        return b, 1, [b]
    
    # c) Si fa · fb > 0, finalizar: error no verifica Bolzano
    if fa * fb > 0:
        return None, -1, []
    
    # Inicialización de la lista de aproximaciones sucesivas
    suc = []
    info = 2  # Por defecto, asumimos que se alcanzará maxiter
    
    # ========================================================================
    # PASO 2: REPETIR UN MÁXIMO DE n ITERACIONES
    # ========================================================================
    for i in range(maxiter):
        
        # a) Elegir un valor xn ∈ (an, bn)
        if meth == 'di':
            # 1) Bisección o dicotomía: xn = 1/2 (an + bn)
            xn = 0.5 * (a + b)
        
        elif meth == 'rf' or meth == 'fm':
            # 2) Regula Falsi: xn = a·f(b) − b·f(a) / [f(b) − f(a)]
            # Equivalente a: xn = a - fa / [(fb - fa) / (b - a)]
            xn = (a * fb - b * fa) / (fb - fa)
        
        # b) Asignar fx ← f(xn)
        fx = funx(xn)
        suc.append(xn)
        
        # c) Si |fx| < EF, finalizar r = xn
        if abs(fx) < EF:
            info = 1
            break
        
        # d) Si fx · fa > 0 entonces asignar fa ← fx, a ← x
        if fx * fa > 0:
            a = xn
            fa = fx
            # [Si fx · fxn > 0 entonces asignar fb ← 1/2 fb]* (solo para fm)
            if meth == 'fm' and i > 0 and fx * fxn > 0:
                fb = 0.5 * fb
        
        # e) sino asignar fb ← fx, b ← x
        else:
            b = xn
            fb = fx
            # [Si fx · fxn > 0 entonces asignar fa ← 1/2 fa]* (solo para fm)
            if meth == 'fm' and i > 0 and fx * fxn > 0:
                fa = 0.5 * fa
        
        # f) [Asignar fxn ← fx]* (solo para fm)
        if meth == 'fm':
            fxn = fx
        
        # g) Asignar xtol ← máx(EX, eX |xn|)
        xtol = max(Ex, ex * abs(xn))
        
        # h) Si |b − a| < xtol, finalizar r = xn
        if abs(b - a) < xtol:
            info = 0
            break
    
    # ========================================================================
    # PASO 3: ACABADO EL BUCLE SIN CONVERGENCIA
    # ========================================================================
    # Finalizar r = xn (la última aproximación calculada)
    r = suc[-1] if suc else None
    
    return r, info, suc


def autoval_potencia(A, delta=np.inf, tol=1E-6, niter=100):
    """
    Calcula el autovalor más próximo al valor indicado mediante el método de la potencia 
    o la potencia inversa.
    
    Parámetros:
    -----------
    A : ndarray
        Matriz cuadrada a estudiar.
    delta : float, opcional
        Referencia del autovalor. Por defecto np.inf.
        - ±∞: calcula el autovalor mayor en valor absoluto (método de la potencia)
        - 0: calcula el autovalor menor no nulo
        - cualquier otro valor: calcula el autovalor más próximo a delta (método potencia inversa)
    tol : float o list/tuple, opcional
        Tolerancia del error. Por defecto 1E-6.
        - Si es un valor: se mide con norma 2
        - Si es un par [tol, p]: usa la norma p (1, 2, o np.inf)
    niter : int, opcional
        Número máximo de iteraciones. Por defecto 100.
    
    Retorna:
    --------
    valor : float
        Autovalor aproximado.
    vector : ndarray
        Autovector aproximado normalizado.
    
    Algoritmo:
    ----------
    1. Si δ = ±∞: usa el método de la potencia directa
    2. Si δ ≠ ±∞: usa el método de la potencia inversa con factorización PLU
    """
    
    # 1. Calcular el orden n de la matriz
    n = A.shape[0]
    
    # 2. Interrumpir si la matriz no es cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada")
    
    # 3. Obtener la tolerancia ε y la norma p
    if isinstance(tol, (list, tuple)):
        eps = tol[0]
        p = tol[1]
    else:
        eps = tol
        p = 2
    
    # 4. Generar un vector aleatorio w0 y normalizarlo como u0
    w0 = np.random.rand(n)
    u0 = w0 / la.norm(w0, 2)
    
    # 5. SI δ = ±∞ (Método de la potencia)
    if delta == np.inf or delta == -np.inf:
        uk_prev = u0
        
        # a) Repetir para k desde 1 hasta el máximo de iteraciones
        for k in range(1, niter + 1):
            # 1) Calcular wk = A·uk-1, uk = wk/||wk||_2
            wk = A @ uk_prev
            uk = wk / la.norm(wk, 2)
            
            # 2) Si ||uk - uk-1||_p < ε devolver (uk^T · A · uk, uk)
            if la.norm(uk - uk_prev, p) < eps:
                valor = uk.T @ A @ uk
                return (valor, uk)
            
            uk_prev = uk
        
        # b) Aviso de fin de iteraciones y devolver (uk^T · A · uk, uk)
        print(f"Advertencia: Se alcanzó el máximo de iteraciones ({niter}) sin convergencia")
        valor = uk.T @ A @ uk
        return (valor, uk)
    
    # 6. SINO (Método de la potencia inversa)
    else:
        # a) Factorizar B = (A - δI) = PLU
        I = np.eye(n)
        B = A - delta * I
        P, L, U = sla.lu(B)
        
        # b) SI algún elemento de la diagonal de U se anula devolver δ y None
        if np.any(np.abs(np.diag(U)) < 1e-14):
            print(f"El valor {delta} es un autovalor exacto de la matriz")
            return (delta, None)
        
        uk_prev = u0
        
        # c) Repetir para k desde 1 hasta el máximo de iteraciones
        for k in range(1, niter + 1):
            # 1) Resolver Ly = P·uk-1
            y = la.solve(L, P @ uk_prev)
            
            # 2) Resolver U·wk = y
            wk = la.solve(U, y)
            
            # 3) Normalizar uk = wk/||wk||_2
            uk = wk / la.norm(wk, 2)
            
            # 4) Si ||uk - uk-1||_p < ε devolver (uk^T·B·uk + δ, uk)
            if la.norm(uk - uk_prev, p) < eps:
                valor = uk.T @ B @ uk + delta
                return (valor, uk)
            
            uk_prev = uk
        
        # d) Aviso de fin de iteraciones y devolver (uk^T·B·uk + δ, uk)
        print(f"Advertencia: Se alcanzó el máximo de iteraciones ({niter}) sin convergencia")
        valor = uk.T @ B @ uk + delta
        return (valor, uk)


def basegj(C1, L1, C2, L2, C3, L3):
    """
    Determina las posiciones de intersección entre dos elipsoides y un cilindro parabólico.
    
    Problema: La base submarina de GI-Joe está formada por dos domos elipsoidales con 
    una entrada tubular. Se debe determinar dónde las tres superficies se intersectan.
    
    Parámetros:
    -----------
    C1 : list o array
        Centro del primer elipsoide [x, y, z]
    L1 : list o array
        Semiejes del primer elipsoide [a, b, c]
    C2 : list o array
        Centro del segundo elipsoide [x, y, z]
    L2 : list o array
        Semiejes del segundo elipsoide [a, b, c]
    C3 : list o array
        Centro del cilindro parabólico [x, y, z] (se ignora x)
    L3 : list o array
        Parámetros del cilindro [ignorado, b, c] (se ignora el primer elemento)
    
    Retorna:
    --------
    tuple
        Tupla con todas las soluciones encontradas (puntos de intersección)
    
    Ecuaciones del sistema:
    -----------------------
    Elipsoide 1: ((x-C1[0])/L1[0])² + ((y-C1[1])/L1[1])² + ((z-C1[2])/L1[2])² = 1
    Elipsoide 2: ((x-C2[0])/L2[0])² + ((y-C2[1])/L2[1])² + ((z-C2[2])/L2[2])² = 1
    Cilindro:    ((y-C3[1])/L3[1])² + ((z-C3[2])/L3[2])² = 1
    """
    
    # Definir el sistema de ecuaciones no lineales
    # F(x, y, z) = 0 representa las tres superficies
    F = lambda x: np.array([
        # Ecuación del primer elipsoide
        (x[0] - C1[0])**2 / L1[0]**2 + 
        (x[1] - C1[1])**2 / L1[1]**2 + 
        (x[2] - C1[2])**2 / L1[2]**2 - 1,
        
        # Ecuación del segundo elipsoide
        (x[0] - C2[0])**2 / L2[0]**2 + 
        (x[1] - C2[1])**2 / L2[1]**2 + 
        (x[2] - C2[2])**2 / L2[2]**2 - 1,
        
        # Ecuación del cilindro parabólico (no depende de x)
        (x[1] - C3[1])**2 / L3[1]**2 + 
        (x[2] - C3[2])**2 / L3[2]**2 - 1
    ])
    
    # Resolver el sistema con diferentes puntos iniciales para encontrar todas las soluciones
    # Debido a la simetría del problema, buscamos en los cuatro cuadrantes (x, z)
    soluciones = []
    
    # Puntos iniciales en diferentes regiones del espacio
    puntos_iniciales = [
        [7, -4, 1.5],    # Cuadrante (+x, +z)
        [-7, -4, 1.5],   # Cuadrante (-x, +z)
        [7, -4, -1.5],   # Cuadrante (+x, -z)
        [-7, -4, -1.5]   # Cuadrante (-x, -z)
    ]
    
    # Buscar soluciones desde cada punto inicial
    for p0 in puntos_iniciales:
        sol = scop.fsolve(F, p0)
        # Verificar que la solución es válida (residuo pequeño)
        if np.linalg.norm(F(sol)) < 1e-6:
            # Verificar si esta solución ya fue encontrada
            es_nueva = True
            for s in soluciones:
                if np.linalg.norm(sol - s) < 1e-4:
                    es_nueva = False
                    break
            if es_nueva:
                soluciones.append(sol)
    
    return tuple(soluciones)



def aproxmc1c(base, ab, funcion):
    """
    Plantea y resuelve las ecuaciones normales para un ajuste por mínimos cuadrados 
    continuo en un intervalo dado.
    
    El método encuentra los coeficientes c = [c₁, c₂, ..., cₙ] que minimizan el 
    error cuadrático entre la función f(x) y su aproximación g(x) = Σ cᵢ φᵢ(x).
    
    Parámetros:
    -----------
    base : list
        Lista con las funciones de la base B₁ = {φ₁(x), ..., φₙ(x)}
        Cada elemento debe ser una función callable.
    ab : list o tuple
        Intervalo de aproximación [a, b]
    funcion : callable
        Función a aproximar f(x)
    
    Retorna:
    --------
    c : ndarray
        Vector con los coeficientes en la base
    Ecm : float
        Error cuadrático medio
    r2 : float
        Coeficiente de determinación
    
    Algoritmo:
    ----------
    1. Construye la matriz de Gram G donde Gᵢⱼ = ⟨φᵢ, φⱼ⟩ = ∫ φᵢ(x)φⱼ(x)dx
    2. Construye el vector F donde Fᵢ = ⟨f, φᵢ⟩ = ∫ f(x)φᵢ(x)dx
    3. Resuelve el sistema Gc = F
    4. Calcula Ec = ⟨f,f⟩ - c^T·F
    5. Calcula Ecm = Ec/(b-a)
    6. Calcula r² = 1 - Ec/S²ᵧ donde S²ᵧ = ∫(f(x)-ȳ)²dx
    """
    
    # Extraer límites del intervalo
    a, b = ab[0], ab[1]
    n = len(base)
    
    # Inicializar matriz de Gram y vector de productos internos
    G = np.zeros((n, n))
    F = np.zeros(n)
    
    # Construir la matriz de Gram G y el vector F
    for i in range(n):
        # Calcular Fᵢ = ⟨f, φᵢ⟩ = ∫ f(x)·φᵢ(x) dx
        h = lambda x, idx=i: funcion(x) * base[idx](x)
        F[i] = scin.quad(h, a, b)[0]
        
        # Calcular Gᵢⱼ = ⟨φᵢ, φⱼ⟩ = ∫ φᵢ(x)·φⱼ(x) dx
        for j in range(n):
            producto = lambda x, idx_i=i, idx_j=j: base[idx_i](x) * base[idx_j](x)
            G[i, j] = scin.quad(producto, a, b)[0]
    
    # Resolver el sistema de ecuaciones normales Gc = F
    c = la.solve(G, F)
    
    # Calcular el error cuadrático: Ec = ⟨f,f⟩ - c^T·F
    producto_ff = lambda x: funcion(x)**2
    ff = scin.quad(producto_ff, a, b)[0]
    Ec = ff - np.dot(c, F)
    
    # Calcular el error cuadrático medio: Ecm = Ec/(b-a)
    Ecm = Ec / (b - a)
    
    # Calcular el coeficiente de determinación r²
    # Primero calcular ȳ = (1/(b-a)) ∫ f(x) dx
    integral_f = scin.quad(funcion, a, b)[0]
    y_media = integral_f / (b - a)
    
    # Calcular S²ᵧ = ∫ (f(x) - ȳ)² dx
    varianza_f = lambda x: (funcion(x) - y_media)**2
    Sy2 = scin.quad(varianza_f, a, b)[0]
    
    # Calcular r² = 1 - Ec/S²ᵧ
    r2 = 1 - Ec / Sy2
    
    return c, Ecm, r2


def aproxmc1d(base, x, y):
    """
    Plantea y resuelve las ecuaciones normales para un ajuste por mínimos cuadrados 
    discreto en un conjunto de puntos dados.
    
    El método encuentra los coeficientes c = [c₁, c₂, ..., cₙ] que minimizan el 
    error cuadrático entre los valores y y su aproximación g(x) = Σ cᵢ φᵢ(x).
    
    Parámetros:
    -----------
    base : list
        Lista con las funciones de la base B₁ = {φ₁(x), ..., φₙ(x)}
        Cada elemento debe ser una función callable.
    x : array-like
        Puntos del soporte (coordenadas x)
    y : array-like
        Valores en los puntos (coordenadas y)
    
    Retorna:
    --------
    c : ndarray
        Vector con los coeficientes en la base
    Ecm : float
        Error cuadrático medio
    r2 : float
        Coeficiente de determinación
    
    Algoritmo:
    ----------
    1. Construye la matriz de Gram G donde Gᵢⱼ = Σ φᵢ(xₖ)·φⱼ(xₖ)
    2. Construye el vector F donde Fᵢ = Σ yₖ·φᵢ(xₖ)
    3. Resuelve el sistema Gc = F
    4. Calcula Ec = Σ yᵢ² - c^T·F
    5. Calcula Ecm = Ec/N
    6. Calcula r² = 1 - Ec/S²ᵧ donde S²ᵧ = Σ(yᵢ-ȳ)²
    """
    
    # Convertir a arrays de numpy
    x = np.array(x)
    y = np.array(y)
    N = len(x)
    n = len(base)
    
    # Validar que x e y tengan la misma longitud
    if len(x) != len(y):
        raise ValueError("Los vectores x e y deben tener la misma longitud")
    
    # Inicializar matriz de Gram y vector de productos internos
    G = np.zeros((n, n))
    F = np.zeros(n)
    
    # Construir la matriz de Gram G y el vector F
    for i in range(n):
        # Calcular Fᵢ = Σ yₖ·φᵢ(xₖ)
        phi_i_vals = np.array([base[i](xk) for xk in x])
        F[i] = np.sum(y * phi_i_vals)
        
        # Calcular Gᵢⱼ = Σ φᵢ(xₖ)·φⱼ(xₖ)
        for j in range(n):
            phi_j_vals = np.array([base[j](xk) for xk in x])
            G[i, j] = np.sum(phi_i_vals * phi_j_vals)

    # Resolver el sistema de ecuaciones normales Gc = F
    c = la.solve(G, F)
    
    # Calcular el error cuadrático: Ec = Σ yᵢ² - c^T·F
    sum_y2 = np.sum(y**2)
    Ec = sum_y2 - np.dot(c, F)
    
    # Calcular el error cuadrático medio: Ecm = Ec/N
    Ecm = Ec / N
    
    # Calcular el coeficiente de determinación r²
    # Primero calcular ȳ = (1/N) Σ yᵢ
    y_media = np.mean(y)
    
    # Calcular S²ᵧ = Σ (yᵢ - ȳ)²
    Sy2 = np.sum((y - y_media)**2)
    
    # Calcular r² = 1 - Ec/S²ᵧ
    r2 = 1 - Ec / Sy2
    
    return c, Ecm, r2




def aproxmc1d_eval(base, coef, z):
    """
    Evalúa la función de aproximación por mínimos cuadrados en puntos dados.
    
    Dada una base B = {φ₁(x), ..., φₙ(x)} y coeficientes {c₁, ..., cₙ}, 
    calcula ψ(z) = Σ cᵢ·φᵢ(z) para cada punto z.
    
    Esta función es válida tanto para aproximaciones continuas como discretas,
    ya que la evaluación de la función aproximada es independiente del método
    usado para obtener los coeficientes.
    
    Parámetros:
    -----------
    base : list
        Lista con las funciones de la base B = {φ₁(x), ..., φₙ(x)}
        Cada elemento debe ser una función callable.
    coef : array-like
        Coeficientes obtenidos por mínimos cuadrados [c₁, c₂, ..., cₙ]
    z : array-like o float
        Punto(s) donde se evalúa la función de aproximación.
        Puede ser un único valor o un array de valores.
    
    Retorna:
    --------
    psi_z : ndarray o float
        Valor(es) de ψ(z) = Σ cᵢ·φᵢ(z)
        Si z es escalar, retorna un escalar.
        Si z es array, retorna un array con los valores evaluados.
    
    Fórmula:
    --------
    ψ(x) = c₁·φ₁(x) + c₂·φ₂(x) + ... + cₙ·φₙ(x)
    
    Ejemplo:
    --------
    Para base B = {1, x, x²} con coeficientes c = [1, -2, 3]:
    ψ(x) = 1·1 + (-2)·x + 3·x² = 1 - 2x + 3x²
    """
    
    # Convertir coeficientes a array de numpy
    coef = np.array(coef)
    n = len(base)
    
    # Validar que el número de coeficientes coincida con el número de funciones base
    if len(coef) != n:
        raise ValueError(f"El número de coeficientes ({len(coef)}) debe coincidir "
                        f"con el número de funciones base ({n})")
    
    # Determinar si z es escalar o array
    es_escalar = np.isscalar(z)
    z_array = np.atleast_1d(z)
    
    # Inicializar resultado
    psi_z = np.zeros(len(z_array))
    
    # Calcular ψ(z) = Σ cᵢ·φᵢ(z) para cada punto z
    for i in range(n):
        # Evaluar φᵢ(z) para todos los puntos z
        phi_i_z = np.array([base[i](zk) for zk in z_array])
        # Sumar la contribución cᵢ·φᵢ(z)
        psi_z += coef[i] * phi_i_z
    
    # Retornar escalar si la entrada fue escalar
    if es_escalar:
        return psi_z[0]
    else:
        return psi_z


def poly_optimo(x, y):
    """
    Encuentra el polinomio de menor error estándar que ajusta una nube de puntos.
    
    La función prueba polinomios de grado creciente (desde 0 hasta N-1, donde N es el 
    número de puntos) y selecciona aquel que minimiza el error cuadrático medio (Ecm).
    
    El error estándar se calcula como la raíz cuadrada del error cuadrático medio:
    σ = √(Ecm) = √(Σ(yᵢ - ψ(xᵢ))² / N)
    
    Parámetros:
    -----------
    x : array-like
        Puntos del soporte (coordenadas x)
    y : array-like
        Valores en los puntos (coordenadas y)
    
    Retorna:
    --------
    poly_opt : numpy.poly1d
        Polinomio óptimo que minimiza el error estándar
    errores : ndarray
        Vector con todos los errores cuadráticos medios calculados para cada grado
        errores[i] corresponde al Ecm del polinomio de grado i
    
    Algoritmo:
    ----------
    1. Para cada grado k desde 0 hasta N-1:
       a. Construir la base B = {1, x, x², ..., xᵏ}
       b. Calcular coeficientes usando mínimos cuadrados discretos
       c. Calcular el error cuadrático medio Ecm
       d. Guardar el error
    2. Seleccionar el grado con menor Ecm
    3. Retornar el polinomio óptimo y todos los errores calculados
    
    Notas:
    ------
    - El grado máximo probado es N-1 (número de puntos - 1)
    - Un polinomio de grado N-1 interpola exactamente N puntos
    - El error estándar σ = √Ecm mide la desviación típica del ajuste
    """
    
    # Convertir a arrays de numpy
    x = np.array(x)
    y = np.array(y)
    N = len(x)
    
    # Validar que x e y tengan la misma longitud
    if len(x) != len(y):
        raise ValueError("Los vectores x e y deben tener la misma longitud")
    
    # Inicializar vector de errores
    errores = np.zeros(N)
    
    # Almacenar coeficientes de cada polinomio
    coeficientes = []
    
    print(f"Probando polinomios de grado 0 a {N-1}...\n")
    
    # Probar todos los grados posibles desde 0 hasta N-1
    for grado in range(N):
        # Construir la base polinomial: {1, x, x², ..., xᵍʳᵃᵈᵒ}
        base = [lambda xi, k=k: xi**k for k in range(grado + 1)]
        
        # Calcular aproximación por mínimos cuadrados
        try:
            c, Ecm, r2 = aproxmc1d(base, x, y)
            errores[grado] = Ecm
            coeficientes.append(c)
            
            # Calcular error estándar
            error_estandar = np.sqrt(Ecm)
            
            print(f"Grado {grado}: Ecm = {Ecm:.6e}, Error estándar σ = {error_estandar:.6e}, r² = {r2:.6f}")
            
        except Exception as e:
            print(f"Grado {grado}: Error en el cálculo - {e}")
            errores[grado] = np.inf
            coeficientes.append(None)
    
    # Encontrar el grado óptimo (menor Ecm)
    grado_optimo = np.argmin(errores)
    Ecm_optimo = errores[grado_optimo]
    c_optimo = coeficientes[grado_optimo]
    
    print(f"\n{'='*70}")
    print(f"Grado óptimo: {grado_optimo}")
    print(f"Error cuadrático medio mínimo: {Ecm_optimo:.6e}")
    print(f"Error estándar mínimo: {np.sqrt(Ecm_optimo):.6e}")
    print(f"{'='*70}\n")
    
    # Construir el polinomio óptimo usando numpy.poly1d
    # Los coeficientes deben estar en orden descendente para poly1d
    # c_optimo está en orden [c₀, c₁, c₂, ...] → invertir para poly1d
    coef_poly1d = c_optimo[::-1]
    poly_opt = np.poly1d(coef_poly1d)
    
    return poly_opt, errores

