
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
