from math import gcd
from math import factorial
import numpy as np
import copy
import numpy.polynomial as P
import scipy . interpolate as scip
import sympy as sp
import copy
import scipy.integrate as scin
import numpy.linalg as la
import scipy.optimize as scop
import scipy.linalg as sla
from random import randint


#PRACTICA 1

def euclides(D,d):
    '''
    Esta función calcula el máximo común divisor de dos enteros dados.

    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    D: primer número entero 
    d: segundo número entero 
    '''
    

    #Nos aseguramos de que realmente los parámetros que nos han sido proporcionados
    #son números enteros

    assert type(D)==int, 'Debe introducir un número entero'
    assert type(d)==int, 'Debe introducir un número entero'

    #Calculamos el máximo común divisor y lo guardamos en la variable c

    c=mcd = gcd (D,d)

    return c

def raiznsima(a,n,niter,tol):
    '''
    Esta función calcula la raíz enésima de un número 'a' dado.

    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    a: número del cual se quiere calcular la raíz enésima
    n: índice de la raíz
    niter: número de iteraciones que se desea realizar
    tol: error máximo permitido
    '''
    x = a
    for i in range(niter): 
        xkm1 = x*(1-1/n)+a/(n*x**(n-1))
        if abs(x-xkm1)<tol: 
            break #finaliza el bucle entero, si fuera continue simplemente pasaría a la siguiente iteración
        x = xkm1
    return x

#PRÁCTICA 2

def exponencial(x,n):
    '''
    Esta función evalúa el desarrollo de Taylor de la función exponencial
    utilizando dos fórmulas distintas.
    
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
    x: punto donde queremos evaluar la función
    n: número de sumandos que debe tener el desarrollo
    '''
    
    #Primera fórmula
    t1=0
    f1=[]

    for k in range(n+1):
        xk=(x**k)/factorial(k)
        tk=t1+xk
        f1.append(tk)
        t1=tk
        
    #Segunda fórmula
    t2=0
    f2=[]
    for k in range(n+1):
        xk=(((-1)*x)**k)/factorial(k)
        tk=t2+xk
        tk=1/tk
        f2.append(tk)
        t2=tk

    return f1,f2

#PRÁCTICA 4
def base_lagrange(x):
    '''
    Devuelve una lista con los polinomios de la base de Lagrange ordenados para el soporte x.

    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    x: lista de puntos del soporte
    '''
    
    # El soporte x debe ser una lista
    #isinstance(objeto,tipo)
    
    if isinstance(x,np.ndarray):
        x = list(x)
    elif isinstance(x,tuple):
        x = list(x)
    elif isinstance(x,list):
        pass
    else:
        raise ValueError('El parámetro de la función tiene que ser una lista, una tupla, o un array de Numpy')
    
    
    #Ahora creamos una lista donde iremos añadiendo los polinomios de Lagrange
    L_k = []

    # Cálculamos los polinomios de la base de Lagrange
    
    for k in range(len(x)):
        x2 = copy.deepcopy(x)              # Empezamos creando una copia independiente de x
        x2.pop(k)                          # Elimino el elemento de índice k
        pol = P.Polynomial.fromroots(x2)   # Creo un polinomio que tenga ceros en los puntos x2
        pol_lagrange = pol/pol(x[k])       # Calculo el polinomio de Lagrange 
        
        L_k.append(pol_lagrange)

    return L_k


def itp_Tchebishev(fun,npts,a,b):
    '''
    Devuelve el polinomio de interpolación de la función numérica (fun) en un intervalo ([a,b]) 
    utilizando varios puntos (npts) elegidos para minorar el error del interpolante.
    PARÁMETROS DE LA FUNCIÓN
    ------------------------
     fun: función a interpolar
     a,b: extremos inferior y superior, respectivamente, 
     del intervalo en el que se quiere interpolar
     npts: número de puntos del soporte 
    '''
    #Calculo los puntos del soporte que vamos a utilizar, es decir, las raíces de los polinomios de Tchevishev
    gi_k = np.cos((2*np.arange(npts)+1)/(npts*np.pi()/2))
    
    #Como nos encontramos en un intervalo [a,b] cualquiera, y no en el intervalo [-1,1]
    xk=(a+b)/2 + gi_k*(b-a)/2
    
    #Evaluamos los puntos del soporte en la función que queremos interpolar
    yk=fun(xk)

    # Función interpolante
    pol_t = scip.BarycentricInterpolator(xk,yk)

    return pol_t


# PRÁCTICA 5

def dncoef_base(soporte, puntos, orden):
    '''
    Esta función calcula los coeficientes de una regla de derivación
    numérica y devuelve una lista con ellos.
    
    PARÁMETROS DE LA FUNCIÓN
    ----------
    soporte : soporte de los polinomios de la base de Lagrange.
    puntos : puntos en los cuales queremos saber los coeficientes de 
            la interpolación numérica.
    orden : orden de la derivada que queremos calcular.


    '''
    
    #Comprobación de los datos de entrada
    if not (isinstance(orden, int)) and orden>0:
        raise ValueError('El orden de la derivada debe ser un número entero positivo')
        
    #Calculamos los polinomios de la base de Lagrange para ese soporte
    Pol_Lg=base_lagrange(soporte)
    
    #Ahora vamos a derivar cada uno de los polinomios de la base
    #Creo una lista donde iré añadiendo las derivadas de los polinomios
    Derivadas=[]
    
    for L in Pol_Lg:
        dif = L.deriv(orden)
        Derivadas.append(dif)
    
    Derivadas=np.array(Derivadas)
    #También se podría hacer Derivadas=[L.deriv(orden) for L in Pol_Lg]
    
    
    #Ahora evaluamos las derivadas en cada uno de los puntos
    
    if isinstance(puntos, ((int, float))):
        coef = Derivadas(puntos)
        
    elif isinstance(puntos, ((list, tuple, np.ndarray))):
        
        #Creamos una lista donde se irán añadiendo los coeficientes
        coef_0=[]
        for x in puntos:
            coef_0=[d(x) for d in Derivadas]

        coef = [round(float(x), 2) for x in coef_0]

    return coef

#PRÁCTICA 6

def deriva2(fun, puntos, h):

    '''
    Esta función calcula la derivada segunda de una función “fun” en “puntos”
    utilizando las reglas numéricas indicadas con el valor “h”. Devuelve una lista 
    con los resultados de las fórmulas para cada uno de los puntos.

    PARÁMETROS DE LA FUNCIÓN
    -------------------------
    fun: función de la cual se quiere calcular la derivada segunda
    puntos: puntos en los que se quiere calcular la derivada
    h: tamaño del intervalo entre un punto y su consecutivo

    '''
    if isinstance(puntos, (list,tuple, np.ndarray)) and isinstance(h,(int,float)):

        #Creamos una lista donde añadiremos listas con los resultados de cada método 
        #para todos los puntos: der2=[[método 1],[método 2],[método 3],[método 4]]
        der2=[]

        #Creamos una lista vacía para cada método
        m1=[]
        m2=[]
        m3=[]
        m4=[]
        
        for x in puntos:
           f1=(fun(x) - 2*fun(x+h) + fun(x+2*h))/(h**2)
           m1.append(f1)
           f2=(fun(x-h) - 2*fun(x) + fun(x+h))/(h**2)
           m2.append(f2)
           f3=(2*fun(x) - 5*fun(x+h) + 4*fun(x+2*h) - fun(x+3*h))/(h**2)
           m3.append(f3)
           f4=(-2*fun(x-2*h) + 16*fun(x-h) + 30*fun(x) + 16*fun(x+h) - fun(x+2*h))/(h**2)
           m4.append(f4)
        
        der2.append(m1)
        der2.append(m2)
        der2.append(m3)
        der2.append(m4)

    elif isinstance(puntos, (int,float)) and isinstance(h,(list,tuple, np.ndarray)):

        #Creamos una lista donde añadiremos listas con los resultados de cada método 
        #para todos los posibles valores de h: der2=[[método 1],[método 2],[método 3],[método 4]]
        der2=[]

        #Creamos una lista vacía para cada método
        m1=[]
        m2=[]
        m3=[]
        m4=[]
        
        for i in h:
           h1=(fun(puntos) - 2*fun(puntos+i) + fun(puntos+2*i))/(i**2)
           m1.append(h1)
           h2=(fun(puntos-i) - 2*fun(puntos) + fun(puntos+i))/(i**2)
           m2.append(h2)
           h3=(2*fun(puntos) - 5*fun(puntos+i) + 4*fun(puntos+2*i) - fun(puntos+3*i))/(i**2)
           m3.append(h3)
           h4=(-2*fun(puntos-2*i) + 16*fun(puntos-i) + 30*fun(puntos) + 16*fun(puntos+i) - fun(puntos+2*i))/(i**2)
           m4.append(h4)
        
        der2.append(m1)
        der2.append(m2)
        der2.append(m3)
        der2.append(m4)

    elif isinstance(puntos, (int,float)) and isinstance(h,(int,float)):

        #Creamos una lista donde añadiremos listas con los resultados de cada método 
        #der2=[método 1,método 2,método 3,método 4]
        der2=[]

        f1=(fun(puntos) - 2*fun(puntos+h) + fun(puntos+2*h))/(h**2)
        der2.append(f1)
        f2=(fun(puntos-h) - 2*fun(puntos) + fun(puntos+h))/(h**2)
        der2.append(f2)
        f3=(2*fun(puntos) - 5*fun(puntos+h) + 4*fun(puntos+2*h) - fun(puntos+3*h))/(h**2)
        der2.append(f3)
        f4=(-2*fun(puntos-2*h) + 16*fun(puntos-h) + 30*fun(puntos) + 16*fun(puntos+h) - fun(puntos+2*h))/(h**2)
        der2.append(f4)

    else:
        raise ValueError('Puntos y h no pueden ser vectores simultáneamente')
    
    der2 = [[round(float(x), 2) for x in lista] for lista in der2]
    return der2


def incoef_base(soporte,a,b):

    '''
    Esta función calcula los coeficientes de una regla de integración
    numérica para calcular la integral entre a y b usando los polinomios de la 
    base de Lagrange definidos por soporte. Devuelve una lista con estos coeficientes.
    
    PARÁMETROS DE LA FUNCIÓN
    ----------
    soporte : soporte de los polinomios de la base de Lagrange.
    a,b : extremos inicial y final, respectivamente, del intervalo en el que queremos 
    calcular la integral.
    '''
        
    #Calculamos los polinomios de la base de Lagrange para ese soporte
    Pol_Lg=base_lagrange(soporte)
    
    #Ahora vamos a derivar cada uno de los polinomios de la base
    #Creo una lista donde iré añadiendo las derivadas de los polinomios
    Integrales=[]

    for L in Pol_Lg:
        int_a_x = L.integ(lbnd=a)
        int_a_b= int_a_x(b)
        Integrales.append(int_a_b)

    return Integrales

def in_romberg(fun,a,b,nivel=10,tol=1e-6):

    '''
    Esta función calcula la integral definida de una función utilizando el método
    de Romberg, hasta alcanzar una tolerancia dada (10-6, por defecto) y sin exceder 
    el nivel máximo de subdivisiones (10 por defecto). Devuelve el valor estimado de la integral, 
    valor estimado del error y la tabla de aproximaciones. 

    PARÁMETROS DE LA FUNCIÓN:
    --------------------------
    fun: función de la cual queremos calcular su integral
    a,b : extremos inicial y final, respectivamente, del intervalo en el que queremos 
    calcular la integral.
    nivel: nivel máximo de subdivisiones
    tol: error máximo permitido
    '''
    #Creamos una matriz cuadrada de orden 'nivel'
    N = np.zeros((nivel, nivel))

    #Definimos el equiespaciado h
    h=b-a

    #Asignamos el siguiente valor al elemento (0,0) de la matriz
    N[0][0]=(fun(a)+fun(b))*(h/2)
    p=1

    #Para n desde 1 hasta 'nivel'-1
    for n in range(1, nivel):
        h=h/2

        #Calculamos los p puntos equiespaciados por la regla del trapecio compuesta
        puntos = [a+(2*i-1)*h for i in range(1,p+1)]

        N[n][0]=0.5*N[n-1][0]+h*sum([fun(i) for i in puntos])
        q=1

        for j in range(1,n+1):
            q=q*4
            delta=(1/(q-1))*(N[n][j-1]-N[n-1][j-1])
            N[n][j]=N[n][j-1]+delta
        
        if abs(delta)<tol:
            return N[n][n],delta, N
        else:
            p=2*p

def paracaidista(y0,v0,m,cx,At,apertura=1500,rovar=False):
    '''
    Esta función determina el tiempo y la velocidad a la que toma tierra
    un paracaidista con las condiciones determinadas en los parámetros 
    
    PARÁMETROS DE LA FUNCIÓN:
    ------------------------
    y0,v0: posición y velocidad inicial del salto en m y m/s respectivamente
    m: masa del paracaidista equipado en kg
    cx: iterable con los coeficientes de arrastre del paracaidista antes 
    y después de la apertura del paracaídas
    At:iterable con el área transversal del paracaidista en m^2 antes y después de la apertura del paracaídas
    apertura: altura a la que se abre el paracaídas en m, tiene un valor por defecto de 1500 m
    rovar: valor lógico que indica si la densidad es variable o no. Por defecto es falso

    RESULTADO:
    ----------
    lista con cuatro valores: [velocidad máxima, velocidad de impacto, tiempo hasta que abre el paracaídas,
    tiempo total de vuelo]
    '''

    #Definimos el valor de la gravedad, un valor orientativo para el tiempo final, y una lista con los valores iniciales de las variables
    g=9.81
    t_fin1=100
    t_fin2=500
    ci=[y0,v0]

    #Antes de la apertura del paracaídas la SEDO a resolver es:

    def sedo(t ,Y ,m ,cx ,At , rovar ):
        ro = 1.225
        if rovar==True : ro=1.225*np.exp(-Y[0]/8243)
        kw = cx * ro * At /2
        dY = np . array ([ Y [1] , - g - kw *Y [1]* abs (Y [1]) /m ])
        return dY
    
    #Tenemos que resolver dos problemas de valor inicial. Cada uno con unos valores iniciales y un evento diferente

    #Empezamos definiendo el evento de que se abre el paracaídas
    def abreParaca (t ,Y ,m ,cx ,At , rovar ) : return Y[0] - apertura
    abreParaca . terminal = True
    abreParaca . direction = -1

    sol1 = scin . solve_ivp ( sedo ,[0,t_fin1],ci, args =[m ,cx[0] , At[0] , rovar ], events = abreParaca)

    #Ahora definimos el evento de que impacte en el suelo
    def impactoSuelo (t ,Y ,m , cx ,At , rovar ): return Y[0]
    impactoSuelo . terminal = True
    impactoSuelo . direction = -1

    #Ahora ya podemos resolver la segunda parte del recorrido, teniendo en cuenta los cambios en los coeficientes y 
    #que este segundo tramo toma como valores iniciales del primero
    sol2 = scin . solve_ivp ( sedo , [sol1.t[-1],t_fin2],[sol1.y[0][-1], sol1.y[1][-1]], args =[m ,cx[1] , At[1] , rovar ], events = impactoSuelo)

    v_max = max(np.max(np.abs(sol1.y[1])), np.max(np.abs(sol2.y[1])))
    v_impacto=sol2.y[1][-1]
    t_paraca=sol1.t[-1]
    t_impacto=sol2.t[-1]

    return [v_max, v_impacto, t_paraca, t_impacto]


#PRÁCTICA 8

def disparo(F,ab,cc,mi=[0,1],niter=100, xtol=1e-6, ftol=1e-9,**opt):
    '''
    Esta función resuelve un problema de contorno transformándolo en uno de valor inicial.
    
    PARÁMETROS DE LA FUNCIÓN:
    -------------------------
    F : función (vectorial) que define el SEDO
    ab : intervalo de resolución del problema de contorno. Lista con los extremos inicial y final
    cc : condiciones de contorno (Dirichlet) en los extremos por orden cc=[cc_a, cc_b]
    mi : valores de la pendiente en las dos primeras iteraciones (valor por defecto:[0,1])
    niter: número máximo de iteraciones (valor por defecto: 100)
    xtol: error admisible en la pendiente (valor por defecto: 1E-6)
    ftol : error admisible en la solución (valor por defecto: 1E-9)
    opt: diccionario con las opciones para solve_ivp (method, dense_output, events).    

    RESULTADOS:
    -----------
    S: estructura de solve_ivp        
    '''

    opc = {"method":"RK45", "dense_output": False , "events": None }
    keys = list ( opt )
    for clave , valor in opt . items () :
        clave = clave . lower ()
        if clave in keys :
            opt [ clave ] = valor
        else :
            print (" Opt {} inválida ". format ( clave ))

    #Resolvemos la SEDO con mi[0]. Sabemos que el primer valor inicial es cc[0], y suponemos que el 
    #segundo, el de la derivada evaluada en el 0, es mi[0]
    
    s1=scin.solve_ivp(F,ab,[cc[0],mi[0]],method=opc['method'], dense_output=opc['dense_output'], events=opc['events'])

    #El segundo argumento que nos devuelve s1.y son las derivadas en los puntos del intervalo usados
    #Así que la derivada en el extremo final b será s1.y[1][-1], y tendremos que ver más adelante si este coincide con cc_b

    w_m1=s1.y[0][-1]

    #Repetimos este mismo procedimiento con mi[1]
    s2=scin.solve_ivp(F,ab,[cc[0],mi[1]],method=opc['method'], dense_output=opc['dense_output'], events=opc['events'])
    w_m2=s2.y[0][-1]

    #Ahora utilizamos un bucle para las siguiuentes pendientes
    m_k=mi[0]
    m_k1=mi[1]
    w_k = w_m1
    w_k1 = w_m2


    for k in range(1,niter+1):
       m_k2=m_k1+(cc[1]-w_k1)*((m_k1-m_k)/(w_k1-w_k))
       s_k2=scin.solve_ivp(F,ab,[cc[0],m_k2],method=opc['method'], dense_output=opc['dense_output'], events=opc['events'])
       w_k2=s_k2.y[0][-1]

       if abs(m_k2-m_k1)< xtol or abs(w_k2-cc[1])< ftol: break

       m_k, m_k1 = m_k1, m_k2
       w_k, w_k1 = w_k1, w_k2
    
    return s_k2


#PRÁCTICA 9

def enlsolver(funx, a, b, meth= "rf", maxiter=128 , tol=[10e-9, 10e-6, 10e-12]):
    '''
    Esta función halla las raíces de una función dada por alguno de los métodos de intervalo admisibles

    PARÁMETROS DE LA FUNCIÓN:
    ------------------------
    funx: función cuyo cero se desea determinar
    a,b: extremos del intervalo
    meth: método de resolución: 'di' (dicotomía), 'rf' (Regula Falsi) ó 'fm' (R.F. modificada)
    maxiter: número máximo de iteraciones
    tol: lista que contiene la tolerancia absoluta y relativa del intervalo y de la función respectivamente; 
    [Ex, ex, EF]

    RESULTADOS DE LA FUNCIÓN:
    ------------------------
    r: cero aproximado
    info: motivo por el que finalizó la función: -2 (Método desconocido), -1 (no verifica Bolzano) , 
    0 (tolX alcanzada), 1 (tolFun alcanzada), 2 (maxiter alcanzada)
    
    '''
    #Creamos una lista donde se irán añadiendo los valores de la secuencia x_n
    suc=[]

    #Definimos los errores
    E_x=tol[0]
    e_x=tol[1]
    E_F=tol[2]

    

    if meth !='di ' and meth!= 'rf' and meth!= 'fm':
        r=None
        info=-2
        return r, info, np.array(suc)


        
    #Calculamos las imágenes de los extremos del intervalo
    fa=funx(a)
    fb=funx(b)
    if meth=='fm': fxn=0

    #Comprobamos que se cumple bolzano y que no se haya alcanzado ya la tolerancia
    if abs(fa)<E_F: 
        r=a
        info=1
        suc.append(a)
        return r, info, np.array(suc)
    
    if abs(fb)<E_F: 
        r=b
        info=1
        suc.append(b)
        return r, info, np.array(suc)
    
    if fa*fb>0: 
        r=None
        info=-1
        print('Error, no verifica Bolzano')
        return r, info, np.array(suc)
    
    for n in range(1,maxiter+1):
        #Elegimos un valor x_n dentro del intervalo  [a_n,b_n]


        if meth=='di':
            x_n= 0.5*(a+b)

        else: 
            x_n=(a*fb-b*fa)/(fb-fa)

        suc.append(x_n)

        fx= funx(x_n)

        if fx*fa>0:
            fa=fx
            a=x_n

            if meth=='fm' and fx*fxn>0:
                fb=0.5*fb

        else:
            fb=fx
            b=x_n

            if meth=='fm' and fx*fxn>0:
                fa=0.5*fa

        if meth=='fm': fxn=fx

        xtol=max(E_x, abs(x_n)*e_x)

        if abs(b-a)<xtol:
            r=x_n
            info=0

            return r, info, np.array(suc)

    r=suc[maxiter-1]
    info=2

    return r, info, np.array(suc)


#PRÁCTICA 10
def enlsteffensen(funx, x0, maxiter=128, tol=[10e-9, 10e-5,10e-12]):
    '''
    Esta función calcula las raíces de una función dada por el método de Steffensen

    PARÁMETROS DE LA FUNCIÓN:
    -------------------------
    funx: función numérica cuyo cero se quiere aproximar
    x0: punto inicial
    maxiter: númeo máximo de iteraciones
    tol: lista con las tolerancias absoluta y relativa del punto y de la función 

    RESULTADOS DE LA FUNCIÓN:
    ------------------------
    r: cero aproximado de la función
    info: motivo por el que finalizó: 0(xtol alcanzada), 1(rtol alcanzada), 2(maxiter alcanzado)
    suc: sucesión de valores {x_n} 

    '''

    #Asignamos los valores de las tolerancias
    E_x=tol[0]
    e_x=tol[1]
    E_F=tol[2]


    #Creamos un alista en la que iremos añadiendo los valores de x
    X=[x0]

    for k in range(1,maxiter+1):
        fx=funx(X[k-1])

        if abs(fx)<E_F:
            r=X[k-1]
            info=1
            suc=np.array(X)

            return r,info,suc

        fy=funx(X[k-1]+fx)
        gx=(fy/fx)-1

        if gx<-2 or gx>1: print('Aviso, probable divergencia')

        x_k=X[k-1]-(fx/gx)
        X.append(x_k)

        epsilon=max(E_x, e_x*abs(X[k]))

        if abs(X[k]-X[k-1]) < epsilon: 
            r=X[k]
            info=0
            suc=np.array(X)
            return r,info,suc


    #Si acaba el bucle sin convergencia
    r=X[maxiter]
    info=2
    suc=np.array(X)


    return r,info,suc


#PRÁCTICA 11 
def sor_interval(A):
    '''
    Esta función calcula, para una matriz dada, el intervalo en que se puede aplicar
    el método SOR y sus valores óptimos.

    PARÁMETROS DE LA FUNCIÓN:
    ------------------------
    A: matriz a estudiar

    RESULTADOS:
    ----------
    inter: intervalo [wi,wf] donde el método es convergente. Si no lo hay devuelve una lista vacía
    ropt: valor del menor radio espectral, ro_min. Si no lo hay devuelve None
    wopt: valor del intervalo que corresponde a ro_min, redondeado a la centésima. Si no existe devuelve None
    
    '''
    #Descomposición de la matriz A=D-L-U
    D = np . diag ( np . diag (A))
    L = -np . tril (A , -1)
    U = -np . triu (A ,1)

    #Creamos un vector con valores equiespaciados en el intervalo [0,2]
    W=np.arange(0,2.01,0.01)

    #Creamos una lista para almacenar los valores de ro y w válidos
    ro_v=[]
    w_v=[]

    for w in W:

        #Calculamos la matriz SOR 
        Mi = la.inv (D -w*L) # Matriz inversa
        Bs = np.dot (Mi ,(1 - w)*D+ w*U)

        #Radio espectral
        rs = max(abs ( eigvals ( Bs )))

        #Si es un valor válido lo añadimos a la lista 
        if rs<1: 
            ro_v.append(rs)
            w_v.append(w)

    #Ahora encontramos el valor mínimo de esa lista

    ro_min= min(ro_v)

    #Recorremos esa lista para ver cuál es el índice de ese elemento

    for index,r in enumerate(ro_v):
        if r==ro_min:
            indice=index
            break

    w_ro_min=round(w_v[indice],2)
    

    if len(w_v)<2:
        inter=[]
        ropt=None
        wopt=None
    
    else:
        inter=np.array([round(w_v[0],2), round(w_v[-1],2)])
        ropt=ro_min
        wopt=w_ro_min

    return inter.tolist(), ropt, wopt
        
#PRÁCTICA 12
def autoval_potencia(A,w,tol,niter):
    '''
    Esta función calcula, para una matriz dada, su autovalor más próximo al indicado.

    PARÁMETROS DE LA FUNCIÓN:
    ------------------------
    A: matriz a estudiar
    w: referencia del autovalor. ±∞ para el mayor en valor absoluto, 0 para el menor no nulo, y un valor cualquiera
    en otro caso.
    tol: tolerancia del error. Si es un valor, se mide con la norma ∥◦∥2 , sino debe ser un par de valores, donde el segundo
    indica la norma usada en la tolerancia.
    niter: numero máximo de iteraciones del proceso.

    RESULTADOS:
    ----------
    valor: valor propio aproximado
    vector: vector propio aproximado
    '''

    if type(tol)==float or type(tol)==int:
        epsilon=tol
        tipo_norma= None

    else:
        epsilon=tol[0]
        tipo_norma= tol[1]


    #Empezamos calculando el orden n, es decir, el rango, de la matriz A
    n=la.matrix_rank(A)

    #Ahora generamos un vector aleatorio de n elementos y lo normalizamos
    a=randint(1, 50)
    b= randint(1, 50)

    while b==a:
        b= randint(1, 50)

    w_0=np.linspace(a,b,n)
    w_0_norm= w_0/la.norm(w_0)

    #Si queremos saber el mayor autovalor en valor absoluto usamos el método de la potencia

    u_0 = w_0.copy()

    if w==np.inf or w== -np.inf:
        for k in range(1,niter+1):
            u_k=A@w_0
            w_k= u_k/la.norm(u_k)

            if la.norm(u_k-u_0, tipo_norma) < epsilon:
                valor_p= np.transpose(w_k)@A@w_k
                vect_p= w_k

                return valor_p, vect_p 

#PRÁCTICA 13
def  aproxmc1c(base,ab,funcion):
    '''
    Esta función plantea y resuelve las ecuaciones normales para un ajuste por 
    mínimos cuadrados continuo.

    PARÁMETROS:
    -----------
    base: lista con las funciones que conforman la base
    ab: intervalo de la aproximación en forma de lista
    funcion: función a aproximar

    RESULTADOS:
    -----------
    coef: lista o vector con los coeficientes de la base

    '''
    #Para hallar los coeficientes de los elementos de la base tenemos que resolver el sistema 
    #Gc=f; donde G es la matriz de Gram, c es una matriz columna que contiene los coeficientes
    #y f es una matriz columna cuyos elementos son de la forma <f,phi_i(x)> 
    #con phi_i los elementos de la base

    #Empezamos hallando la dimensión de la base, que tendrá que cioncidir con el rango de la función de Gram
    dim_b=len(base)

    #Creamos una matriz que nos servirá como soporte para crear la matriz de Gram
    gram=np.zeros(shape=(dim_b,dim_b))

    #Ahora vamos a construir y colocar los elementos de la matriz de Gram
    #Vamos a 
    for i in range(0,dim_b):
        for j in range(0,dim_b):
            #Para integrar necesitamos UNA sola función numérica, no un producto de dos
            func=lambda x: base[i](x)*base[j](x)

            elemento=scin.quad(func,ab[0],ab[1])[0]

            gram[i][j]=elemento
            gram[j][i]=elemento


    #El siguiente paso es construir la matriz f
    f=np.zeros(dim_b)
    for i in range(0,dim_b):
        #Para integrar necesitamos UNA sola función numérica, no un producto de dos
        func=lambda x: base[i](x)*funcion(x)

        elemento=scin.quad(func,ab[0],ab[1])[0]

        f[i]=elemento

    #Una vez definidas todas las matrices necesarias, solo falta resolver el sistema
    coef=sla.solve(gram,f,assume_a='sym')

    return coef.tolist()


def aproxmc1d(base,x,y):
    '''
    
    
    '''




    





























