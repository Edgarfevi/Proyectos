# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:21:24 2025

@author: cesarm

Definicion paramétrica de curvas habituales en 2D (XZ) y 3D (XYZ)
"""
import numbers
import numpy as np
import sympy as sp
import scipy as sc
import scipy.integrate as scin
# Para salvar la animación en formato mpeg, es necesario instalar el paquete
# conda install -c conda-forge ffmpeg

import numbers
import types
#
# Generación de curvas típicas en 2D y 3D
"""
    sigmoide: A/(1+np.exp(m*p)). Argumentos A (amplitud) m (pendiente)
        recorrido~A[1-ε,ε], intervalo~[ln(A/ε-1)/m] , ε=0.005
        https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide
    clotoide: A*int(cos(p^2)), A*int(sin(p^2)). Argumentos A, t0 (inicio)
        intervalo~2A, recorrido~[0,A], maximo en x~0.663*A
        https://es.wikipedia.org/wiki/Clotoide
    helicoide: A*cos(p), A*sin(p). Argumentos A
        DD [0,2 pi]  recorrido [-1,1]
        elipse(p,a,b):
            elipse(p,a,b):
                parabola(p,a):
        https://es.wikipedia.org/wiki/H%C3%A9lice_(geometr%C3%ADa)
    gausiana: A exp(-(p/s)^2)
        intervalo~±2.6s, recorrido~[0,0.995 A]
        https://es.wikipedia.org/wiki/Funci%C3%B3n_gaussiana
"""
#
###############################################################################
#    GEOMETRIA
###############################################################################
def curva3ds(tipo,p,A=1,C=[0,0,0],paso=0, plano="xz",args=[]):
    pass

def curva3d(tipo,p,A=1,C=[0,0,0],paso=0, plano="xz",args=[]):
    """
    Evalua la curva indicada que depende del parámetro p  

    Parameters
    ----------
    p : float
        parámetro
    A : float
        Escalado (amplitud) de la curva
    C : terna de float
        origen de coordenadas
    paso : float o list of floats
        paso en la tercera dimensión
    plano : cadena de caracteres
        plano normal al desplazamiento {xy, xz, yz, 2D}. Por defecto "xz"
    args: list of floats
        argumentos para cada tipo de curva

    Returns
    -------
    data : 2/3D array of floats
        coordenadas bi/tridimensionales de los puntos

    """
    plano = plano.lower()
    if isinstance(C,(tuple,list,np.ndarray)):
        if plano=="2d" and len(C)==2:
            if isinstance(C, np.array):
                C = C.tolist()
            elif isinstance(C,tuple):
                C = list(C)
            else:
                pass
            C.append(0)
    else:
        raise ValueError("Origen de coordenadas mal definido")
    if len(C) != 3:
        raise ValueError("Nº de elementos del origen de coordenadas incorrecto")
    C = np.atleast_2d(C).T
    
    if len(args)>0:
        s = args[0]
    else:
        s = None

    tipo = tipo.lower()[0]
    if tipo == "h":
        x,y = elipse(p,A,A)
    elif tipo == "c":
        x,y = clotoide(p,A,s)
    elif tipo == "l": # bucle, looping
        x,y = clotoide(p,A,s)
        xk  = 2*x[-1] - x[::-1] # calculo de la parte simétrica
        yk  = y[::-1]
        x   = np.concatenate((x,xk[1:]))
        y   = np.concatenate((y,yk[1:]))
        # sustituimos el parametro para la dimension z
        p   = np.cumsum(np.abs(np.diff(x)))
        p   = np.array([0,*p])/p[-1]
    elif tipo == "s":
        x,y = sigmoide(p,A,s)
    elif tipo == "g":
        x,y = gausiana(p,A,s)
    else:
        raise ValueError("tipo de curva no contemplada")

    if isinstance(paso,(int,float)):
        z = paso*p
    elif isinstance(paso,(list,tuple,np.ndarray)) and \
         isinstance(p,(list,tuple,np.ndarray)) and \
         len(paso)==len(p):
            z = paso*p
    else:
        raise ValueError("Valor del paso incoherente")

    if plano=="xy":
        data = np.array([x,y,z])
    elif plano=="xz":
        data = np.array([x,z,y])
    elif plano=="yz":
        data = np.array([z,x,y])
    else:
        raise ValueError("Plano de representación no válido")
    data +=  C
    return data

def sigmoide(p,A=1,s=1):
    """
    Genera la función sigmoide. Util para la bajada inicial
    ===> Parametros
    p : parámetro [float]
    A : amplitud (altura) [float]
    s : controla de pendiente [float]
    ===> Resultados
    x,y : coordenadas de los puntos [1D array of floats]
    """
    x = np.array(p)
    y = A/(1+np.exp(s*x))
    return [x,y]

def elipse(p,a,b):
    """
    Genera una elipse de semiejes a y b 
    ===> Parametros
    p : parámetro [float]
    a,b : semiejes [float]
    ===> Resultados
    x,y : coordenadas de los puntos [1D array of floats]
    """
    x = a*np.cos(p)
    y = b*np.sin(p)
    return [x,y]

def hiperbola(p,a,b):
    """
    Genera una hipérbola de semiejes a y b 
    ===> Parametros
    p : parámetro [float]
    a,b : semiejes [float]
    ===> Resultados
    x,y : coordenadas de los puntos [1D array of floats]
    """
    x = a*np.cosh(p)
    y = b*np.sinh(p)
    return [x,y]

def parabola(p,a):
    """
    Genera una parábola de semiejes a
    ===> Parametros
    p : parámetro [float]
    a : semiejes [float]
    ===> Resultados
    x,y : coordenadas de los puntos [1D array of floats]
    """
    x = p
    y = a*p*p
    return [x,y]

def gausiana(p,A,s):
    """
    Genera una campana de Gauss
    ===> Parametros
    p : parámetro [float]
    A : amplitud (altura) [float]
    s : desviación típica [float]
    ===> Resultados
    x,y : coordenadas de los puntos [1D array of floats]
    """
    x = np.array(p)
    y = A*np.exp(-(p/s)**2)
    return [x,y]

def clotoide(p,A=1,p0=0):
    """
    Genera la curva clotoide 
    ===> Parametros
    p : parámetro [float]
    A : amplitud (altura y anchura) [float]
    s : escala en x [float]
    ===> Resultados
    x,y : coordenadas de los puntos [1D array of floats]
    """
    fx= lambda x: np.cos(x**2)
    fy= lambda x: np.sin(x**2)
    if isinstance(p,(int,float)):
        y, ey = scin.quad(fx, p0, p)
        x, ex = scin.quad(fy, p0, p)
    else:
        p = p.tolist()
        kkx = scin.quad(fx, p0, p[0])
        kky = scin.quad(fy, p0, p[0])
        x = [kkx[0]]
        y = [kkx[0]]
        for i in range(len(p)-1):
            kkx = scin.quad(fx, p[i], p[i+1])
            kky = scin.quad(fy, p[i], p[i+1])
            x.append(kkx[0])
            y.append(kky[0])
        x = np.cumsum(x)
        y = np.cumsum(y)
    return [A*x,A*y]

# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:21:24 2025

@author: cesarm

Modelado y simulación de una montaña rusa.
La simulación se hace en XZ (2D) o XYZ (3D)
"""
import numbers
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import scipy.interpolate as scip
import scipy.integrate as scin
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
# Para salvar la animación en formato mpeg, es necesario instalar el paquete
# conda install -c conda-forge ffmpeg

import numbers
import types
# print(scipy.__version__)

# import utiles as UT

# class datos:
#     def __init__(self, coefRoz=0, masa=1, ca=0, sf=1, densa=1.225, grav=9.81, 
#                  vel0=1, tfin=1E6):
#         self.coefRoz = coefRoz  # Coeficiente de rozamiento (adimensional)
#         self.masa    = masa     # masa (Kg)
#         self.ca      = ca       # coeficiente de arrastre (adimensional) 
#         self.sf      = sf       # superficie frontal (m^2)
#         self.densa   = densa    # densidad del aire (Kg/m^3)
#         self.grav    = grav     # gravedad (m/s^2)
#         self.vel0    = vel0     # velocidad inicial (m/s)
#         self.tfin    = tfin     # tiempo final de la simulacion (s)
#         self.coefVis = ca*sf*densa/2/masa
#         self.y0      = [0,vel0]
#         self.tspan   = [0,tfin]

#     def actualizar(self, coefRoz=None, masa=None, ca=None, sf=None, densa=None, 
#                    grav=None, vel0=None, tfin=None):
#         if coefRoz is not None: self.coefRoz = coefRoz
#         if masa is not None:    self.masa    = masa
#         if ca is not None:      self.ca      = ca
#         if sf is not None:      self.sf      = sf
#         if densa is not None:   self.densa   = densa
#         if coefRoz is not None:
#             self.coefRoz = coefRoz

###############################################################################
#    FUNCIONES UTILES
###############################################################################
def issymbolic(x):
    '''
    comprueba una lista tiene términos simbólicos
    '''
    n = len(x)
    if np.ndim(x) !=1:
        raise ValueError('x no es una lista o vector unidimensional')
    simbolico = False
    var = []
    for i in range(n):
        if isinstance(x[i],(numbers.Number,types.LambdaType)): # valores numéricos
            continue
        elif isinstance(x[i], (sp.core.Basic)):
           simbolico = True
           new_vars = list(x[i].free_symbols) # variables
           for vari in new_vars:
               if vari not in var:
                   var.append(vari)
        else:
            raise ValueError("los valores deben ser numéricos o simbólicos") 
    return [simbolico,var]

# def lambdify_wrapper(*args):
#     tmp_res = orig_lambda_fnc(*args) # returns a list
#     # use args as reference for broadcasting
#     tmp_res.extend(args)
#     # use the explicit broadcasting trick from above
#     bc_res = numpy.array(numpy.broadcast_arrays(tmp_res))
#     # remove args
#     # use reshape to restore original shape of expr + extra dimensions due to
#     # array valued arguments
#     return reshaped_res 

def lambdaeval(funcion, puntos):
    result = funcion(puntos)
    if isinstance(puntos,numbers.Number): # evaluación en un punto
        result = np.array(result) # vector
    elif isinstance(puntos,(list,tuple,np.ndarray)): # evaluación en varios puntos
        cte = sum([isinstance(x,numbers.Number) for x in result])
        # todas las componentes de la función son constantes
        if cte == len(result):
            result =  np.atleast_2d(result).T # vector columna
            # result = result.reshape([result.shape[0],1]) solo con array
            result =  np.tile(result,len(puntos))
        # alguna componente de la función es constante
        elif cte>0:
            result = np.array(np.broadcast_arrays(*result))
        # ninguna componente de la función es constante
        else:
            pass
    return result

def broadcast(fun):
    return lambda *x: np.broadcast_arrays(fun(*x), *x)[0]

def lambdify2(variables, function):
    tmp = broadcast(sp.lambdify(variables, function))
    return tmp

# shape = broadcast([x1, x2, ...]).shape # x1, x2, ... are the inputs to the lambdified function
# return array(...) # Where ... uses full(..., shape) for constant inputs, or broadcast_to for inputs that use only a subset of the input variables

def lambdify1(var,fun):
    '''
    test if all items from array or list are numbers
    '''
    if isinstance(fun,numbers.Number):
        funcion = lambda z, c=fun : 0*z+c
    elif isinstance(fun,(list,tuple,np.ndarray)):
        funcion = [lambdify2(var,fun2) for fun2 in fun]
    else:
        funcion = sp.lambdify(var,fun,"numpy")
    return funcion
        
###############################################################################
#    POSICION, DERIVADAS Y TRIEDRO DE FRENET
###############################################################################
# es_par = lambda x: "Par" if x % 2 == 0 else "Impar"

def derivando(x):
    if isinstance(x,(int, float, complex)):
        der = 0
    else:
        simbolica, var = issymbolic(x)
        if simbolica:
            if len(var)==1:
                sol = x.diff(var[0])
            else:
                raise ValueError("Se requiere derivación direccional (>1D)")
        else:
            raise ValueError("No se puede derivar una función que no es simbólica")
    return sol

def trayec_der(curva):
    """
    Calcula las derivadas 1ª, 2ª y 3ª de la curva que define la trayectoria del
    cuerpo
    curva = r(u) = (x(u), y(u), z(u))

    Parameters
    ----------
    curva : function
        función paramétrica que describe la trayectoria

    Returns
    -------
    sol : list
        lista con r, r', r'' y r'''.

    """
    if isinstance(curva,scip.BSpline): # BSpline
        derivadas = [curva]
        for i in range(3):
            derivadas.append(derivadas[i].derivative())
    elif isinstance(curva,types.LambdaType): # Funcion vectorial
        raise ValueError("Definición de la curva no contemplada")
    elif isinstance(curva,(list,tuple,np.ndarray)): # iterable con funciones
        if len(curva) !=3: # Debe tener 3 componentes, aunque una sea nula
            raise ValueError("La curva no está 2D o 3D")
        simbolica, var = issymbolic(curva)
        if simbolica: # curva como lista de funciones simbolicas
            if len(var) != 1:
                raise ValueError("La curva debe depender de una sóla variable")
            else:
                var = var[0]
            # Derivada de cada componente simbólica
            d1c   = [fun.diff(var) for fun in curva]
            d2c   = [fun.diff(var) for fun in d1c]
            d3c   = [fun.diff(var) for fun in d2c]
            # Conversión a numérica para su evaluación rápida
            derivadas = []
            # derivadas.append(lambdify2(var,curva))
            # derivadas.append(lambdify2(var,d1c  ))
            # derivadas.append(lambdify2(var,d2c  ))
            # derivadas.append(lambdify2(var,d3c  ))
            derivadas.append(sp.lambdify(var,curva))
            derivadas.append(sp.lambdify(var,d1c  ))
            derivadas.append(sp.lambdify(var,d2c  ))
            derivadas.append(sp.lambdify(var,d3c  ))
        else: # curva como lista de funciones numéricas
            raise ValueError("Definición de la curva no contemplada")
    else:
        raise ValueError("Definición de la curva no contemplada")
    return derivadas

def triedoFrenet(derivadas,u,tol=1E-12):
    """
    Calcula los elementos del triedro de frenet de una curva parametrizada
    curva = r(u) = (x(u), y(u), z(u))

    Parameters
    ----------
    derivadas : list of functions
        lista con la expresión de la curva y sus derivadas 1ª, 2ª y 3ª
    u : float
        parámetro de la curva
    tol : float
        valor inferior de la norma, por debajo se considera el vector nulo

    Returns
    -------
    sol : list
        lista con los vectores tangente, normal y binormal así como la norma de
        r', la curvatura y la torsión.

    """
    if isinstance(derivadas[0],scip.BSpline): # Interpolación con BSplines
        d1c_u = derivadas[1](u)
        d2c_u = derivadas[2](u)
        d3c_u = derivadas[3](u)
    elif callable(derivadas[0]): # Funcion de lambdify
        d1c_u = lambdaeval(derivadas[1],u)
        d2c_u = lambdaeval(derivadas[2],u)
        d3c_u = lambdaeval(derivadas[3],u)
    else:
        raise ValueError("Definición de la curva no contemplada")

    # Vector tangente r'(t) y su norma
    vt  = d1c_u
    nvt = np.linalg.norm(vt)
    if nvt<tol: # Punto singular
        raise ValueError("La curva no es regular en u={}".format(u))
    
    # Vector binormal (producto cruzado r'(t) x r''(t)) y su norma
    # Curvatura k = ||vb|| /||r'(t)||^3
    vb   = np.cross(d1c_u,d2c_u)
    nvb  = np.linalg.norm(vb)
    curv = nvb/nvt**3
    if nvb<tol: # Es una recta. Metodo de Gram-Schmidt
        ejez = np.array([0,0,1])
        alfa = np.dot(vt,ejez)/(nvt**2)
        vn   = ejez - alfa*vt
        nvn  = np.linalg.norm(vn)
        if nvn<tol: # movimiento rectilineo vertival
            if vt[2] >0:
                vn = np.array([0,-1,0])
            else:
                vn = np.array([0,1,0])
            nvn = 1
        vb   = np.cross(vt,vn)
        nvb  = np.linalg.norm(vb)
    # Vector normal N(t) (producto cruzado entre B(t) y T(t))
    else:
        vn   = np.cross(vb, vt)
        nvn  = np.linalg.norm(vn) # innecesario al ser producto de unitarios
   
    # Torsion [r',r'',r''']/||r'xr''||^2
    if abs(curv)<1E-10:
        tors = 0
    else:
        tors = abs(np.dot(vb,d3c_u))/nvb**2
    
    # Normalización de los vectores
    vt /= nvt
    vn /= nvn
    vb /= nvb
    return [[vt,vn,vb],[nvt,curv,tors]]   

###############################################################################
#    FUERZAS Y ENERGÍA
###############################################################################

def fuerzaNormal(u,v,triedro,grav=9.81):
    """
    Cálculo de la fuerza normal por unidad de masa

    Parameters
    ----------
    u : parámetro (float)
    v : velocidad (float)
    triedro : posición y derivadas (list of bsplines)

    Returns
    -------
    sol: valor de la fuerza normal

    """
    # baseLocal: triedro de Frenet en ese punto = norma vector tg, curvatura y torsión
    # ctes :     norma vector tg, curvatura y torsión
    baseLocal, ctes = triedoFrenet(triedro,u)
    # ctes = norma vector tg, curvatura y torsión
    fuerzaN = ctes[1]*v**2 + grav*baseLocal[1][2] # k ∙ n
    return [fuerzaN, baseLocal,ctes]

def aceleracion(v,base,coefRoz,coefVis,normal,curva, grav=9.81):
    """
    Cálcula la aceleración tangencial utilizando la expresion de dv/dt, la normal
    mediante v^2/R y la total como suma de ambas 

    Parameters
    ----------
    v :     velocidad [float]
    base:   triedro de Frenet [list of arrays]
    coefRoz:coeficiente de rozamiento [float]
    coefVis:coeficiente viscoso [float]
    normal: fuerza normal por unidad de masa [float]
    curva:  curvatura [float]
    grav:   aceleración de la gravedad [float]

    Returns
    -------
    sol:    lista con los valores de la aceleración total, la tangencial y la normal

    """
    # baseLocal: triedro de Frenet en ese punto = norma vector tg, curvatura y torsión
    # ctes :     norma vector tg, curvatura y torsión
    atg = aceleratg(v,base,coefRoz,coefVis,normal,grav)
    anr = curva*v**2
    ace = np.sqrt(atg*atg + anr*anr)
    return [ace, atg, anr]

def aceleratg(v,base,coefRoz,coefVis,normal,grav=9.81):
    """
    Cálculo de la aceleración tangencial (derivada de la velocidad)

    Parameters
    ----------
    v :     velocidad [float]
    base:   triedro de Frenet [list of arrays]
    coefRoz:coeficiente de rozamiento [float]
    coefVis:coeficiente viscoso [float]
    normal: fuerza normal por unidad de masa [float]
    grav:   aceleración de la gravedad [float]
    
    Returns
    -------
    atg: valor de la aceleracion tangencia

    """
    # baseLocal: triedro de Frenet en ese punto = norma vector tg, curvatura y torsión
    # ctes :     norma vector tg, curvatura y torsión
    signo= np.array(v>=0,dtype=int)-np.array(v<0,dtype=int)
    atg  = -grav *base[0][2] #  k ∙ t
    atg += -coefRoz*abs(normal)*signo
    atg += -coefVis*v**2*signo
    return atg

def energia(u,v,posicion,grav=9.81):
    """
    Cálculo de la energia por unidad de masa

    Parameters
    ----------
    u : parámetro (float)
    v : velocidad (float)
    posicion : posición (u) (list of bsplines)

    Returns
    -------
    sol: valor de la energía

    """
    # baseLocal: triedro de Frenet en ese punto = norma vector tg, curvatura y torsión
    # ctes :     norma vector tg, curvatura y torsión
    coords = np.array(np.broadcast_arrays(*posicion(u)))
    sol = v**2/2 + grav*coords[-1]
    return sol

###############################################################################
#    SEDO Y EVENTOS
###############################################################################
def edofun_mr(t,y,*args):
    """
    Ecuación diferencial de la montaña rusa

    Parameters
    ----------
    t : real
        tiempo
    y : tupla | array
        parámetro de posición y velocidad
    args : function
        Parametrización de la curva

    Returns
    -------
    derivadas de y

    """
    # baseLocal: triedro de Frenet en ese punto = norma vector tg, curvatura y torsión
    # ctes :     norma vector tg, curvatura y torsión
    posyder, coefRoz, coefVis, grav = args
    fuerzaN, baseLocal,ctes = fuerzaNormal(y[0],y[1],posyder,grav)
    # ctes = norma vector tg, curvatura y torsión
    du  = y[1]/ctes[0]
    dv  = aceleratg(y[1],baseLocal,coefRoz,coefVis,fuerzaN,grav)
    return [du,dv]

def finalVia (t, y, *args): return y[0] -1
finalVia.terminal  = True
finalVia.direction = 0

def paradaVagon (t, y, *args): return y[1]
paradaVagon.terminal  = True
paradaVagon.direction = -1

###############################################################################
#    REPRESENTACIONES Y ANIMACIONES
###############################################################################
def animaVagon(sol,curva,dimension=2):   
    time    = np.linspace(0,sol.t[-1],501)
    data    = sol.sol(time)
    parm    = data[0]
    veloc   = data[1]
    posicion= curva(parm)
    if dimension == 2: # Curva bidimensional (uso x y z)
        fig, ax = plt.subplots()
        xpos = posicion[0]
        ypos = posicion[2]
        ax.set_xlim(min(xpos)-1, max(xpos)+1)
        ax.set_ylim(min(ypos)-1, max(ypos)+1)
        # ax.set_aspect('equal')
        tray, = ax.plot(xpos,ypos, ls = "-", color='b', alpha=0.4,
                        label="t={} s".format(""))

        point, = ax.plot([], [], 'ro',label="v={} m/s".format("0"))
        plt.title("Movimiento de un bólido sobre una curva")
        plt.xlabel("x")
        plt.ylabel("y")
        # time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
        # velocity_text = ax.text(0.20, 0.95, '', transform=ax.transAxes)
        # normal_text = ax.text(0.40, 0.95, '', transform=ax.transAxes)
        leyenda = ax.legend()

        def update(frame):
            txt_time = "t={:8.3g} s".format(time[frame])
            txt_velo = "v={:8.3g} m/s".format(veloc[frame])
            tray.set_label(txt_time)
            point.set_data([xpos[frame]], [ypos[frame]])
            point.set_label(txt_velo)
            # time_text.set_text('Tiempo = {:.2f} m/s'.format(time[frame]))
            # velocity_text.set_text('Velocidad = {:.2f} m/s'.format(veloc[frame]))
            # return point, time_text, velocity_text
            # la leyenda se actualiza borrandola y volviendola a escribir
            # ax.legend().remove()
            # ax.legend()
            leyenda.get_texts()[0].set_text(txt_time)
            leyenda.get_texts()[1].set_text(txt_velo)

            return tray, point, leyenda

        ani = FuncAnimation(fig, update, frames=len(time), interval=20, blit=False)
        plt.tight_layout()
        plt.show()
    
    elif dimension==3:
        fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="3d"))
        ax.set(xlabel="x",ylabel="y",zlabel="z")
        xpos = posicion[0]
        ypos = posicion[1]
        zpos = posicion[2]
        ax.set_xlim(min(xpos)-1, max(xpos)+1)
        ax.set_ylim(min(ypos)-1, max(ypos)+1)
        ax.set_zlim(min(zpos)-1, max(zpos)+1)
        # ax.set_aspect('equal')
        tray, = ax.plot(xpos, ypos, zpos, ls = "-", color='b', alpha=0.4,
                        label="t={} s".format(""))
        point, = ax.plot([], [],  [], 'ro',label="v={} m/s".format("0"))
        plt.title("Movimiento de un bólido sobre una curva")
        leyenda = ax.legend()

        def update(frame):
            txt_time = "t={:8.3g} s".format(time[frame])
            txt_velo = "v={:8.3g} m/s".format(veloc[frame])
            tray.set_label(txt_time)
            point.set_data_3d([xpos[frame]], [ypos[frame]], [zpos[frame]])
            point.set_label(txt_velo)
            leyenda.get_texts()[0].set_text(txt_time)
            leyenda.get_texts()[1].set_text(txt_velo)

            return tray, point, leyenda

        ani = FuncAnimation(fig, update, frames=len(time), interval=20, blit=False)
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Dimensión del espacio inválida")
    return ani
