import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import factorial

'''ECUACIÓN DE SCHRÖDINGER'''

'''PARÁMETROS DEL SISTEMA '''
nx = 1001 #Número de pasos espaciales
nt=50000 #Número de pasos temporales

#Extremos del intervalo espacial [-5,5]
a=-5
b=5
x=np.linspace(a,b,nx)
dx=(b-a)/(nx-1)
dt = (dx**2)/2
x0=0
sigma=0.5
k0=10
Vx=0 #Empezamos con una partícula libre, es decir, el potencial es nulo
omega=4 

#Para la animación
t_repre=50 #represento los resultados una vez que hallan pasado 50 tiempos 



'''FUNCIONES DEL PROBLEMA '''

def deriva2_sin_cond_periodicas(psi, delta_x=dx):
    '''
    Esta función calcula la derivada espacial segunda de una función de onda psi dada en un intervalo espacial dado cuando NO hay condiciones periódicas.
    ----------------------------
    PARÁMETROS DE LA FUNCIÓN:
    ----------------------------
    psi: array o lista con los valores espaciales de la función psi
    delta_x: tamaño del paso espacial
    '''
    der2 = np.zeros_like(psi)
    der2[1:-1] = (psi[2:] + psi[:-2] - 2*psi[1:-1]) / (delta_x**2)

    return der2

def deriva2_con_cond_periodicas(psi,delta_x=dx):
    '''
    Esta función calcula la derivada espacial segunda de una función de onda psi dada cuando SÍ hay condiciones periódicas.
    ----------------------------
    PARÁMETROS DE LA FUNCIÓN:
    ----------------------------
    psi: array o lista con los valores espaciales de la función de onda psi
    delta_x: tamaño del paso espacial
    '''

    der2=(1/(delta_x**2))*(np.roll(psi,-1)+np.roll(psi,1)-2*psi)

    return der2

def amplitud(x,s=sigma, x0=x0):
    '''
    Función que calcula la amplitud de la función de onda de una partícula libre en el punto x
    ----------------------------
    PARÁMETROS DE LA FUNCIÓN:
    ----------------------------
    x: punto del espacio en el que se quiere calcular la amplitud de la función de onda
    s: desviación típica de la distribución
    x0: x0 es el punto donde está centrada la distribución
    '''
    amp=np.exp(-0.5*((x-x0)/s)**2)
    return amp

def potencial_armonico(x,w=omega):
    '''
    Función que calcula el valor del potencial de tipo armónico en el/los puntos x
    ----------------------------
    PARÁMETROS DE LA FUNCIÓN:
    ----------------------------
    x: punto del espacio en el que se quiere calcular la amplitud de la función de onda
    w: frecuencia angular del potencial armónico
    '''
    return 0.5*w**2*x**2


def norma(psi_r,psi_i,delta_x=dx):
    '''
    Esta función calcula la norma de una función de onda dadas sus partes real e imaginaria
    psi_r: array con los valores de la parte real de la función de onda en todos los puntos del mallado espacial
    psi_i: array con los valores de la parte imaginaria de la función de onda en todos los puntos del mallado espacial
    '''

    return np.sqrt(np.sum(psi_r**2 + psi_i**2) * delta_x)

def hermite(x,n):
    '''
    Esta función calcula el valor de los polinomios de Hermite de grado n en una serie de puntos x.
    -----------------------------------------------------------------------------------------------
    PARÁMETROS DE LA FUNCIÓN:
    x: valor/es espaciales en los que se quiere evaluar el polinomio de Hermite [float, list, np.array]
    n: grado del polinomio de Hermite que se desea calcular [int]
    -----------------------------------------------------------------------------------------------
    RETURNS:
    H: valor/es que toma el polinomio de Hermite de grado n en el/los puntos x 
    '''

    if n==0:
        H=1
        return H
    
    elif n==1:
        H=2*x
        return H
    
    else:
        H0=1
        H1=2*x

        for deg in range(2,n+1):
            H=2*x*H1-2*(deg-1)*H0
            H0=H1
            H1=H
        return H




''' RESOLUCIÓN DEL PROBLEMA '''

''' 1. Sin condiciones periódicas'''

#Dividimos la función de onda en parte real e imaginaria 
#phi=exp(-0.5*((x-x0)/sigma)**2)*cos(k0*x)+i*exp(-0.5*((x-x0)/sigma)**2)*sen(k0*x)

psi_r=amplitud(x)*np.cos(k0*x)
psi_i=amplitud(x)*np.sin(k0*x)

#Los normalizamos
N=norma(psi_r, psi_i)
psi_r /= N
psi_i /= N


#Condiciones de contorno para esta parte:
psi_r[0],psi_r[-1],psi_i[0],psi_i[-1]=0,0,0,0


#Creo la figura para representar las funciones de onda
fig1, ax1 = plt.subplots(figsize=(10, 5))
line_psi_r, = ax1.plot(x, psi_r, color='lightblue', alpha=0.5, label=r"$\psi_{r}$")
line_psi_i, = ax1.plot(x, psi_i, color='pink', alpha=0.5, label=r"$\psi_{i}$")
line_psi, = ax1.plot(x, np.sqrt(psi_r**2 + psi_i**2), color='navy', lw=2, label=r"$|\psi|$")

ax1.set_xlim(a, b) 
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel("Posición (x)")
ax1.set_ylabel("Amplitud")
ax1.set_title("Función de onda SIN condiciones periódicas")
ax1.legend(loc='upper right')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)


#Para ver la evolución con el tiempo de la función de onda hacemos un bucle temporal

def update1(frame):

    global psi_r, psi_i

    for f in range(t_repre):
        #En cada iteración actualizamos todos los puntos menos el primero y el último que deben permanecer a cero
        psi_i=psi_i+(dt/2)*deriva2_sin_cond_periodicas(psi_r)
        psi_r=psi_r-dt*deriva2_sin_cond_periodicas(psi_i)
        psi_i=psi_i+(dt/2)*deriva2_sin_cond_periodicas(psi_r)

    line_psi_r.set_ydata(psi_r)
    line_psi_i.set_ydata(psi_i)
    line_psi.set_ydata(np.sqrt(psi_r**2+psi_i**2))
    time_text.set_text(f"Tiempo: {frame * t_repre * dt:.4f} s")
    
    return line_psi_r, line_psi_i, line_psi, time_text

ani = FuncAnimation(fig1, update1, frames=nt//t_repre, interval=10, blit=True)

plt.show()


''' 2. Con condiciones periódicas'''

#Dividimos de nuevo la función de onda en parte real e imaginaria 
psi_r_p=amplitud(x)*np.cos(k0*x)
psi_i_p=amplitud(x)*np.sin(k0*x)

#Los normalizamos
N2=norma(psi_r_p, psi_i_p)
psi_r_p /= N2
psi_i_p /= N2

#Condiciones de contorno iniciales:
psi_r_p[0],psi_r_p[-1],psi_i_p[0],psi_i_p[-1]=0,0,0,0

#Creo la figura para representar las funciones de onda
fig2, ax2 = plt.subplots(figsize=(10, 5))
line_psi_p_r, = ax2.plot(x, psi_r_p, color='orange', alpha=0.5, label=r"$\psi_{r}$")
line_psi_p_i, = ax2.plot(x, psi_i_p, color='darkgreen', alpha=0.5, label=r"$\psi_{i}$")
line_psi_p, = ax2.plot(x, np.sqrt(psi_r_p**2 + psi_i_p**2), color='sienna', lw=2, label=r"$|\psi|$")

ax2.set_xlim(a, b) 
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel("Posición (x)")
ax2.set_ylabel("Amplitud")
ax2.set_title("Función de onda CON condiciones periódicas")
ax2.legend(loc='upper right')
time_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)



def update2(frame):

    global psi_r_p, psi_i_p

    for f in range(t_repre):
        #En cada iteración actualizamos todos los puntos menos el primero y el último que deben permanecer a cero
        psi_i_p=psi_i_p+(dt/2)*deriva2_con_cond_periodicas(psi_r_p)
        psi_r_p=psi_r_p-dt*deriva2_con_cond_periodicas(psi_i_p)
        psi_i_p=psi_i_p+(dt/2)*deriva2_con_cond_periodicas(psi_r_p)

    line_psi_p_r.set_ydata(psi_r_p)
    line_psi_p_i.set_ydata(psi_i_p)
    line_psi_p.set_ydata(np.sqrt(psi_r_p**2+psi_i_p**2))
    time_text2.set_text(f"Tiempo: {frame * t_repre * dt:.4f} s")
    
    return line_psi_p_r, line_psi_p_i, line_psi_p, time_text2

ani = FuncAnimation(fig2, update2, frames=nt//t_repre, interval=10, blit=True)

plt.show()

''' 3. Añadiendo un potencial armónico'''

#Para esta última parte, vamos a cambiar la definición inicial de la función de onda. 
#Vamos a construir las funciones de onda como una combinación lineal de los 10 primeros polinomios de Hermite

psi_iv = np.zeros_like(x)
psi_rv = np.zeros_like(x)

for n in range(10):
    psi_rv+=np.exp(-0.5*omega*x**2)*np.cos(n*np.pi*0.25)*(1/factorial(n))*hermite(np.sqrt(omega)*x,n)
    psi_iv+=np.exp(-0.5*omega*x**2)*np.sin(n*np.pi*0.25)*(1/factorial(n))*hermite(np.sqrt(omega)*x,n)


#Los normalizamos
N3=norma(psi_rv, psi_iv)
psi_rv /= N3
psi_iv /= N3

print(f"Norma de la función de onda con potencial armónico: {norma(psi_rv, psi_iv)}")

#Ahora calculamos el potencial en todos los puntos del mallado
V=potencial_armonico(x,omega)

#Condiciones de contorno iniciales:
psi_rv[0],psi_rv[-1],psi_iv[0],psi_iv[-1]=0,0,0,0

#Creo la figura para representar las funciones de onda
fig3, ax3 = plt.subplots(figsize=(10, 5))
line_psi_rv, = ax3.plot(x, psi_rv, color='yellow', alpha=0.5, label=r"$\psi_{r}$")
line_psi_iv, = ax3.plot(x, psi_iv, color='pink', alpha=0.5, label=r"$\psi_{i}$")
line_psi_v, = ax3.plot(x, np.sqrt(psi_rv**2 + psi_iv**2), color='purple', lw=2, label=r"$|\psi|$")

ax3.set_xlim(a, b) 
ax3.set_ylim(-1.5, 1.5)
ax3.set_xlabel("Posición (x)")
ax3.set_ylabel("Amplitud")
ax3.plot(x, V/np.max(V), color="black", alpha=0.3)
ax3.set_title("Función de onda de una partícula sometida a un potencial armónico")
ax3.legend(loc='upper right')
time_text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)


#Para ver la evolución con el tiempo de la función de onda hacemos un bucle temporal

def update3(frame):

    global psi_rv, psi_iv

    for f in range(t_repre):
        #En cada iteración actualizamos todos los puntos menos el primero y el último que deben permanecer a cero
        psi_iv=psi_iv+(dt/2)*deriva2_sin_cond_periodicas(psi_rv)-2*dt*V*psi_rv
        psi_rv=psi_rv-dt*deriva2_sin_cond_periodicas(psi_iv)+2*dt*V*psi_iv
        psi_iv=psi_iv+(dt/2)*deriva2_sin_cond_periodicas(psi_rv)-2*dt*V*psi_rv

        #Condiciones de contorno 
        psi_rv[0], psi_rv[-1] = 0,0
        psi_iv[0], psi_iv[-1] = 0,0

    line_psi_rv.set_ydata(psi_rv)
    line_psi_iv.set_ydata(psi_iv)
    line_psi_v.set_ydata(np.sqrt(psi_rv**2+psi_iv**2))
    time_text3.set_text(f"Tiempo: {frame * t_repre * dt:.4f} s")
    
    return line_psi_rv, line_psi_iv, line_psi_v, time_text3

ani = FuncAnimation(fig3, update3, frames=nt//t_repre, interval=10, blit=True)
plt.show()








