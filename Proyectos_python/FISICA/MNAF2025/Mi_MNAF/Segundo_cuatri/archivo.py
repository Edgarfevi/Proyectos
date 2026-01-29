import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation


# ==============================================
# Definicion de variables del problema
# ==============================================

l_c = 1 # Lado de la caja 
N = 10 # Numero de particulas
m_0 = 1 # Masa de las particulas
n_pasos_sim = 100000 # Numero de pasos de la simulacion
n_pasos_repres = 10 # Numero de pasos entre representaciones
v_0 = 1 # Velocidad inicial de las particulas, (modulo)
delta_t = 0.001 # Paso temporal de la simulacion
pausa_repres = 0.01 # Pausa entre representaciones (en segundos)

# ==============================================
# Inicializacion de variables (creamos funciones)
# ==============================================
def inicializar_variables_necesarias(N, v_0, l_c):
    """
    Inicializa las variables necesarias para la simulacion:
    - Posiciones iniciales de las particulas (array de Nx2)
    - Velocidades iniciales de las particulas (array de Nx2)
    Argumentos:
    N = numero de particulas
    v_0 = velocidad inicial de las particulas (modulo)
    l_c = lado de la caja
    Retorna:
    variables_iniciales: dict con las variables iniciales
    """

    # Inicializamos las posiciones
    posiciones = np.random.rand(N, 2) * l_c
    # Inicializamos las velocidades
    angulos = np.random.rand(N) * 2 * np.pi
    velocidades = np.zeros((N, 2))
    velocidades[:, 0] = v_0 * np.cos(angulos)
    velocidades[:, 1] = v_0 * np.sin(angulos)
    # Creamos el diccionario de variables iniciales
    variables_iniciales = {
        'posiciones': posiciones,
        'velocidades': velocidades
    }
    return variables_iniciales

def actualizar_posiciones(posiciones, velocidades, delta_t, l_c):
    """
    Actualiza las posiciones de las particulas segun sus velocidades y el paso temporal.
    Argumentos:
    posiciones = array de Nx2 con las posiciones actuales de las particulas
    velocidades = array de Nx2 con las velocidades actuales de las particulas
    delta_t = paso temporal
    l_c = lado de la caja
    ajusta la velocidad si la particula choca con las paredes
    Retorna:
    nuevas_posiciones = array de Nx2 con las nuevas posiciones de las particulas
    """
    nuevas_posiciones = posiciones + velocidades * delta_t
    
    for i in range(len(posiciones)):
        # Check x-coordinate (column 0)
        if nuevas_posiciones[i, 0] < 0:
            velocidades[i, 0] *= -1
        elif nuevas_posiciones[i, 0] > l_c:
            velocidades[i, 0] *= -1
            
        # Check y-coordinate (column 1)
        if nuevas_posiciones[i, 1] < 0:
            velocidades[i, 1] *= -1
        elif nuevas_posiciones[i, 1] > l_c:
            velocidades[i, 1] *= -1
            
    return nuevas_posiciones


# llamamos a las funciones para inicializar las variables
variables_iniciales = inicializar_variables_necesarias(N, v_0, l_c)
posiciones = variables_iniciales['posiciones']
velocidades = variables_iniciales['velocidades']

print(velocidades)

fig, ax = plt.subplots(figsize=(10,10))

ax.set_title('Simulacion de particulas en una caja')
ax.set_xlim(0, l_c)
ax.set_ylim(0, l_c)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# Inicializar el plot de las particulas 
Particulas = []
for i in range(N):
    particula, =ax.plot([], [], 'o', markersize=8)
    Particulas.append(particula)

posiciones_tot = {}
for rango in range(n_pasos_sim):
    posiciones_tot[rango] = posiciones.copy()
    posiciones = actualizar_posiciones(posiciones, velocidades, delta_t, l_c)

print(Particulas)

def init():
    for particula in Particulas:
        particula.set_data([], [])
    return Particulas

def animate(frame):
    global posiciones_tot, num_pasos_repres
    
    # Actualizamos el plot
    for i, particula in enumerate(Particulas):
        particula.set_data([posiciones_tot[frame*n_pasos_repres][i, 0]], [posiciones_tot[frame*n_pasos_repres][i, 1]])
    return Particulas

# Calculamos el numero de frames basado en los pasos totales y los pasos entre representaciones
num_frames = n_pasos_sim // n_pasos_repres

ani = FuncAnimation(fig, animate, init_func=init, frames=1000, 
                    blit=True, interval=pausa_repres*1000, repeat=True)

plt.show()


def distancias_particulas(posiciones):
    """
    Calcula las distancias entre todas las particulas.
    Argumentos:
    posiciones = array de Nx2 con las posiciones actuales de las particulas
    Retorna:
    distancias = array de NxN con las distancias entre las particulas
    """
    global N

    distancias = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                distancias[i, j] = np.linalg.norm(posiciones[i] - posiciones[j])
    return distancias

def vectores_unitarios(i,j, posiciones):
    """
    Calcula el vector unitario que va de la particula i a la particula j.
    Argumentos:
    i = indice de la particula i
    j = indice de la particula j
    posiciones = array de Nx2 con las posiciones actuales de las particulas
    Retorna:
    vector_unitario = array de 2 con el vector unitario de i a j
    vector_normal_unitario = array de 2 con el vector normal unitario de i a j
    """
    vector_ij = posiciones[j] - posiciones[i]
    distancia_ij = np.linalg.norm(vector_ij)
    vector_unitario = vector_ij / distancia_ij

    vector_normal = posiciones[i]-(vector_unitario[0]*posiciones[i][0] + vector_unitario[1]*posiciones[i][1])*vector_unitario
    vector_normal_unitario = vector_normal / np.linalg.norm(vector_normal)
    return vector_unitario, vector_normal_unitario

def actualizar_posiciones_con_choques(posiciones, delta_t, l_c):
    """
    Actualiza las velocidades de las particulas en caso de choque.
    Argumentos:
    posiciones = array de Nx2 con las posiciones actuales de las particulas
    velocidades = array de Nx2 con las velocidades actuales de las particulas
    Retorna:
    nuevas_velocidades = array de Nx2 con las nuevas velocidades de las particulas
    """
    global N, velocidades

    distancias = distancias_particulas(posiciones)
    
    for i in range(N):
        for j in range(i,N):
            if i != j:
                if distancias[i, j] == 0:
                    print("Error: Division por cero en distancias, en particulas ", i, " y ", j)
                if distancias[i, j] <= 0.02:  # Umbral de choque
                    u_ij, n_ij = vectores_unitarios(i, j, posiciones)
                    
                    velocidad_tangencial_i = np.dot(velocidades[i], u_ij) * u_ij
                    velocidad_tangencial_j = np.dot(velocidades[j], u_ij) * u_ij
                    velocidad_normal_i = velocidades[i] - velocidad_tangencial_i
                    velocidad_normal_j = velocidades[j] - velocidad_tangencial_j

                    velocidades[i] = velocidad_tangencial_j + velocidad_normal_i 
                    velocidades[j] = velocidad_tangencial_i + velocidad_normal_j
            else:
                continue


    nuevas_posiciones = posiciones + velocidades * delta_t
    
    for i in range(len(posiciones)):
        # Check x-coordinate (column 0)
        if nuevas_posiciones[i, 0] < 0:
            velocidades[i, 0] *= -1
        elif nuevas_posiciones[i, 0] > l_c:
            velocidades[i, 0] *= -1
            
        # Check y-coordinate (column 1)
        if nuevas_posiciones[i, 1] < 0:
            velocidades[i, 1] *= -1
        elif nuevas_posiciones[i, 1] > l_c:
            velocidades[i, 1] *= -1
            
    return nuevas_posiciones
    
fig, ax = plt.subplots(figsize=(10,10))

ax.set_title('Simulacion de particulas en una caja')
ax.set_xlim(0, l_c)
ax.set_ylim(0, l_c)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')


posiciones_tot = {}
for rango in range(n_pasos_sim):
    posiciones_tot[rango] = posiciones.copy()
    posiciones = actualizar_posiciones_con_choques(posiciones, delta_t, l_c)

Particulas = []
for i in range(N):
    particula, =ax.plot([], [], 'o', markersize=8)
    Particulas.append(particula)

def init():
    for particula in Particulas:
        particula.set_data([], [])
    return Particulas

def animate(frame):
    global posiciones_tot, num_pasos_repres
    
    # Actualizamos el plot
    for i, particula in enumerate(Particulas):
        particula.set_data([posiciones_tot[frame*n_pasos_repres][i, 0]], [posiciones_tot[frame*n_pasos_repres][i, 1]])
    return Particulas

# Calculamos el numero de frames basado en los pasos totales y los pasos entre representaciones
num_frames = n_pasos_sim // n_pasos_repres

ani = FuncAnimation(fig, animate, init_func=init, frames=1000, 
                    blit=True, interval=pausa_repres*1000, repeat=True)

plt.show()
