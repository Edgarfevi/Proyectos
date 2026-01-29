import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================================
# Definicion de variables
# ==============================================
l_c = 1 
N = 10 
n_pasos_sim = 2000 # Reduje esto para probar más rápido, súbelo si quieres
n_pasos_repres = 5 
v_0 = 0.5 # Bajé un poco la velocidad para ver mejor los choques
delta_t = 0.005 
pausa_repres = 0.001
radio_particula = 0.02 # Definimos el radio visual y físico

# ==============================================
# Funciones
# ==============================================

def inicializar_variables_necesarias(N, v_0, l_c):
    posiciones = np.random.rand(N, 2) * l_c
    angulos = np.random.rand(N) * 2 * np.pi
    velocidades = np.zeros((N, 2))
    velocidades[:, 0] = v_0 * np.cos(angulos)
    velocidades[:, 1] = v_0 * np.sin(angulos)
    return posiciones, velocidades

def distancias_particulas(posiciones):
    """ Retorna matriz NxN de distancias """
    n_part = len(posiciones)
    distancias = np.zeros((n_part, n_part))
    # Truco de numpy para calcular distancias sin bucles (más rápido)
    # Pero tu método de bucles está bien para N=10
    for i in range(n_part):
        for j in range(i+1, n_part): # Solo calculamos la mitad superior
            d = np.linalg.norm(posiciones[i] - posiciones[j])
            distancias[i, j] = d
            distancias[j, i] = d # La matriz es simétrica
    return distancias

def actualizar_posiciones_con_choques(posiciones, velocidades, delta_t, l_c):
    global N
    
    # 1. Detectar y resolver choques entre partículas
    distancias = distancias_particulas(posiciones)
    umbral = radio_particula * 2 # Diámetro
    
    for i in range(N):
        for j in range(i + 1, N): # Iteramos pares únicos (i, j)
            if distancias[i, j] <= umbral:
                
                # --- FÍSICA CORREGIDA ---
                
                # Vector Normal (une los centros)
                res = posiciones[j] - posiciones[i]
                norm_res = np.linalg.norm(res)
                if norm_res == 0: continue # Evitar error raro si están en el mismo punto exacto
                
                n_ij = res / norm_res        # Vector unitario Normal
                t_ij = np.array([-n_ij[1], n_ij[0]]) # Vector unitario Tangencial (rotar 90 gados)
                
                # Proyectar velocidades actuales en base Normal/Tangencial
                v_i_n = np.dot(velocidades[i], n_ij)
                v_i_t = np.dot(velocidades[i], t_ij)
                
                v_j_n = np.dot(velocidades[j], n_ij)
                v_j_t = np.dot(velocidades[j], t_ij)
                
                # CHEQUEO IMPORTANTE:
                # Solo chocar si se están acercando. Si v_relativa < 0 ya se están alejando.
                # Esto evita que se "peguen" si en el siguiente frame siguen solapadas.
                v_rel_n = v_i_n - v_j_n
                
                if v_rel_n > 0: 
                    # Choque elástico masas iguales:
                    # Intercambian velocidad Normal. Conservan Tangencial.
                    # Nueva velocidad i = (v_j_n en dirección normal) + (v_i_t propio en tangencial)
                    velocidades[i] = v_j_n * n_ij + v_i_t * t_ij
                    velocidades[j] = v_i_n * n_ij + v_j_t * t_ij

    # 2. Actualizar posición (Euler)
    nuevas_posiciones = posiciones + velocidades * delta_t
    
    # 3. Rebotes con paredes
    for i in range(N):
        # Paredes X
        if nuevas_posiciones[i, 0] < 0:
            nuevas_posiciones[i, 0] = -nuevas_posiciones[i, 0] # Rebote posición para que no se salga
            velocidades[i, 0] *= -1
        elif nuevas_posiciones[i, 0] > l_c:
            nuevas_posiciones[i, 0] = 2*l_c - nuevas_posiciones[i, 0]
            velocidades[i, 0] *= -1
            
        # Paredes Y
        if nuevas_posiciones[i, 1] < 0:
            nuevas_posiciones[i, 1] = -nuevas_posiciones[i, 1]
            velocidades[i, 1] *= -1
        elif nuevas_posiciones[i, 1] > l_c:
            nuevas_posiciones[i, 1] = 2*l_c - nuevas_posiciones[i, 1]
            velocidades[i, 1] *= -1
            
    return nuevas_posiciones, velocidades

# ==============================================
# Main
# ==============================================

posiciones, velocidades = inicializar_variables_necesarias(N, v_0, l_c)

# Pre-computo de la simulación
posiciones_tot = np.zeros((n_pasos_sim, N, 2)) # Usar numpy array es más eficiente que dict

print("Calculando simulación...")
for paso in range(n_pasos_sim):
    posiciones_tot[paso] = posiciones.copy()
    # Pasamos 'velocidades' y recibimos las actualizadas
    posiciones, velocidades = actualizar_posiciones_con_choques(posiciones, velocidades, delta_t, l_c)
print("Simulación terminada. Generando animación...")

# ==============================================
# Animación
# ==============================================

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, l_c)
ax.set_ylim(0, l_c)
ax.set_aspect('equal') # Importante para que los choques se vean físicos

# CORRECCIÓN DE PLOT: Guardar los objetos Line2D correctamente
Particulas = []
for i in range(N):
    # Notar la coma después de 'linea', o usar [0]
    linea, = ax.plot([], [], 'o', markersize=8) 
    Particulas.append(linea)

def init():
    for particula in Particulas:
        particula.set_data([], [])
    return Particulas

def animate(frame):
    idx = frame * n_pasos_repres
    if idx >= n_pasos_sim: return Particulas # Seguridad
    
    datos_paso = posiciones_tot[idx]
    
    for i, particula in enumerate(Particulas):
        # set_data espera (x, y) o ([x], [y]) para puntos
        particula.set_data([datos_paso[i, 0]], [datos_paso[i, 1]])
        
    return Particulas

num_frames = n_pasos_sim // n_pasos_repres

# Asignar a variable 'ani' para evitar Garbage Collection
ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, 
                    blit=True, interval=20, repeat=False)

plt.show()