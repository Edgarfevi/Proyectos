import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuración de la física ---
R = 2.0          # Radio de la fuente volumétrica
c = 1.0          # Velocidad de la luz (normalizada)
k = 5.0          # Frecuencia espacial (número de onda)
omega = c * k    # Frecuencia temporal
fps = 30         # Frames por segundo
duracion = 10    # Segundos de animación

# --- Configuración del espacio ---
r = np.linspace(0.01, 20, 1000)
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2, color='blue', label='Campo Eléctrico $E(r, t)$')
front_line = ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Frente de onda ($R+ct$)')
source_area = ax.axvspan(0, R, color='yellow', alpha=0.3, label='Fuente Volumétrica')

def init():
    ax.set_xlim(0, 20)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Distancia al centro ($r$)')
    ax.set_ylabel('Amplitud del Campo')
    ax.set_title('Propagación Causal: Campo generado por Fuente Volumétrica')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    return line, front_line

def update(frame):
    t = frame / fps
    r_front = R + c * t  # El límite de la causalidad
    
    # Aplicamos la solución demostrada:
    # E = (A/r) * sin(k(r - ct)) * Theta(R + ct - r)
    E = np.zeros_like(r)
    
    # Máscara de causalidad: Solo hay campo donde la luz ha llegado
    mask = r <= r_front
    
    # Calculamos la oscilación retardada
    # Usamos (r-R) para que el pulso nazca en la superficie de la esfera
    E[mask] = (1.5 / (r[mask] + 0.5)) * np.sin(k * (r[mask] - c * t))
    
    line.set_data(r, E)
    front_line.set_xdata([r_front])
    return line, front_line

ani = animation.FuncAnimation(fig, update, frames=fps*duracion, 
                              init_func=init, blit=True, interval=1000/fps)

# Para mostrarlo en una ventana local (necesitas tener instalado un backend como PyQt o Tkinter)
plt.show()

# Si estás en un notebook, puedes guardarlo como gif:
# ani.save('propagacion_electromagnetica.gif', writer='pillow')