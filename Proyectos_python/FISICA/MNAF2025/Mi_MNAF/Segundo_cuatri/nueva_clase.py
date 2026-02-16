# ...existing code...
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Propagacion_ondas:
    """
    Docstring for Propagacion_ondas
    """

    def __init__(
        self,
        c=3e8,
        espaciado_mallado=10e-9,
        puntos_mallado=1001,
        simulacion_pasos=10000,
        delta_xp=400e-9,
        E0=1,
        kp=250,
        distancia_simulacion=10,
        er1=1,
        er2=4
    ):
        self.velocidad_prop = c
        self.espaciado_mallado = espaciado_mallado
        self.paso_temporal = self.espaciado_mallado / (2 * self.velocidad_prop)
        self.num_puntos = puntos_mallado
        self.simulacion_pasos = simulacion_pasos

        self.delta_xp = delta_xp
        self.E0 = E0
        self.kp = kp
        self.delta_tp = self.delta_xp / self.velocidad_prop
        self.top = 5 * self.delta_tp
        self.distancia_simulacion = distancia_simulacion
        self.er1 = er1
        self.er2 = er2

        self.malla_x = None
        self.malla_t = None
        self.Ey = None
        self.Hz = None

        self.variable_apoyo1 = np.zeros(np.round(2*np.sqrt(self.er1)).astype(int))
        self.variable_apoyo2 = np.zeros(np.round(2*np.sqrt(self.er1)).astype(int))
        self.variable_apoyo3 = np.zeros(np.round(2*np.sqrt(self.er2)).astype(int))
        self.variable_apoyo4 = np.zeros(np.round(2*np.sqrt(self.er2)).astype(int))
        

    def generar_mallado(self):
        self.malla_x = np.linspace(0, self.num_puntos * self.espaciado_mallado, self.num_puntos)
        self.malla_t = np.arange(0, self.simulacion_pasos * self.paso_temporal, self.paso_temporal)

    def Onda_1D(self):
        self.Ey = np.zeros(self.num_puntos)
        self.Hz = np.zeros(self.num_puntos)

    def Pulso(self, t):
        return self.E0 * np.exp(-((t - self.top) ** 2) / (2 * (self.delta_tp ** 2)))
    
    def E_r(self):
        e0=np.ones(500)*(1/(2*self.er1))
        er = np.ones(501)*(1/(2*self.er2))

        return np.concatenate((e0,er),axis = None)



    def Propagacion_campos(self, t):
        self.Ey[1:] = self.Ey[1:] - 0.5 * (self.Hz[1:] - self.Hz[:-1])
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        self.Hz[:-1] = self.Hz[:-1] - 0.5 * (self.Ey[1:] - self.Ey[:-1])

    def Absorbente(self,t):
        self.Ey[1:] = self.Ey[1:] - 0.5 * (self.Hz[1:] - self.Hz[:-1])
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        self.Ey[0] = self.variable_apoyo1
        self.variable_apoyo1 = self.variable_apoyo2
        self.variable_apoyo2 = self.Ey[1]
        self.Ey[-1] = self.variable_apoyo3
        self.variable_apoyo3 = self.variable_apoyo4
        self.variable_apoyo4 = self.Ey[-2]
        self.Hz[:-1] = self.Hz[:-1] - 0.5 * (self.Ey[1:] - self.Ey[:-1])

    def Absorbente_relativo(self, t):
        self.Ey[1:] = self.Ey[1:] - self.E_r()[1:]*(self.Hz[1:] - self.Hz[:-1])
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        self.Ey[0] = self.variable_apoyo1[0]
        self.variable_apoyo1[:-1] = self.variable_apoyo2[1:]
        self.variable_apoyo2[-1] = self.Ey[1]
        self.Ey[-1] = self.variable_apoyo3[-1]
        self.variable_apoyo3[1:] = self.variable_apoyo4[:-1]
        self.variable_apoyo4[0] = self.Ey[-2]
        self.Hz[:-1] = self.Hz[:-1] - 0.5 *(self.Ey[1:] - self.Ey[:-1])

    def generar_animacion(self):
        if self.malla_x is None or self.malla_t is None:
            self.generar_mallado()

        if self.kp < 0 or self.kp >= self.num_puntos:
            raise ValueError(f"kp fuera de rango: {self.kp} (0..{self.num_puntos-1})")

        self.Onda_1D()
        
        

        fig, ax = plt.subplots()
        Onda_Ey = ax.plot(self.malla_x*1e6, self.Ey, label='Campo Eléctrico (Ey)')[0]
        Onda_Hz = ax.plot(self.malla_x*1e6, self.Hz, label='Campo Magnético (Hz)')[0]
        ax.set_xlabel('Posición (µm)')
        ax.set_ylabel('Amplitud')
        Titulo = ax.set_title('Propagación de Onda Electromagnética') 
        ax.legend()
        ax.set_ylim(-2 * self.E0, 2 * self.E0)
        ax.vlines(500 * self.espaciado_mallado * 1e6, -2 * self.E0, 2 * self.E0, color='green', linestyle='--', label='Fuente')
        total_frames = self.simulacion_pasos // self.distancia_simulacion

        def animacion(frame):

            inicio = frame * self.distancia_simulacion
            fin = min(inicio + self.distancia_simulacion, self.simulacion_pasos)
            for t in range(int(inicio), int(fin)):
                tiempo = t * self.paso_temporal
                
                self.Absorbente_relativo(tiempo)


            Titulo.set_text(f'Propagación de Onda Electromagnética -- t = {(frame * self.distancia_simulacion * self.paso_temporal * 10**15):.2f} fs')
            Onda_Ey.set_ydata(self.Ey)
            Onda_Hz.set_ydata(self.Hz)


            return Titulo, Onda_Ey, Onda_Hz

        self.anim = FuncAnimation(
            fig, animacion, frames=total_frames, interval=10, blit=False, repeat=False
        )
        plt.show()



# ...existing code...
Prueba = Propagacion_ondas()

print(Prueba.E_r())
Prueba.generar_animacion()
# ...existing code...


