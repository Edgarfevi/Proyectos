# ...existing code...
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Propagacion_ondas_1D:
    """
    Docstring for Propagacion_ondas_1D class.
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
        er1=16,
        er2=49
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
        self.variable_apoyo1[:-1] = self.variable_apoyo1[1:]
        self.variable_apoyo1[-1] = self.Ey[1]
        self.Ey[-1] = self.variable_apoyo3[-1]
        self.variable_apoyo3[1:] = self.variable_apoyo3[:-1]
        self.variable_apoyo3[0] = self.Ey[-2]
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



class Propagacion_ondas_2D(Propagacion_ondas_1D):
    def __init__(
        self,
        c=3e8,
        espaciado_mallado=10e-9,
        puntos_mallado=401,
        simulacion_pasos=10000,
        delta_xp=40e-9,
        E0=1,
        kp=200,
        distancia_simulacion=10,
        er1=1,
        er2=4,
        lp=200
    ):
        super().__init__(
            c=c,
            espaciado_mallado=espaciado_mallado,
            puntos_mallado=puntos_mallado,
            simulacion_pasos=simulacion_pasos,
            delta_xp=delta_xp,
            E0=E0,
            kp=kp,
            distancia_simulacion=distancia_simulacion,
            er1=er1,
            er2=er2
        )
    
        self.malla_y = None
        self.Ez = None
        self.Hx = None
        self.Hy = None
        self.lp = lp

    def generar_mallado(self):
        super().generar_mallado()
        self.malla_y = np.linspace(0, self.num_puntos * self.espaciado_mallado, self.num_puntos)
        print(self.malla_x.shape == self.malla_y.shape)

        Malla_Y, Malla_X = np.meshgrid(self.malla_y, self.malla_x)
        self.malla_x = Malla_X
        self.malla_y = Malla_Y

        return Malla_X, Malla_Y
    
    def Onda_2D(self):
        self.Ez = np.zeros((self.num_puntos, self.num_puntos))
        self.Hx = np.zeros((self.num_puntos, self.num_puntos))
        self.Hy = np.zeros((self.num_puntos, self.num_puntos))

    def Pulo_2D(self, t):
        return self.E0 * np.exp(-((t - self.top) ** 2) / (2 * (self.delta_tp ** 2)))
    
    def Propagacion_campos_2D(self, t):
        self.Ez[1:, 1:] = self.Ez[1:, 1:] + 0.5 * (self.Hy[1:, 1:] - self.Hy[:-1, 1:]) - 0.5*(self.Hx[1:, 1:] - self.Hx[1:, :-1])
        self.Ez[self.kp, self.lp] = self.Pulo_2D(t)
        self.Hx[:, :-1] = self.Hx[:, :-1] - 0.5 * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy[:-1, :] = self.Hy[:-1, :] + 0.5 * (self.Ez[1:, :] - self.Ez[:-1, :])

    def generar_animacion_2D(self):
        if self.malla_x is None or self.malla_y is None or self.malla_t is None:
            self.generar_mallado()

        if self.kp < 0 or self.kp >= self.num_puntos:
            raise ValueError(f"kp fuera de rango: {self.kp} (0..{self.num_puntos-1})")

        self.Onda_2D()
        niveles=np.linspace(-0.1,0.1,21)
        fig, ax = plt.subplots()
        cs = ax.contourf(self.malla_x, self.malla_y, np.clip(self.Ez, -0.1, 0.1), niveles, cmap='hot')
        barracolor = plt.colorbar(cs)


        total_frames = self.simulacion_pasos // self.distancia_simulacion

        def animacion(frame):
            inicio = frame * self.distancia_simulacion
            fin = min(inicio + self.distancia_simulacion, self.simulacion_pasos)
            for t in range(int(inicio), int(fin)):
                tiempo = t * self.paso_temporal
                self.Propagacion_campos_2D(tiempo)
            ax.cla()
            ax.contourf(self.malla_x, self.malla_y, np.clip(self.Ez, -0.1, 0.1), niveles, cmap='hot')
            ax.set_title(f'Propagación de Onda Electromagnética en 2D -- t = {(frame * self.distancia_simulacion * self.paso_temporal * 10**15):.2f} fs')
            
            return None


        self.anim = FuncAnimation(
            fig, animacion, frames=total_frames, interval=10, blit=False, repeat=False
        )
        plt.show()

"""
Propagacion_ondas_2D_instance = Propagacion_ondas_2D()
print(Propagacion_ondas_2D_instance.delta_tp)

Propagacion_ondas_2D_instance.generar_animacion_2D()

"""
Propagacion_ondas_1D_instance = Propagacion_ondas_1D()
Propagacion_ondas_1D_instance.generar_animacion()

