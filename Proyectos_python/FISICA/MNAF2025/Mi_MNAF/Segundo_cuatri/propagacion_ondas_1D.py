
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Propagacion_ondas_1D:
    """
    Clase para simular la propagación de ondas electromagnéticas en 1D usando el método FDTD (Finite-Difference Time-Domain).
    
    Permite calcular y visualizar la evolución temporal y espacial de los campos eléctrico (Ey) y magnético (Hz)
    en un dominio 1D, con opciones de incluir condiciones de contorno absorbentes y medios con conductividad.
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
        er2=49,
        tipo = 'c',
        e0 = 8.854e-12,
        sigma1 = 0,
        sigma2 = 4000
    ):
        """
        Inicializa los parámetros de la simulación de propagación de ondas electromagnéticas en 1D.
        
        Parameters:
            c (float): Velocidad de propagación de la onda (velocidad de la luz), default 3e8 m/s
            espaciado_mallado (float): Resolución espacial del mallado (dx), default 10e-9 m
            puntos_mallado (int): Número de puntos en el mallado espacial, default 1001
            simulacion_pasos (int): Número de pasos temporales a simular, default 10000
            delta_xp (float): Ancho espacial del pulso, default 400e-9 m
            E0 (float): Amplitud inicial del campo eléctrico, default 1
            kp (int): Índice espacial donde se inyecta el pulso, default 250
            distancia_simulacion (int): Número de pasos para graficar en cada frame, default 10
            er1 (float): Permitividad relativa para la PML en la primera frontera, default 16
            er2 (float): Permitividad relativa para la PML en la segunda frontera, default 49
            sigma1 (float): Conductividad para la primera región, default 0
            sigma2 (float): Conductividad para la segunda región, default 4000
        """
        # Parámetros de propagación
        self.velocidad_prop = c  # Velocidad de propagación de la onda
        self.espaciado_mallado = espaciado_mallado  # Resolución espacial (dx)
        self.paso_temporal = self.espaciado_mallado / (2 * self.velocidad_prop)  # Paso de tiempo (dt) según criterio de estabilidad FDTD
        self.num_puntos = puntos_mallado  # Número de nodos en la malla espacial
        self.simulacion_pasos = simulacion_pasos  # Número total de pasos temporales

        # Parámetros del pulso gaussiano
        self.delta_xp = delta_xp  # Ancho espacial del pulso
        self.E0 = E0  # Amplitud del campo eléctrico
        self.kp = kp  # Posición donde se inyecta el pulso
        self.delta_tp = self.delta_xp / self.velocidad_prop  # Ancho temporal del pulso
        self.top = 5 * self.delta_tp  # Instante de tiempo cuando alcanza el máximo el pulso
        
        # Parámetros de visualización y simulación
        self.distancia_simulacion = distancia_simulacion  # Pasos temporales por frame de animación
        self.er1 = er1  # Permitividad relativa para PML frontera 1
        self.er2 = er2  # Permitividad relativa para PML frontera 2
        self.e0 = e0  # Permitividad del vacío
        self.sigma1 = sigma1  # Conductividad para la primera región
        self.sigma2 = sigma2  # Conductividad para la segunda región
        self.ctc1 = sigma1 * self.paso_temporal / (2 * self.e0 * self.er1)  # Coeficiente de conductividad para región 1 
        self.ctc2 = sigma2 * self.paso_temporal / (2 * self.e0 * self.er2)  # Coeficiente de conductividad para región 2
        # Mallados espacial y temporal
        self.malla_x = None  # Coordenadas espaciales
        self.malla_t = None  # Coordenadas temporales
        
        # Campos electromagnéticos
        self.Ey = None  # Campo eléctrico en dirección y
        self.Hz = None  # Campo magnético en dirección z
        
        # Variables auxiliares para condiciones de contorno absorbentes (ABC)
        self.variable_apoyo1 = 0  # Histórico frontera izquierda campo Ey (paso anterior)
        self.variable_apoyo2 = 0  # Histórico frontera izquierda campo Ey (paso anterior-anterior)
        self.variable_apoyo3 = 0  # Histórico frontera derecha campo Ey (paso anterior)
        self.variable_apoyo4 = 0  # Histórico frontera derecha campo Ey (paso anterior-anterior)

        # Buffers para capas PML (Perfectly Matched Layer)
        self.variable_apoyo_er1 = np.zeros(np.round(2*np.sqrt(self.er1)).astype(int))  # Búfer PML frontera izquierda
        self.variable_apoyo2_er2 = np.zeros(np.round(2*np.sqrt(self.er2)).astype(int))  # Búfer PML frontera derecha

        if tipo == 'a':
            self.tipo = 'absorbente_relativo'  # Usar capas PML para condiciones absorbentes
        elif tipo == 'b':
            self.tipo = 'absorbente_cte'  # Usar ABC simple para condiciones absorbentes
        elif tipo == 'c':
            self.tipo = 'conductor'  # Simular medio conductor (E=0 en región conductora)
        else:
            raise ValueError(f"Tipo de medio no reconocido: {tipo} (usar 'a' para PML, 'b' para ABC, 'c' para conductor)")
        

    def generar_mallado_1D(self):
        """
        Genera los mallados espacial y temporal para la simulación.
        
        Crea vectores de coordenadas (x, t) que definen los puntos donde se calculan
        los campos electromagnéticos.
        
        Returns:
            None (actualiza atributos: self.malla_x y self.malla_t)
        """
        # Mallado espacial: línea recta desde 0 hasta L (puntos_mallado * espaciado_mallado)
        self.malla_x = np.linspace(0, self.num_puntos * self.espaciado_mallado, self.num_puntos)
        # Mallado temporal: desde 0 hasta T_final con paso dt
        self.malla_t = np.arange(0, self.simulacion_pasos * self.paso_temporal, self.paso_temporal)

        return None

    def Onda_1D(self):
        """
        Inicializa los campos electromagnéticos a ceros.
        
        Crea los vectores que almacenarán los valores del campo eléctrico (Ey)
        y del campo magnético (Hz) en todos los puntos del mallado.
        
        Returns:
            None (actualiza atributos: self.Ey y self.Hz)
        """
        # Inicializa campo eléctrico Ey con ceros en todos los puntos
        self.Ey = np.zeros(self.num_puntos)
        # Inicializa campo magnético Hz con ceros en todos los puntos
        self.Hz = np.zeros(self.num_puntos)

        return None

    def Pulso(self, t):
        """
        Calcula la amplitud de un pulso gaussiano en tiempo t.
        
        Parameters:
            t (float): Instante de tiempo actual de la simulación
            
        Returns:
            float o array: Valor del pulso gaussiano centrado en self.top con amplitud E0
        """
        # Envolvente gaussiana: E(t) = E0 * exp(-(t - t0)^2 / (2 * sigma^2))
        return self.E0 * np.exp(-((t - self.top) ** 2) / (2 * (self.delta_tp ** 2)))
    
    def E_r(self):
        """
        Calcula los coeficientes de permitividad relativa inversa para capas PML.
        
        Genera un perfil de permitividad inversa para implementar capas absorbentes
        perfectamente acopladas (PML) en los dos extremos del dominio.
        
        Returns:
            array: Vector de coeficientes 1/(2*epsilon_r) concatenados para ambas regiones
        """
        # Primera región: mitad de los puntos con permitividad inversa basada en er1
        e0 = np.ones(self.num_puntos // 2) * (1 / (2 * self.er1))
        # Segunda región: mitad restante de los puntos con permitividad inversa basada en er2
        er = np.ones((self.num_puntos +1)//2) * (1 / (2 * self.er2))

        # Concatena ambas regiones - total debe ser exactamente self.num_puntos
        return np.concatenate((e0, er), axis=None)

    def Ca(self):
        """
        Calcula los coeficientes de actualización para el campo eléctrico en presencia de conductividad.
        
        Genera un perfil de coeficientes que incluye el efecto de la conductividad en la actualización
        del campo eléctrico, útil para simular medios con pérdidas.
        
        Returns:
            array: Vector de coeficientes Ca para la actualización del campo eléctrico
        """
        # Primera region: mitad de los puntos con coeficiente basado en sigma1
        ca1 = np.ones(self.num_puntos//2) * (1 - self.ctc1) / (1 + self.ctc1)
        # Segunda region: mitad restante de los puntos con coeficiente basado en sigma2
        ca2 = np.ones((self.num_puntos+1)//2) * (1 - self.ctc2) / (1 + self.ctc2)
        # Concatena ambas regiones - total debe ser exactamente self.num_puntos
        return np.concatenate((ca1, ca2), axis=None)
    
    def Cb(self):
        """
        Calcula los coeficientes de actualización para el campo eléctrico en presencia de conductividad.
        
        Genera un perfil de coeficientes que incluye el efecto de la conductividad en la actualización
        del campo eléctrico, útil para simular medios con pérdidas.
        
        Returns:
            array: Vector de coeficientes Cb para la actualización del campo eléctrico
        """
        # Primera region: mitad de los puntos con coeficiente basado en sigma1
        cb1 = np.ones(self.num_puntos//2) * (1 / (2 * self.er1 * (1 + self.ctc1)))
        # Segunda region: mitad restante de los puntos con coeficiente basado en sigma2
        cb2 = np.ones((self.num_puntos+1)//2) * (1 / (2 * self.er2 * (1 + self.ctc2)))
        # Concatena ambas regiones - total debe ser exactamente self.num_puntos
        return np.concatenate((cb1, cb2), axis=None)

    def Propagacion_campos(self, t):
        """
        Actualiza los campos electromagnéticos usando el método FDTD sin condiciones absorbentes.
        
        Implementa las ecuaciones de Maxwell en diferencias finitas para un paso de tiempo,
        sin incluir condiciones de contorno especiales (las ondas se reflejarán en los bordes).
        
        Parameters:
            t (float): Instante de tiempo actual de la simulación
            
        Returns:
            None (actualiza in-place los atributos self.Ey y self.Hz)
        """
        # Actualiza campo eléctrico Ey: 
        # Ey = Ey - 0.5 * (Hz[i+1] - Hz[i]) / dx
        self.Ey[1:] = self.Ey[1:] - 0.5 * (self.Hz[1:] - self.Hz[:-1])
        # Inyecta el pulso de fuente en la posición kp
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        # Actualiza campo magnético Hz:
        # Hz = Hz - 0.5 * (Ey[i] - Ey[i-1]) / dx
        self.Hz[:-1] = self.Hz[:-1] - 0.5 * (self.Ey[1:] - self.Ey[:-1])

        return None

    def Absorbente(self, t):
        """
        Actualiza los campos electromagnéticos con condiciones de contorno absorbentes (ABC).
        
        Implementa el método FDTD con condiciones de contorno absorbentes de primer orden
        para minimizar las reflexiones en los bordes del dominio.
        
        Parameters:
            t (float): Instante de tiempo actual de la simulación
            
        Returns:
            None (actualiza in-place los atributos self.Ey y self.Hz)
        """
        # Actualiza campo eléctrico Ey (ecuación de Maxwell estándar)
        self.Ey[1:] = self.Ey[1:] - 0.5 * (self.Hz[1:] - self.Hz[:-1])
        # Inyecta el pulso de fuente
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        
        # Condición absorbente en frontera izquierda (x=0):
        # Usa valores históricos para absorber ondas que salen
        self.Ey[0] = self.variable_apoyo1
        self.variable_apoyo1 = self.variable_apoyo2
        self.variable_apoyo2 = self.Ey[1]
        
        # Condición absorbente en frontera derecha (x=L):
        self.Ey[-1] = self.variable_apoyo3
        self.variable_apoyo3 = self.variable_apoyo4
        self.variable_apoyo4 = self.Ey[-2]
        
        # Actualiza campo magnético Hz (ecuación de Maxwell estándar)
        self.Hz[:-1] = self.Hz[:-1] - 0.5 * (self.Ey[1:] - self.Ey[:-1])


        return None

    def Absorbente_relativo(self, t):
        """
        Actualiza los campos electromagnéticos con condiciones PML (capas absorbentes perfectas).
        
        Implementa el método FDTD con capas absorbentes perfectamente acopladas (PML)
        usando coeficientes de permitividad relativa para mejorar la absorción de ondas
        en los bordes del dominio.
        
        Parameters:
            t (float): Instante de tiempo actual de la simulación
            
        Returns:
            None (actualiza in-place los atributos self.Ey y self.Hz)
        """
        # Actualiza campo eléctrico Ey con coeficientes de permitividad relativos
        self.Ey[1:] = self.Ey[1:] - self.E_r()[1:] * (self.Hz[1:] - self.Hz[:-1])
        # Inyecta el pulso de fuente
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        
        # Condición PML en frontera izquierda:
        # Extrae el primer valor del búfer PML
        self.Ey[0] = self.variable_apoyo_er1[0]
        # Desplaza el búfer hacia atrás (FIFO)
        self.variable_apoyo_er1[:-1] = self.variable_apoyo_er1[1:]
        # Añade el nuevo valor al final del búfer
        self.variable_apoyo_er1[-1] = self.Ey[1]
        
        # Condición PML en frontera derecha:
        self.Ey[-1] = self.variable_apoyo2_er2[-1]
        # Desplaza el búfer hacia adelante (LIFO)
        self.variable_apoyo2_er2[1:] = self.variable_apoyo2_er2[:-1]
        # Añade el nuevo valor al principio del búfer
        self.variable_apoyo2_er2[0] = self.Ey[-2]
        
        # Actualiza campo magnético Hz
        self.Hz[:-1] = self.Hz[:-1] - 0.5 * (self.Ey[1:] - self.Ey[:-1])


        return None

    def Medio_conductor(self, t):
        """
        Actualiza los campos electromagnéticos en presencia de un medio conductor.
        
        Implementa el método FDTD estableciendo el campo eléctrico a cero en la región
        conductora (los campos no pueden propagarse a través de un conductor perfecto).
        
        Parameters:
            t (float): Instante de tiempo actual de la simulación
            
        Returns:
            None (actualiza in-place los atributos self.Ey y self.Hz)
        """
        # Actualiza campo eléctrico Ey con coeficientes de permitividad relativos
        self.Ey[1:] = self.Ey[1:]*self.Ca()[1:] - self.Cb()[1:] * (self.Hz[1:] - self.Hz[:-1])
        # Inyecta el pulso de fuente
        self.Ey[self.kp] = self.Ey[self.kp] + self.Pulso(t)
        
        # Condición PML en frontera izquierda:
        # Extrae el primer valor del búfer PML
        self.Ey[0] = self.variable_apoyo_er1[0]
        # Desplaza el búfer hacia atrás (FIFO)
        self.variable_apoyo_er1[:-1] = self.variable_apoyo_er1[1:]
        # Añade el nuevo valor al final del búfer
        self.variable_apoyo_er1[-1] = self.Ey[1]
        
        # Condición PML en frontera derecha:
        self.Ey[-1] = self.variable_apoyo2_er2[-1]
        # Desplaza el búfer hacia adelante (LIFO)
        self.variable_apoyo2_er2[1:] = self.variable_apoyo2_er2[:-1]
        # Añade el nuevo valor al principio del búfer
        self.variable_apoyo2_er2[0] = self.Ey[-2]
        
        # Actualiza campo magnético Hz
        self.Hz[:-1] = self.Hz[:-1] - 0.5 * (self.Ey[1:] - self.Ey[:-1])

        return None

    def generar_animacion(self):
        """
        Genera y muestra una animación de la propagación de la onda electromagnética en 1D.
        
        Realiza la simulación temporal y anima el comportamiento de los campos Ey y Hz
        mientras se propagan en el dominio 1D. El tipo de condiciones de contorno se determina
        por el parámetro `tipo` establecido durante la inicialización.
        
        Returns:
            None (muestra animación en ventana matplotlib)
        """
        # Genera el mallado si no existe
        if self.malla_x is None or self.malla_t is None:
            self.generar_mallado_1D()

        # Valida que el índice de fuente esté dentro del rango
        if self.kp < 0 or self.kp >= self.num_puntos:
            raise ValueError(f"kp fuera de rango: {self.kp} (0..{self.num_puntos-1})")

        # Inicializa los campos a cero
        self.Onda_1D()
        
        
        # Configura la figura y los ejes
        fig, ax = plt.subplots()
        # Línea para el campo eléctrico Ey
        Onda_Ey = ax.plot(self.malla_x * 1e6, self.Ey, label='Campo Eléctrico (Ey)')[0]
        # Línea para el campo magnético Hz
        Onda_Hz = ax.plot(self.malla_x * 1e6, self.Hz, label='Campo Magnético (Hz)')[0]
        # Etiquetas y título
        ax.set_xlabel('Posición (µm)')
        ax.set_ylabel('Amplitud de Campo (A/m - Ey | A/m - Hz)')
        Titulo = ax.set_title('Propagación de Onda Electromagnética en 1D')
        ax.legend()
        # Fija los límites del eje y
        if self.tipo == 'absorbente_relativo' or self.tipo == 'conductor':
            ax.set_ylim(-1.5 * self.er1,1.5 *self.er1)
            ax.vlines(self.num_puntos//2 * self.espaciado_mallado * 1e6, -2 * self.er1, 2 * self.er1, color='green', linestyle='--', label='Fuente')
        else:
            ax.set_ylim(-2 * self.E0, 2 * self.E0)
            ax.vlines(self.num_puntos//2 * self.espaciado_mallado * 1e6, -2 * self.E0, 2 * self.E0, color='green', linestyle='--', label='Fuente')
        # Marca visual de la posición de la fuente
        # Calcula el número total de frames para la animación
        total_frames = self.simulacion_pasos // self.distancia_simulacion

        def animacion(frame):
            """
            Función interna que ejecuta un paso de la animación.
            
            Parameters:
                frame (int): Índice del frame actual
                
            Returns:
                tuple: Elementos actualizados (Titulo, Onda_Ey, Onda_Hz)
            """
            # Rango temporal para este frame
            inicio = frame * self.distancia_simulacion
            fin = min(inicio + self.distancia_simulacion, self.simulacion_pasos)
            
            # Ejecuta varios pasos temporales antes de graficar
            for t in range(int(inicio), int(fin)):
                tiempo = t * self.paso_temporal
                # Selecciona el tipo de propagación según las flags
                if self.tipo == 'absorbente_relativo':
                    self.Absorbente_relativo(tiempo)  # PML
                elif self.tipo == 'absorbente_cte':
                    self.Absorbente(tiempo)  # ABC simple
                elif self.tipo == 'conductor':
                    self.Medio_conductor(tiempo)  # Con conductor
            
            # Actualiza el título con el tiempo actual en femtosegundos
            Titulo.set_text(f'Propagación de Onda Electromagnética -- t = {(frame * self.distancia_simulacion * self.paso_temporal * 10**15):.2f} fs')
            # Actualiza los datos de las líneas
            Onda_Ey.set_ydata(self.Ey)
            Onda_Hz.set_ydata(self.Hz)

            return Titulo, Onda_Ey, Onda_Hz

        # Crea la animación
        self.anim = FuncAnimation(
            fig, animacion, frames=total_frames, interval=10, blit=False , repeat=False
        )
        # Muestra la ventana con la animación
        plt.show()

        return None


