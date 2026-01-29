import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class dinamica_particulas_confinada_2D:
    """
    
    """
    def __init__(self, N_particulas, radio_particula, l_caja, v_0 = 1.0):
        self.N = N_particulas
        self.r_radio = radio_particula
        self.l_c = l_caja
        self.v_0 = v_0
        def inicializar_variables_necesarias():
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
            Particulas = {}
            for i in range(self.N):
                Particulas[f"particula_{i}"] = {
                    'posicion': np.zeros(2),
                    'velocidad': np.zeros(2)
                }
            # Inicializamos las posiciones
            posiciones = np.random.rand(self.N, 2) * self.l_c
            # Inicializamos las velocidades
            angulos = np.random.rand(self.N) * 2 * np.pi
            velocidades = np.zeros((self.N, 2))
            velocidades[:, 0] = self.v_0 * np.cos(angulos)
            velocidades[:, 1] = self.v_0 * np.sin(angulos)
            # Creamos el diccionario de variables iniciales
            for i in range(self.N):
                Particulas[f"particula_{i}"]['posicion'] = posiciones[i]
                Particulas[f"particula_{i}"]['velocidad'] = velocidades[i]
            return Particulas
        
        variables_iniciales = inicializar_variables_necesarias()
        self.situacion_inicial = variables_iniciales

    def calcular_distancias_particulas(self, Posiciones_totales, paso):
        """ Retorna matriz NxN de distancias """
        n_part = self.N
        distancias = np.zeros((n_part, n_part))

        for i in range(n_part):
            for j in range(i+1, n_part): # Solo calculamos la mitad superior
                d = np.linalg.norm(Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"] - Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"])
                distancias[i, j] = d
                distancias[j, i] = d # La matriz es simétrica
        return distancias
    
    def simular_dinamica(self, n_pasos, delta_t,choques):
        """
        
        """
        if choques == False:
            Posiciones_totales = {}
            velocidades = {}
            for i in range(self.N):
                Posiciones_totales[f"particula_{i}"] = {}
                velocidades[f"particula_{i}"] = self.situacion_inicial[f"particula_{i}"]['velocidad']
                for paso in range(n_pasos):
                    if paso == 0:
                        Posiciones_totales[f"particula_{i}"][f"paso_{paso}"] = self.situacion_inicial[f"particula_{i}"]['posicion']
                    else:
                        Posiciones_totales[f"particula_{i}"][f"paso_{paso}"] = Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"] + velocidades[f"particula_{i}"] * delta_t
                        # Verificar colisiones con paredes y ajustar velocidad
                        for dim in range(2): # 0 = x, 1 = y
                                if Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] <= 0:
                                    velocidades[f"particula_{i}"][dim] *= -1
                                elif Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] >= self.l_c:
                                    velocidades[f"particula_{i}"][dim] *= -1
                self.dinamica_sin_choques = Posiciones_totales
                return Posiciones_totales
        else:
            Posiciones_totales = {}
            velocidades = {}
            for paso in range(n_pasos):
                rango = f"paso_{paso}"
                if paso != 0:
                    distancias = self.calcular_distancias_particulas(Posiciones_totales, paso)
                        
                    umbral = self.r_radio * 2 # Diámetro
                    for j in range(self.N):
                        for k in range(j + 1, self.N): # Iteramos pares únicos (j, k)
                            if distancias[j, k] <= umbral:
                                vector_ij = Posiciones_totales[f"particula_{k}"][f"paso_{paso-1}"] - Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"]
                                distancia_ij = np.linalg.norm(vector_ij)
                                vector_unitario = vector_ij / distancia_ij

                                velocidad_tangencial_j = np.dot(velocidades[f"particula_{j}"], vector_unitario) * vector_unitario
                                velocidad_tangencial_k = np.dot(velocidades[f"particula_{k}"], vector_unitario) * vector_unitario
                                velocidad_normal_j = velocidades[f"particula_{j}"] - velocidad_tangencial_j
                                velocidad_normal_k = velocidades[f"particula_{k}"] - velocidad_tangencial_k

                                velocidades[f"particula_{j}"] = velocidad_tangencial_k + velocidad_normal_j 
                                velocidades[f"particula_{k}"] = velocidad_tangencial_j + velocidad_normal_k
                            else:
                                continue
                
                for i in range(self.N):
                    if paso == 0:
                        Posiciones_totales[f"particula_{i}"] = {}
                        velocidades[f"particula_{i}"] = self.situacion_inicial[f"particula_{i}"]['velocidad']
                        Posiciones_totales[f"particula_{i}"][f"paso_0"] = self.situacion_inicial[f"particula_{i}"]['posicion']
                    else:
                        

                        Posiciones_totales[f"particula_{i}"][f"paso_{paso}"] = Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"] + velocidades[f"particula_{i}"] * delta_t
                        # Verificar colisiones con paredes y ajustar velocidad
                        for dim in range(2): # 0 = x, 1 = y
                            if Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] <= 0:
                                velocidades[f"particula_{i}"][dim] *= -1
                            elif Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] >= self.l_c:
                                velocidades[f"particula_{i}"][dim] *= -1
            self.dinamica_con_choques = Posiciones_totales
            return Posiciones_totales

    def generar_animacion(self, distancia_frame, cant_frames,tiempo_frame, chocan = False):
        """
        Docstring for generar_animacion
        
        :param self: Description
        """
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(0, self.l_c)
        ax.set_ylim(0, self.l_c)
        ax.set_aspect('equal')
        
        Particulas = []
        for i in range(self.N):
            particula, =ax.plot([], [], 'o', markersize=8)
            Particulas.append(particula)

        def init():
            for particula in Particulas:
                particula.set_data([], [])
            return Particulas

        if chocan == False:
            def animate(frame):
                paso = frame * distancia_frame
                for i, particula in enumerate(Particulas):
                    particula.set_data([self.dinamica_sin_choques[f"particula_{i}"][f"paso_{paso}"][0]], [self.dinamica_sin_choques[f"particula_{i}"][f"paso_{paso}"][1]])
                return Particulas
        else:
            def animate(frame):
                paso = frame * distancia_frame
                for i, particula in enumerate(Particulas):
                    particula.set_data([self.dinamica_con_choques[f"particula_{i}"][f"paso_{paso}"][0]], [self.dinamica_con_choques[f"particula_{i}"][f"paso_{paso}"][1]])
                return Particulas
            
        self.animacion = FuncAnimation(fig, animate, init_func=init, frames=cant_frames, 
                    blit=True, interval=tiempo_frame*1000, repeat=True)
        plt.show()



Dinamica = dinamica_particulas_confinada_2D(N_particulas=10, radio_particula=0.01, l_caja=1.0, v_0=0.1)
Dinamica.simular_dinamica(n_pasos=10000, delta_t=0.01, choques=True)

Dinamica.generar_animacion(10,1000,0.01,chocan=True)