import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial as spatial
from timeit import default_timer as timer


class dinamica_particulas_confinada_2D:
    """
    
    """
    def __init__(self, N_particulas, radio_particula, l_caja, T = 50,m = 1.0, v_0 = 1.0, Kb = 0.01):
        """
        
        """
        
        self.N = N_particulas
        self.r_radio = radio_particula
        self.l_c = l_caja
        self.T = T
        self.m = m
        self.v_0 = np.sqrt(2 * (Kb * T) / m)
        def inicializar_variables_necesarias(v_constante = 'si', Kb = 0.01):
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
            posiciones = np.random.rand(self.N, 2) * (self.l_c - 2*self.r_radio)
            # Inicializamos las velocidades
            angulos = np.random.rand(self.N) * 2 * np.pi
            velocidades = np.zeros((self.N, 2))
            E_0 = {}
            if v_constante == 'si':
                velocidades[:, 0] = self.v_0 * np.cos(angulos)
                velocidades[:, 1] = self.v_0 * np.sin(angulos)
                for i in range(self.N):
                    E_0[f"particula_{i}"] = 0.5 * self.m * self.v_0**2
                
            else:
                En_0 = np.random.exponential(scale=Kb*self.T, size=self.N)
                
                for i, E in enumerate(En_0):
                    v = np.sqrt(2 * E / self.m)
                    velocidades[i, 0] = v * np.cos(angulos[i])
                    velocidades[i, 1] = v * np.sin(angulos[i])
                    E_0[f"particula_{i}"] = E

            self.E_0 = E_0
                    
            # Creamos el diccionario de variables iniciales
            for i in range(self.N):
                Particulas[f"particula_{i}"]['posicion'] = posiciones[i]
                Particulas[f"particula_{i}"]['velocidad'] = velocidades[i]
            return Particulas
        v_constante = input("¿Desea que las partículas tengan la misma velocidad inicial? (si/no): ")
        variables_iniciales = inicializar_variables_necesarias(v_constante)
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
    
    def simular_dinamica(self, n_pasos, delta_t,choques,optimized = 'no'):
        """
        
        """
        if choques == 'no':
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
                                if Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] <= 0 + self.r_radio:
                                    velocidades[f"particula_{i}"][dim] *= -1
                                elif Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] >= self.l_c - self.r_radio:
                                    velocidades[f"particula_{i}"][dim] *= -1
            self.dinamica_sin_choques = Posiciones_totales
            self.velocidades_finales = velocidades
            return Posiciones_totales
        else:
            Posiciones_totales = {}
            velocidades = {}
            for paso in range(n_pasos):
                if optimized == 'yes':
                    if paso != 0:
                        tstart = timer()
                        puntos =np.zeros((self.N,2))
                        for j in range(self.N):
                            puntos[j,0] = Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"][0]
                            puntos[j,1] = Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"][1]
                        árbol = spatial.KDTree(puntos)
                        pares_cercanos = árbol.query_pairs(r=2*self.r_radio)

                        for (j, k) in pares_cercanos:

                            vector_ij = Posiciones_totales[f"particula_{k}"][f"paso_{paso-1}"] - Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"]
                            distancia_ij = np.linalg.norm(vector_ij)
                            vector_unitario = vector_ij / distancia_ij

                            velocidad_tangencial_j = np.dot(velocidades[f"particula_{j}"], vector_unitario) * vector_unitario
                            velocidad_tangencial_k = np.dot(velocidades[f"particula_{k}"], vector_unitario) * vector_unitario
                            velocidad_normal_j = velocidades[f"particula_{j}"] - velocidad_tangencial_j
                            velocidad_normal_k = velocidades[f"particula_{k}"] - velocidad_tangencial_k

                            if np.dot(velocidades[f"particula_{k}"], vector_unitario) - np.dot(velocidades[f"particula_{j}"], vector_unitario) < 0:
                                velocidades[f"particula_{j}"] = velocidad_tangencial_k + velocidad_normal_j 
                                velocidades[f"particula_{k}"] = velocidad_tangencial_j + velocidad_normal_k
                            else:
                                continue
                        tend = timer()
                        print(f"Tiempo para procesar colisiones optimizadas de las 100 partículas en el paso {paso}: {tend - tstart} segundos")
                else:
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

                                    if np.dot(velocidades[f"particula_{k}"], vector_unitario) - np.dot(velocidades[f"particula_{j}"], vector_unitario) < 0:
                                        velocidades[f"particula_{j}"] = velocidad_tangencial_k + velocidad_normal_j 
                                        velocidades[f"particula_{k}"] = velocidad_tangencial_j + velocidad_normal_k
                                    else:
                                        continue
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
                            if Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] <= 0 + self.r_radio:
                                velocidades[f"particula_{i}"][dim] *= -1
                            elif Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] >= self.l_c - self.r_radio:
                                velocidades[f"particula_{i}"][dim] *= -1
            self.dinamica_con_choques = Posiciones_totales
            self.velocidades_finales = velocidades
            return Posiciones_totales

    def generar_animacion(self, distancia_frame, cant_frames,tiempo_frame, chocan = 'no'):
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
            particula, =ax.plot([], [], 'o', markersize=10)
            Particulas.append(particula)

        def init():
            for particula in Particulas:
                particula.set_data([], [])
            return Particulas

        if chocan == 'no':
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
    
    def histograma_energia(self,choques = 'si'):
        """
        Docstring for histograma_energia
        :param self: Description
        """
        Energias = {}
        for particulas in range(self.N):
            velocidad_final = self.velocidades_finales[f"particula_{particulas}"]
            energia_cinetica = 0.5 * self.m * (np.linalg.norm(velocidad_final))**2
            Energias[f"particula_{particulas}"] = energia_cinetica
        energias_array = np.array(list(Energias.values()))
        self.E_f = Energias
        plt.figure(figsize=(8,6))
        plt.hist(energias_array, bins=25, edgecolor='black',range=(0, 1))
        plt.title('Histograma de Energías Cinéticas Finales')
        plt.xlabel('Energía Cinética')
        plt.ylabel('Número de Partículas')
        plt.show()
        if choques == 'si':
            Energia_total_final = np.sum(energias_array)

            velocidad_inicial = np.zeros((self.N,2))
            Energias_iniciales = np.zeros(self.N)
            for particulas in range(self.N):
                velocidad_inicial[particulas] = self.situacion_inicial[f"particula_{particulas}"]['velocidad']
                Energias_iniciales[particulas] = 0.5 * self.m * (np.linalg.norm(velocidad_inicial[particulas]))**2
            Energias_iniciales_total = np.sum(Energias_iniciales) 
            print(f"Energía Cinética Total Inicial: {Energias_iniciales_total}")
            print(f"Energía Cinética Total Final: {Energia_total_final}")
        return None
    def comprobar_velocidades_finales(self):
        """
        Docstring for comprobar_velocidades_finales
        :param self: Description
        """
        velocidad_final = np.zeros((self.N,2))
        velocid_inicial = np.zeros((self.N,2))
        for particula in range(self.N):
            velocidad_final[particula] = self.velocidades_finales[f"particula_{particula}"]
            velocid_inicial[particula] = self.situacion_inicial[f"particula_{particula}"]['velocidad']
            
            print(velocidad_final == velocid_inicial)
            
        return None

    def comprobar_T(self):
        """
        Docstring for comprobar_T
        :param self: Description
        """
        Kb = 0.01
        Prom_energia_inicial = 0
        Prom_energia_final = 0
        for rango in range(self.N):
            Prom_energia_inicial += self.E_0[f"particula_{rango}"]
            Prom_energia_final += self.E_f[f"particula_{rango}"]

        T_inicial = Prom_energia_inicial/(Kb*self.N)
        T_final = Prom_energia_final/(Kb*self.N)
        print("Comprobación de temperaturas inicial y final:")
        print(f"Temperatura inicial: {T_inicial}")
        print(f"Temperatura final: {T_final}")
        return None




Dinamica = dinamica_particulas_confinada_2D(N_particulas=100, radio_particula=0.125, l_caja=10.0, v_0=1.0)
Dinamica.simular_dinamica(n_pasos=10000, delta_t=0.01, choques= 'si',optimized = 'no')

Dinamica.generar_animacion(10,1000,0.01,chocan = 'si')
Dinamica.histograma_energia(choques = 'si')

Dinamica.comprobar_T()

