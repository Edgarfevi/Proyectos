import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial as spatial
from timeit import default_timer as timer


class dinamica_particulas_confinada_2D:
    """
    Clase para simular la dinámica de partículas confinadas en un espacio bidimensional.
    
    Esta clase implementa una simulación de dinámica molecular simplificada donde
    N partículas esféricas se mueven dentro de una caja cuadrada 2D. La simulación
    incluye colisiones elásticas entre partículas y con las paredes del contenedor.
    
    Attributes
    ----------
    N : int
        Número total de partículas en la simulación
    r_radio : float
        Radio de cada partícula (todas tienen el mismo tamaño)
    l_c : float
        Longitud del lado de la caja cuadrada que confina las partículas
    T : float
        Temperatura del sistema en unidades arbitrarias
    m : float
        Masa de cada partícula
    v_0 : float
        Velocidad característica calculada a partir de la temperatura
    situacion_inicial : dict
        Diccionario conteniendo las posiciones y velocidades iniciales de todas las partículas
    E_0 : dict
        Diccionario con las energías cinéticas iniciales de cada partícula
    dinamica_sin_choques : dict, optional
        Trayectorias completas cuando se simula sin colisiones entre partículas
    dinamica_con_choques : dict, optional
        Trayectorias completas cuando se simula con colisiones entre partículas
    velocidades_finales : dict
        Velocidades finales de todas las partículas después de la simulación
    E_f : dict, optional
        Energías cinéticas finales de cada partícula
    animacion : FuncAnimation, optional
        Objeto de animación de matplotlib
    
    Methods
    -------
    calcular_distancias_particulas(Posiciones_totales, paso)
        Calcula la matriz de distancias entre todas las partículas
    simular_dinamica(n_pasos, delta_t, choques, optimized)
        Ejecuta la simulación temporal del sistema
    generar_animacion(distancia_frame, cant_frames, tiempo_frame, chocan='no')
        Crea una animación visual del movimiento de las partículas
    histograma_energia(choques='si')
        Genera histograma de energías y verifica conservación
    comprobar_T()
        Calcula y compara temperaturas inicial y final
    comprobar_velocidades_finales()
        Compara velocidades iniciales y finales para diagnóstico
    """
    def __init__(self, N_particulas, radio_particula, l_caja, T = 50,m = 1.0, v_0 = 1.0, Kb = 0.01):
        """
        Inicializa la simulación de dinámica de partículas confinadas.
        
        Parameters
        ----------
        N_particulas : int
            Número de partículas en la simulación
        radio_particula : float
            Radio de cada partícula (todas son idénticas)
        l_caja : float
            Longitud del lado de la caja cuadrada de confinamiento
        T : float, optional
            Temperatura del sistema en unidades arbitrarias (default: 50)
        m : float, optional
            Masa de cada partícula (default: 1.0)
        v_0 : float, optional
            Velocidad inicial si se elige distribución uniforme (default: 1.0)
        Kb : float, optional
            Constante de Boltzmann en unidades del sistema (default: 0.01)
        
        Notes
        -----
        Durante la inicialización, se solicita al usuario elegir entre:
        - Velocidades con módulo constante (distribución uniforme en ángulos)
        - Velocidades con energías distribuidas exponencialmente (Maxwell-Boltzmann)
        """
        
        self.N = N_particulas
        self.r_radio = radio_particula
        self.l_c = l_caja
        self.T = T
        self.m = m
        self.v_0 = np.sqrt(2 * (Kb * T) / m)

        def inicializar_variables_necesarias(v_constante = 'si', Kb = 0.01):
            """
            Inicializa las posiciones y velocidades de las partículas.
            
            Genera las condiciones iniciales del sistema asignando posiciones
            aleatorias dentro de la caja y velocidades según el método elegido.
            
            Parameters
            ----------
            v_constante : str, optional
                'si' para velocidades con módulo constante v_0
                'no' para distribución exponencial de energías (default: 'si')
            Kb : float, optional
                Constante de Boltzmann para distribución exponencial (default: 0.01)
            
            Returns
            -------
            dict
                Diccionario con estructura:
                {'particula_i': {'posicion': array([x, y]), 'velocidad': array([vx, vy])}}
            
            Notes
            -----
            - Las posiciones se generan aleatoriamente evitando las paredes
            - Si v_constante='si': todas las partículas tienen |v| = v_0
            - Si v_constante='no': energías siguen distribución exponencial
            - Las direcciones de velocidad son siempre aleatorias
            - Actualiza self.E_0 con las energías cinéticas iniciales
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
        """
        Calcula la matriz de distancias euclidianas entre todas las partículas.
        
        Parameters
        ----------
        Posiciones_totales : dict
            Diccionario con las posiciones de todas las partículas en todos los pasos
            Estructura: {'particula_i': {'paso_j': array([x, y])}}
        paso : int
            Número del paso temporal para el cual calcular las distancias
        
        Returns
        -------
        numpy.ndarray
            Matriz simétrica NxN donde el elemento [i,j] contiene la distancia
            euclidiana entre las partículas i y j en el paso especificado
        
        Notes
        -----
        La matriz es simétrica (distancias[i,j] = distancias[j,i])
        Solo se calculan las distancias de la mitad superior para eficiencia
        """
        n_part = self.N
        distancias = np.zeros((n_part, n_part))

        for i in range(n_part):
            for j in range(i+1, n_part): # Solo calculamos la mitad superior
                d = np.linalg.norm(Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"] - Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"])
                distancias[i, j] = d
                distancias[j, i] = d # La matriz es simétrica
        return distancias
    
    
    def simular_dinamica(self, n_pasos, delta_t,choques,optimized):
        """
        Simula la evolución temporal del sistema de partículas.
        
        Integra las ecuaciones de movimiento usando el método de Euler,
        incluyendo colisiones elásticas con las paredes y opcionalmente
        entre partículas.
        
        Parameters
        ----------
        n_pasos : int
            Número de pasos temporales a simular
        delta_t : float
            Incremento de tiempo entre pasos consecutivos
        choques : str
            'si' para incluir colisiones entre partículas
            'no' para solo considerar colisiones con paredes
        optimized : str
            'yes' para usar algoritmo KDTree (eficiente para muchas partículas)
            'no' para cálculo directo de todas las distancias
        
        Returns
        -------
        dict
            Diccionario con todas las posiciones en cada paso temporal:
            {'particula_i': {'paso_j': array([x, y])}}
        
        Notes
        -----
        - Actualiza self.dinamica_sin_choques o self.dinamica_con_choques
        - Actualiza self.velocidades_finales con las velocidades al final
        - Las colisiones son perfectamente elásticas
        - El algoritmo optimizado es recomendado para N > 100
        - Imprime tiempo de procesamiento por paso cuando choques='si'
        
        See Also
        --------
        calcular_distancias_particulas : Método auxiliar para versión no optimizada
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
                tstart = timer()
                if optimized == 'yes':
                    if paso != 0:
                        
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

                tend = timer()
                print(f"Tiempo para procesar colisiones de las {self.N} partículas en el paso {paso}: {tend - tstart} segundos")

            self.dinamica_con_choques = Posiciones_totales
            self.velocidades_finales = velocidades
            return Posiciones_totales

    def generar_animacion(self, distancia_frame, cant_frames,tiempo_frame, chocan = 'no'):
        """
        Genera y muestra una animación del movimiento de las partículas.
        
        Crea una visualización animada del sistema usando los datos de la
        simulación previamente ejecutada.
        
        Parameters
        ----------
        distancia_frame : int
            Número de pasos temporales entre frames consecutivos de la animación
            (permite acelerar la animación saltando pasos)
        cant_frames : int
            Número total de frames en la animación
        tiempo_frame : float
            Duración de cada frame en segundos
        chocan : str, optional
            'si' para animar la dinámica con colisiones entre partículas
            'no' para animar sin colisiones (default: 'no')
        
        Returns
        -------
        None
            Muestra la animación en pantalla
        
        Notes
        -----
        - Requiere haber ejecutado simular_dinamica() previamente
        - La animación se repite indefinidamente
        - Actualiza self.animacion con el objeto FuncAnimation
        - El tamaño de la figura es 10x10 pulgadas
        - Las partículas se muestran como círculos azules
        
        Raises
        ------
        KeyError
            Si no se ha ejecutado la simulación correspondiente antes
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
        Genera histograma de energías cinéticas finales y verifica conservación.
        
        Calcula las energías cinéticas de todas las partículas al final de
        la simulación, muestra su distribución y opcionalmente verifica la
        conservación de energía total.
        
        Parameters
        ----------
        choques : str, optional
            'si' para verificar conservación de energía total
            'no' para solo mostrar histograma (default: 'si')
        
        Returns
        -------
        None
            Muestra el histograma y opcionalmente imprime energías totales
        
        Notes
        -----
        - Actualiza self.E_f con las energías cinéticas finales
        - El histograma usa 25 bins en el rango [0, 1]
        - Si choques='si', compara energía total inicial vs final
        - Para colisiones elásticas, la energía debería conservarse
        - Requiere haber ejecutado simular_dinamica() previamente
        
        See Also
        --------
        comprobar_T : Verifica conservación de temperatura
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


    def comprobar_T(self):
        """
        Calcula y compara las temperaturas inicial y final del sistema.
        
        Utiliza la relación del teorema de equipartición para calcular
        la temperatura a partir de la energía cinética promedio.
        
        Returns
        -------
        None
            Imprime las temperaturas inicial y final
        
        Notes
        -----
        La temperatura se calcula usando: T = <E_cin> / (Kb * N)
        donde <E_cin> es la energía cinética total del sistema.
        
        En un sistema aislado con colisiones elásticas, la temperatura
        debería conservarse.
        
        Requiere:
        - self.E_0: energías iniciales (generadas en __init__)
        - self.E_f: energías finales (generadas en histograma_energia)
        
        Raises
        ------
        AttributeError
            Si no se han calculado E_0 o E_f previamente
        
        See Also
        --------
        histograma_energia : Debe ejecutarse antes para calcular E_f
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

    def comprobar_velocidades_finales(self):
        """
        Compara velocidades iniciales y finales para diagnóstico.
        
        Función de utilidad para debugging que compara elemento a elemento
        las velocidades antes y después de la simulación.
        
        Returns
        -------
        None
            Imprime arrays booleanos indicando igualdad componente a componente
        
        Notes
        -----
        Para cada partícula, imprime un array 2x2 booleano donde:
        - True indica que esa componente no cambió
        - False indica que la componente fue modificada
        
        En una simulación correcta:
        - Sin colisiones y sin paredes: todas deberían ser True
        - Con colisiones o paredes: la mayoría deberían ser False
        
        Esta función es principalmente para verificar que la simulación
        está modificando correctamente las velocidades.
        
        Requires
        --------
        - self.situacion_inicial: condiciones iniciales
        - self.velocidades_finales: velocidades al final de la simulación
        """
        velocidad_final = np.zeros((self.N,2))
        velocid_inicial = np.zeros((self.N,2))
        for particula in range(self.N):
            velocidad_final[particula] = self.velocidades_finales[f"particula_{particula}"]
            velocid_inicial[particula] = self.situacion_inicial[f"particula_{particula}"]['velocidad']
            
            print(velocidad_final == velocid_inicial)
            
        return None



Dinamica = dinamica_particulas_confinada_2D(N_particulas=100, radio_particula=0.125, l_caja=10.0, v_0=1.0)
Dinamica.simular_dinamica(n_pasos=10000, delta_t=0.01, choques= 'si',optimized = 'no')

Dinamica.generar_animacion(10,1000,0.01,chocan = 'si')
Dinamica.histograma_energia(choques = 'si')

Dinamica.comprobar_T()

