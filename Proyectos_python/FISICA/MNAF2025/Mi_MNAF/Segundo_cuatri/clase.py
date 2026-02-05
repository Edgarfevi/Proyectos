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
    inicializar_variables_necesarias(v_constante, Kb)
        Genera posiciones y velocidades iniciales de las partículas
    calcular_distancias_particulas(Posiciones_totales, paso)
        Calcula la matriz de distancias entre todas las partículas
    vectores_choque(Posiciones_totales, velocidades, i, j, paso)
        Calcula componentes de velocidad para colisión elástica
    simular_dinamica(n_pasos, delta_t, choques, optimized)
        Ejecuta la simulación temporal del sistema (choques y optimized son bool)
    generar_animacion(distancia_frame, cant_frames, tiempo_frame)
        Crea una animación visual del movimiento de las partículas
    histograma_energia()
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

    def inicializar_variables_necesarias(self,v_constante = 'si', Kb = 0.01):
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
        # Creamos el diccionario principal que contendrá todas las partículas
        Particulas = {}
        
        # Inicializar la estructura del diccionario con arrays vacíos para cada partícula
        for i in range(self.N):
            Particulas[f"particula_{i}"] = {
                'posicion': np.zeros(2),   # Vector 2D (x, y)
                'velocidad': np.zeros(2)   # Vector 2D (vx, vy)
            }
        
        # Generar posiciones aleatorias dentro de la caja, evitando que se superpongan con las paredes
        # Rango: [r_radio, l_c - r_radio] para garantizar que toda la partícula quede dentro
        posiciones = np.random.rand(self.N, 2) * (self.l_c - 2*self.r_radio) + self.r_radio
        
        # Generar ángulos aleatorios para la dirección de las velocidades (distribución uniforme)
        angulos = np.random.rand(self.N) * 2 * np.pi
        
        # Inicializar array de velocidades (se llenará según el método elegido)
        velocidades = np.zeros((self.N, 2))
        
        # Diccionario para almacenar energías cinéticas iniciales
        E_0 = {}
        
        # CASO 1: Velocidades con módulo constante (todas las partículas con |v| = v_0)
        if v_constante == 'si':
            # Descomponer la velocidad v_0 en componentes x e y usando el ángulo aleatorio
            velocidades[:, 0] = self.v_0 * np.cos(angulos)  # Componente x
            velocidades[:, 1] = self.v_0 * np.sin(angulos)  # Componente y
            
            # Todas las partículas tienen la misma energía cinética
            for i in range(self.N):
                E_0[f"particula_{i}"] = 0.5 * self.m * self.v_0**2  # E = (1/2)mv²
            
        # CASO 2: Energías distribuidas exponencialmente (distribución Maxwell-Boltzmann)
        elif v_constante == 'no':
            # Generar N energías siguiendo distribución exponencial
            En_0 = np.random.exponential(scale=Kb*self.T, size=self.N)
            
            # Para cada partícula, calcular velocidad a partir de su energía asignada
            for i, E in enumerate(En_0):
                v = np.sqrt(2 * E / self.m)
                
                # Descomponer en componentes usando el ángulo aleatorio
                velocidades[i, 0] = v * np.cos(angulos[i])  # Componente x
                velocidades[i, 1] = v * np.sin(angulos[i])  # Componente y
                
                # Guardar la energía inicial de esta partícula
                E_0[f"particula_{i}"] = E

        # Guardar las energías iniciales como atributo de la clase
        self.E_0 = E_0
        
        # Asignar las posiciones y velocidades generadas al diccionario de partículas
        for i in range(self.N):
            Particulas[f"particula_{i}"]['posicion'] = posiciones[i]
            Particulas[f"particula_{i}"]['velocidad'] = velocidades[i]
        
        self.situacion_inicial = Particulas
        return Particulas

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
        # Obtener el número total de partículas
        n_part = self.N
        
        # Crear matriz cuadrada NxN inicializada en ceros
        # Esta matriz almacenará todas las distancias entre pares de partículas
        distancias = np.zeros((n_part, n_part))

        # Iterar sobre todas las partículas
        for i in range(n_part):
            # Iterar solo sobre partículas j > i el resto irá por simetría
            for j in range(i+1, n_part):
                
                # Obtener las posiciones de las partículas i y j en el paso anterior (paso-1)
                pos_i = Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"]
                pos_j = Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"]
                
                # Calcular el vector diferencia y obtenemos su norma (distancia)
                d = np.linalg.norm(pos_j - pos_i)
                
                # Almacenar la distancia en la posición [i,j] (mitad superior)
                distancias[i, j] = d
                
                # Por simetría, almacenar la misma distancia en [j,i] (mitad inferior)
                # Esto completa la matriz (matriz simétrica)
                distancias[j, i] = d
                
        return distancias
    
    def vectores_choque(self,Posiciones_totales, velocidades, i, j,paso):
        """
        Calcula los componentes de velocidad para colisión elástica entre dos partículas.
        
        Descompone las velocidades de las partículas i y j en componentes tangencial
        y normal respecto al vector que las une. Esta descomposición es necesaria para
        aplicar las leyes de conservación en colisiones elásticas.
        
        Parameters
        ----------
        Posiciones_totales : dict
            Diccionario con todas las posiciones: {'particula_i': {'paso_j': array([x, y])}}
        velocidades : dict
            Diccionario con velocidades actuales: {'particula_i': array([vx, vy])}
        i : int
            Índice de la primera partícula en la colisión
        j : int
            Índice de la segunda partícula en la colisión
        paso : int
            Paso temporal actual en la simulación
        
        Returns
        -------
        velocidad_tangencial_i : numpy.ndarray
            Componente de velocidad de partícula i en dirección del vector i→j
        velocidad_tangencial_j : numpy.ndarray
            Componente de velocidad de partícula j en dirección del vector i→j
        velocidad_normal_i : numpy.ndarray
            Componente de velocidad de partícula i perpendicular al vector i→j
        velocidad_normal_j : numpy.ndarray
            Componente de velocidad de partícula j perpendicular al vector i→j
        vector_unitario : numpy.ndarray
            Vector unitario en dirección de i hacia j
        
        Notes
        -----
        En una colisión elástica:
        - Las componentes normales se conservan
        - Las componentes tangenciales se intercambian
        - Este método calcula la descomposición necesaria para ese intercambio
        """
        
        # PASO 1: Calcular el vector que une las dos partículas
        # vector_ij = r_j - r_i (apunta desde la partícula i hacia la partícula j)
        vector_ij = Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"] - Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"]
        
        # PASO 2: Calcular la distancia entre las partículas i y j (módulo del vector)
        distancia_ij = np.linalg.norm(vector_ij)
        
        # PASO 3: Normalizar el vector para obtener la dirección unitaria
        vector_unitario = vector_ij / distancia_ij

        # PASO 4: Calcular componente TANGENCIAL de cada velocidad
        # La componente tangencial es la proyección de la velocidad sobre el vector_unitario
        # Es la velocidad en la dirección de choque
        velocidad_tangencial_j = np.dot(velocidades[f"particula_{j}"], vector_unitario) * vector_unitario
        velocidad_tangencial_i = np.dot(velocidades[f"particula_{i}"], vector_unitario) * vector_unitario
        
        # PASO 5: Calcular componente NORMAL (perpendicular) de cada velocidad
        # Esta componente NO cambia durante la colisión elástica
        velocidad_normal_j = velocidades[f"particula_{j}"] - velocidad_tangencial_j
        velocidad_normal_i = velocidades[f"particula_{i}"] - velocidad_tangencial_i

        # Retornar todos los componentes necesarios para calcular las nuevas velocidades post-colisión
        return velocidad_tangencial_i, velocidad_tangencial_j, velocidad_normal_i, velocidad_normal_j, vector_unitario

    
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
        choques : bool
            True para incluir colisiones entre partículas
            False para solo considerar colisiones con paredes
        optimized : bool
            True para usar algoritmo KDTree (O(N log N), eficiente para muchas partículas)
            False para cálculo directo de todas las distancias (O(N²))
        
        Returns
        -------
        dict
            Diccionario con todas las posiciones en cada paso temporal:
            {'particula_i': {'paso_j': array([x, y])}}
        
        Notes
        -----
        - Solicita al usuario la distribución de velocidades iniciales mediante input()
        - Actualiza self.dinamica_choques o self.dinamica_sin_choques según el caso
        - Actualiza self.velocidades_finales con las velocidades al final
        - Actualiza self.tipo_dinamica con el valor de choques
        - Las colisiones son perfectamente elásticas (conservan energía y momento)
        - Imprime tiempo de procesamiento por paso en cada iteración
        - El algoritmo optimizado usa KDTree para detectar solo pares cercanos
        
        Warnings
        --------
        Esta función solicita input del usuario durante la ejecución, lo cual
        puede no ser deseable para ejecución en batch o testing automatizado.
        
        See Also
        --------
        calcular_distancias_particulas : Método auxiliar para versión no optimizada
        vectores_choque : Calcula componentes de velocidad para colisiones
        """

        # Solicitar al usuario qué distribución de velocidades usar
        v_constante = input("¿Desea que las partículas tengan la misma velocidad inicial? (si/no): ")
        
        # Generar condiciones iniciales (posiciones y velocidades)
        self.inicializar_variables_necesarias(v_constante)

        # Diccionario para almacenar todas las posiciones en cada paso temporal
        # Estructura: {'particula_i': {'paso_j': array([x, y])}}
        Posiciones_totales = {}
        
        # Diccionario para almacenar las velocidades actuales de cada partícula
        # Estructura: {'particula_i': array([vx, vy])}
        velocidades = {}
        
        Fuerzas_totales = {}
        Presion_total = {}
        # Bucle principal sobre todos los pasos temporales
        for paso in range(n_pasos):
            Fuerzas_totales[f"paso_{paso}"] = 0
            # Iterar sobre todas las partículas para actualizar sus posiciones
            for i in range(self.N):
                # --- PASO 0: Inicializar con condiciones iniciales ---
                if paso == 0:
                    # Crear diccionario para almacenar la trayectoria de esta partícula
                    Posiciones_totales[f"particula_{i}"] = {}
                    
                    # Asignar velocidad inicial desde las condiciones iniciales
                    velocidades[f"particula_{i}"] = self.situacion_inicial[f"particula_{i}"]['velocidad']
                    
                    # Asignar posición inicial
                    Posiciones_totales[f"particula_{i}"][f"paso_0"] = self.situacion_inicial[f"particula_{i}"]['posicion']
                    
                # PASOS SIGUIENTES: calcular nuevas posiciones
                else:

                    # Calcular nueva posición usando la velocidad actual y el paso de tiempo
                    Posiciones_totales[f"particula_{i}"][f"paso_{paso}"] = Posiciones_totales[f"particula_{i}"][f"paso_{paso-1}"] + velocidades[f"particula_{i}"] * delta_t
                    
                    # COLISIONES CON PAREDES
                    # Verificar si la partícula choca con alguna pared de la caja
                    for dim in range(2):  # Iterar sobre dimensiones: 0=x, 1=y
                        
                        # Verificar si la partícula toca o atraviesa paredes
                        if Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] <= 0 + self.r_radio or Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] >= self.l_c - self.r_radio:
                            # Invertir componente de velocidad (rebote elástico)
                            velocidades[f"particula_{i}"][dim] *= -1
                            Fuerzas_totales[f"paso_{paso}"] += 2 * self.m *np.abs(velocidades[f"particula_{i}"][dim]) / delta_t
                            # Corregir posición para evitar que la partícula quede fuera de los límites
                            # np.clip asegura que: r_radio <= posición <= l_c - r_radio
                            Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim] = np.clip(
                                Posiciones_totales[f"particula_{i}"][f"paso_{paso}"][dim], 
                                self.r_radio, 
                                self.l_c - self.r_radio
                            )

                        else:
                            continue

            # INICIO DE MEDICIÓN DE TIEMPO DE PROCESAMIENTO
            tstart = timer()
            
            # OPCIÓN 1: ALGORITMO OPTIMIZADO CON KDTREE
            if optimized and choques == True:
                
                if paso != 0:
                    # Convertir posiciones del diccionario a array NumPy para KDTree
                    puntos = np.zeros((self.N, 2))
                    for j in range(self.N):
                        puntos[j, 0] = Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"][0]  # Coordenada x
                        puntos[j, 1] = Posiciones_totales[f"particula_{j}"][f"paso_{paso-1}"][1]  # Coordenada y
                    
                    # Construir árbol KDTree para búsqueda espacial eficiente
                    árbol = spatial.KDTree(puntos)
                    
                    # Buscar todos los pares de partículas a distancia <= 2*r_radio (posible colisión)
                    # query_pairs solo retorna pares únicos (j < k), evitando duplicados
                    pares_cercanos = árbol.query_pairs(r=2*self.r_radio)

                    # Procesar cada par de partículas que están suficientemente cerca
                    for (j, k) in pares_cercanos:

                        # Descomponer velocidades en componentes tangencial y normal
                        velocidad_tangencial_j, velocidad_tangencial_k, velocidad_normal_j, velocidad_normal_k, vector_unitario = self.vectores_choque(Posiciones_totales, velocidades, j, k, paso)

                        # Verificar si las partículas se están acercando (no alejando)
                        if np.dot(velocidades[f"particula_{k}"], vector_unitario) - np.dot(velocidades[f"particula_{j}"], vector_unitario) < 0:
                            # Intercambiar componentes tangenciales (mantener normales)
                            # Velocidad nueva = componente tangencial de la otra + componente normal propia
                            velocidades[f"particula_{j}"] = velocidad_tangencial_k + velocidad_normal_j 
                            velocidades[f"particula_{k}"] = velocidad_tangencial_j + velocidad_normal_k
                        else:
                            continue
                    
                    
            # OPCIÓN 2: ALGORITMO NO OPTIMIZADO (FUERZA BRUTA)
            elif choques == True:
                if paso != 0:
                    # Calcular matriz completa de distancias NxN
                    distancias = self.calcular_distancias_particulas(Posiciones_totales, paso)
                    
                    # Definir umbral de colisión: suma de radios = diámetro
                    umbral = self.r_radio * 2
                    
                    # Iterar sobre todos los pares de partículas (j, k) con j < k
                    for j in range(self.N):
                        for k in range(j + 1, self.N):  # Solo revisar mitad superior para evitar duplicados
                            # Verificar si la distancia es menor o igual al umbral (hay contacto)
                            if distancias[j, k] <= umbral:
                                
                                # Descomponer velocidades en componentes tangencial y normal
                                velocidad_tangencial_j, velocidad_tangencial_k, velocidad_normal_j, velocidad_normal_k, vector_unitario = self.vectores_choque(Posiciones_totales, velocidades, j, k, paso)

                                # Verificar si las partículas se están acercando
                                # (mismo criterio que en versión optimizada)
                                if np.dot(velocidades[f"particula_{k}"], vector_unitario) - np.dot(velocidades[f"particula_{j}"], vector_unitario) < 0:
                                    # Actualizar velocidades después de la colisión elástica
                                    velocidades[f"particula_{j}"] = velocidad_tangencial_k + velocidad_normal_j 
                                    velocidades[f"particula_{k}"] = velocidad_tangencial_j + velocidad_normal_k
                                else:
                                    continue
                            else:
                                continue

            # OPCIÓN 3: SIN COLISIONES ENTRE PARTÍCULAS
            # Solo se consideran colisiones con paredes (ya procesadas arriba)
            else:
                continue

            # Finalizar medición de tiempo y mostrar estadísticas
            tend = timer()
            print(f"Tiempo para procesar colisiones de las {self.N} partículas en el paso {paso}: {tend - tstart} segundos")

            Presion_total[f"paso_{paso}"] = Fuerzas_totales[f"paso_{paso}"] / (4 * self.l_c)  # Presión = Fuerza total / Área de las paredes
        # ALMACENAMIENTO DE RESULTADOS
        # Guardar el tipo de simulación realizada (con o sin colisiones)
        self.tipo_dinamica = choques
        
        # Almacenar resultados en el atributo apropiado según el tipo de simulación
        if choques == True:
            # Simulación CON colisiones entre partículas
            self.dinamica_choques = Posiciones_totales
            self.velocidades_finales = velocidades
            self.Presion_total_choques = Presion_total
            return Posiciones_totales
        
        else:
            # Simulación SIN colisiones entre partículas (solo con paredes)
            self.dinamica_sin_choques = Posiciones_totales
            self.velocidades_finales = velocidades
            self.Presion_total_sin_choques = Presion_total
            return Posiciones_totales

    def generar_animacion(self, distancia_frame, cant_frames,tiempo_frame):
        """
        Genera y muestra una animación del movimiento de las partículas.
        
        Crea una visualización animada del sistema usando los datos de la
        simulación previamente ejecutada. Selecciona automáticamente la fuente
        de datos según el tipo de simulación realizada (con o sin colisiones).
        
        Parameters
        ----------
        distancia_frame : int
            Número de pasos temporales entre frames consecutivos de la animación
            (permite acelerar la animación saltando pasos)
        cant_frames : int
            Número total de frames en la animación
        tiempo_frame : float
            Duración de cada frame en segundos
        
        Returns
        -------
        None
            Muestra la animación en pantalla
        
        Notes
        -----
        - Requiere haber ejecutado simular_dinamica() previamente
        - Usa self.tipo_dinamica para determinar qué datos visualizar:
          * Si False: visualiza self.dinamica_sin_choques
          * Si True: visualiza self.dinamica_choques
        - La animación se repite indefinidamente (repeat=True)
        - Actualiza self.animacion con el objeto FuncAnimation
        - El tamaño de la figura es 10x10 pulgadas
        - Las partículas se muestran como círculos (markersize=10)
        - Los límites de los ejes se ajustan al tamaño de la caja (l_c)
        
        Raises
        ------
        AttributeError
            Si no se ha ejecutado simular_dinamica() antes (falta self.tipo_dinamica)
        KeyError
            Si no existen los datos de simulación correspondientes
        
        See Also
        --------
        simular_dinamica : Debe ejecutarse antes para generar los datos de posición
        """
        # CONFIGURACIÓN DE LA FIGURA Y EJES
        # Crear figura cuadrada de 10x10 pulgadas para visualizar la simulación
        fig, ax = plt.subplots(figsize=(10,10))
        
        # Configurar límites de los ejes para que coincidan con el tamaño de la caja
        ax.set_xlim(0, self.l_c)  # Límite en x: [0, l_c]
        ax.set_ylim(0, self.l_c)  # Límite en y: [0, l_c]
        
        # Mantener relación de aspecto 1:1 (cuadrado perfecto, sin distorsión)
        ax.set_aspect('equal')
        
        # CREAR OBJETOS GRÁFICOS PARA LAS PARTÍCULAS
        # Lista para almacenar los objetos Line2D que representan cada partícula
        Particulas = []
        for i in range(self.N):
            # Crear un objeto de ploteo para cada partícula (inicialmente vacío)
            # 'o' = marcador circular, markersize=10 define el tamaño visual
            particula, = ax.plot([], [], 'o', markersize=10)
            Particulas.append(particula)

        # FUNCIÓN DE INICIALIZACIÓN DE LA ANIMACIÓN
        def init():
            """
            Función llamada al inicio de la animación para configurar el estado inicial.
            Limpia todos los datos de las partículas antes de comenzar.
            """
            for particula in Particulas:
                particula.set_data([], [])  # Establecer posiciones vacías
            return Particulas

        # SELECCIÓN DE FUENTE DE DATOS SEGÚN TIPO DE SIMULACIÓN
        
        # CASO 1: Simulación SIN colisiones entre partículas
        if self.tipo_dinamica == False:
            def animate(frame):
                """
                Función de actualización de animación para dinámica sin choques.
                
                Parameters
                ----------
                frame : int
                    Número de frame actual (0 a cant_frames-1)
                """
                # Calcular qué paso temporal corresponde a este frame
                # Ejemplo: si distancia_frame=10 y frame=5, mostramos paso 50
                paso = frame * distancia_frame
                
                # Actualizar posición de cada partícula
                for i, particula in enumerate(Particulas):
                    # Extraer coordenadas x e y del diccionario de posiciones
                    x = self.dinamica_sin_choques[f"particula_{i}"][f"paso_{paso}"][0]
                    y = self.dinamica_sin_choques[f"particula_{i}"][f"paso_{paso}"][1]
                    particula.set_data([x], [y])
                return Particulas
        
        # CASO 2: Simulación CON colisiones entre partículas
        else:
            def animate(frame):
                """
                Función de actualización de animación para dinámica con choques.
                
                Parameters
                ----------
                frame : int
                    Número de frame actual (0 a cant_frames-1)
                """
                # Calcular paso temporal correspondiente a este frame
                paso = frame * distancia_frame
                
                # Actualizar posición de cada partícula desde el diccionario con choques
                for i, particula in enumerate(Particulas):
                    x = self.dinamica_choques[f"particula_{i}"][f"paso_{paso}"][0]
                    y = self.dinamica_choques[f"particula_{i}"][f"paso_{paso}"][1]
                    particula.set_data([x], [y])
                return Particulas
        
        # CREAR Y EJECUTAR LA ANIMACIÓN
        # FuncAnimation gestiona la animación llamando a animate() repetidamente
        self.animacion = FuncAnimation(
            fig,                          # Figura donde se dibuja
            animate,                      # Función que actualiza cada frame
            init_func=init,              # Función de inicialización
            frames=cant_frames,          # Número total de frames
            blit=True,                   # Optimización: solo redibujar lo que cambió
            interval=tiempo_frame*1000,  # Tiempo entre frames en milisegundos
            repeat=True                  # Repetir animación indefinidamente
        )
        
        # Mostrar la animación en una ventana emergente
        plt.show()
    
    def histograma_energia(self):
        """
        Genera histograma de energías cinéticas finales y verifica conservación.
        
        Calcula las energías cinéticas de todas las partículas al final de
        la simulación, muestra su distribución y verifica la conservación de
        energía total si se simuló con colisiones.
        
        Returns
        -------
        None
            Muestra el histograma y opcionalmente imprime energías totales
        
        Notes
        -----
        - Actualiza self.E_f con las energías cinéticas finales
        - El histograma usa 25 bins en el rango [0, 1]
        - Si self.tipo_dinamica es True (hubo colisiones), compara energía total inicial vs final
        - Para colisiones elásticas, la energía debería conservarse
        - Requiere haber ejecutado simular_dinamica() previamente
        
        Raises
        ------
        AttributeError
            Si no se ha ejecutado simular_dinamica() antes (falta self.velocidades_finales)
        
        See Also
        --------
        comprobar_T : Verifica conservación de temperatura
        simular_dinamica : Debe ejecutarse antes para generar velocidades_finales
        """
        # CÁLCULO DE ENERGÍAS CINÉTICAS FINALES
        # Diccionario para almacenar la energía cinética de cada partícula
        Energias = {}
        
        # Iterar sobre todas las partículas para calcular su energía cinética final
        for particulas in range(self.N):
            # Obtener el vector velocidad final de la partícula
            velocidad_final = self.velocidades_finales[f"particula_{particulas}"]
            
            # Calcular energía cinética
            energia_cinetica = 0.5 * self.m * (np.linalg.norm(velocidad_final))**2
            
            # Almacenar en el diccionario
            Energias[f"particula_{particulas}"] = energia_cinetica
        
        # Convertir diccionario a array NumPy para facilitar operaciones y graficación
        energias_array = np.array(list(Energias.values()))
        
        # Guardar las energías finales como atributo de la clase
        self.E_f = Energias
        
        # GENERACIÓN DEL HISTOGRAMA
        # Crear nueva figura de 8x6 pulgadas
        plt.figure(figsize=(8,6))
        
        # Generar histograma con 25 bins en el rango [0, 1]
        # edgecolor='black' añade bordes negros a las barras para mejor visualización
        plt.hist(energias_array, bins=25, edgecolor='black', range=(0, 1))
        
        # Añadir título y etiquetas a los ejes
        plt.title('Histograma de Energías Cinéticas Finales')
        plt.xlabel('Energía Cinética')
        plt.ylabel('Número de Partículas')
        
        # Mostrar el gráfico
        plt.show()
        
        # VERIFICACIÓN DE CONSERVACIÓN DE ENERGÍA
        # Solo verificar conservación si hubo colisiones entre partículas
        if self.tipo_dinamica == True:
            # Calcular energía total final sumando todas las energías individuales
            Energia_total_final = np.sum(energias_array)

            # Inicializar arrays para almacenar velocidades y energías iniciales
            velocidad_inicial = np.zeros((self.N, 2))
            Energias_iniciales = np.zeros(self.N)
            
            # Calcular energías cinéticas iniciales para cada partícula
            for particulas in range(self.N):
                # Obtener velocidad inicial desde las condiciones iniciales guardadas
                velocidad_inicial[particulas] = self.situacion_inicial[f"particula_{particulas}"]['velocidad']
                
                # Calcular energía cinética inicial
                Energias_iniciales[particulas] = 0.5 * self.m * (np.linalg.norm(velocidad_inicial[particulas]))**2
            
            # Sumar todas las energías iniciales para obtener energía total inicial
            Energias_iniciales_total = np.sum(Energias_iniciales)
            
            # Imprimir comparación de energías (deberían ser iguales en colisiones elásticas)
            print("\n")
            print("="*100)
            print("Comparación de energía cinética total inicial vs final:")
            print("="*100)
            print(f"Energía Cinética Total Inicial: {Energias_iniciales_total}")
            print(f"Energía Cinética Total Final: {Energia_total_final}")
            
        return None


    def comprobar_T(self,Kb=0.01):
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
        
        # CÁLCULO DE ENERGÍA TOTAL INICIAL
        # Inicializar acumulador para la suma de energías iniciales
        Prom_energia_inicial = 0
        
        # Inicializar acumulador para la suma de energías finales
        Prom_energia_final = 0
        
        # Sumar las energías cinéticas de todas las partículas
        for rango in range(self.N):
            # Acumular energía inicial de cada partícula
            Prom_energia_inicial += self.E_0[f"particula_{rango}"]
            
            # Acumular energía final de cada partícula
            Prom_energia_final += self.E_f[f"particula_{rango}"]

        # CÁLCULO DE TEMPERATURA USANDO TEOREMA DE EQUIPARTICIÓN
        # Temperatura inicial calculada desde energías iniciales
        T_inicial = Prom_energia_inicial /  self.N / Kb
        
        # Temperatura final calculada desde energías finales
        T_final = Prom_energia_final /  self.N / Kb
        
        # IMPRIMIR RESULTADOS
        # Mostrar comparación de temperaturas
        # En un sistema con colisiones elásticas, T_inicial = T_final
        print("\n")
        print("="*100)
        print("Comprobación de temperaturas inicial y final:")
        print("="*100)
        print(f"Temperatura inicial: {T_inicial}")
        print(f"Temperatura final: {T_final}")
        print("\n")
        self.T_final = T_final
        self.T_inicial = T_inicial

        return None
    
    def comprobar_presiones(self,n_pasos,pasos_promedio=1000):
        """

        """
        Presiones_choques_array = np.zeros(n_pasos)
        Presiones_sin_choques_array = np.zeros(n_pasos)
        fig = plt.figure(figsize=(8,6))
        plt.title("Presión total del sistema a lo largo de la simulación")
        plt.xlabel("Paso temporal")
        plt.ylabel("Presión total")

        for paso in range(n_pasos):

            if self.tipo_dinamica == True:
                Presiones_choques_array[paso] = self.Presion_total_choques[f"paso_{paso}"]
            
            else:
                Presiones_sin_choques_array[paso] = self.Presion_total_sin_choques[f"paso_{paso}"]
        
        Presiones_choques_promediadas = 0
        Presiones_sin_choques_promediadas = 0
        for paso in range(0, n_pasos):
            if self.tipo_dinamica == True:
                Presiones_choques_promediadas += Presiones_choques_array[paso]
            else:
                Presiones_sin_choques_promediadas += Presiones_sin_choques_array[paso]
                
            if paso % pasos_promedio == pasos_promedio - 1:
                if self.tipo_dinamica == True:
                    Presiones_choques_promediadas /= pasos_promedio            
                    plt.scatter(paso, Presiones_choques_promediadas, color='blue', label='Con choques')
                    Presiones_choques_promediadas = 0
                else:
                    Presiones_sin_choques_promediadas /= pasos_promedio            
                    plt.scatter(paso, Presiones_sin_choques_promediadas, color='orange', label='Sin choques')
                    Presiones_sin_choques_promediadas = 0
        plt.grid()
        plt.show()
        if self.tipo_dinamica == True:
            self.Presiones_choques_prom = np.mean(Presiones_choques_array)
            print("="*100)
            print(f"Presión promedio con choques: ")
            print("="*100)
            print(f"resultado: {self.Presiones_choques_prom}")
            print("\n")
        else:
            self.Presiones_sin_choques_prom = np.mean(Presiones_sin_choques_array)
            print("="*100)
            print(f"Presión promedio sin choques: ")
            print("="*100)
            print(f"resultado: {self.Presiones_sin_choques_prom}")
            print("\n")
        
        return None

    def comprobar_ley_gas_ideal(self,Kb=0.01):
        """

        """
        print("="*100)
        print("Comprobación de la ley de gas ideal: PA/NKbT = 1")
        print("="*100)
        if self.tipo_dinamica == True:
            P = self.Presiones_choques_prom
        else:
            P = self.Presiones_sin_choques_prom
        A = self.l_c**2
        N = self.N
        T = self.T_final
        resultado = P * A / (N * Kb * T)
        print(f"Resultado de PA/NKbT: {resultado}")
        print("\n")
        return None
        
    def comprobar_velocidades_finales(self):
        """
        Compara velocidades iniciales y finales para diagnóstico.
        
        Función de utilidad para debugging que compara elemento a elemento
        las velocidades antes y después de la simulación, Ha sido creada con el simple objetivo de verificar cambios.
        
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
        # Inicializar arreglos para almacenar velocidades finales e iniciales
        velocidad_final = np.zeros((self.N,2))
        velocidad_inicial = np.zeros((self.N,2))
        
        # Iterar sobre todas las partículas del sistema
        for particula in range(self.N):
            # Extraer la velocidad final de la partícula actual del diccionario de resultados
            velocidad_final[particula] = self.velocidades_finales[f"particula_{particula}"]
            
            # Extraer la velocidad inicial de la partícula actual del diccionario de condiciones iniciales
            velocidad_inicial[particula] = self.situacion_inicial[f"particula_{particula}"]['velocidad']
            
            # Comparar elemento a elemento las velocidades y mostrar el resultado booleano
            # True: la componente no cambió, False: la componente cambió
            print(velocidad_final == velocidad_inicial)
            
        return None



Dinamica = dinamica_particulas_confinada_2D(N_particulas=100, radio_particula=0.125, l_caja=10.0, v_0=1.0)
Dinamica.simular_dinamica(n_pasos=10000, delta_t=0.01, choques= True,optimized = True)

Dinamica.generar_animacion(10,1000,0.01)
Dinamica.histograma_energia()
Dinamica.comprobar_T()
Dinamica.comprobar_presiones(10000)
Dinamica.comprobar_ley_gas_ideal()


