#!/usr/bin/env python3
"""
Solver v6: Agente Híbrido Dinámico (Óptimo + Heurístico)
Versión corregida: uso de API key desde entorno, robustez en flow.run, logs y validaciones.
"""
#from workflows.errors import WorkflowTimeoutError
import asyncio
import os
import math
import itertools
import time
import argparse
import logging
import json
from typing import List, Tuple, Iterable, Optional
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step

# -----------------------
# Parámetros por defecto (ajustables vía CLI)
# -----------------------
entrada = "/home/edgar/GitHub/Proyectos/Proyectos_python/Hackathon_agenteia/Formación_ia/Formación completa LLMS_V2(1)/RAG_AMPLIACION_EJEMPLOS_REDUCIDA/entradas/1b.txt"
ncont = 4

def parse_args():
    p = argparse.ArgumentParser(description="Solver Contenedores v6 (Híbrido)")
    p.add_argument("--input", "-i", default=entrada, help="Archivo de entrada")
    p.add_argument("--output-dir", "-o", default="salidas", help="Directorio de salida")
    p.add_argument("--N", type=int, default=ncont, help="Número de contenedores a colocar")
    p.add_argument("--umbral-combinaciones", type=int, default=10_000_000, help="Umbral para decidir fuerza bruta")
    p.add_argument("--base-candidatos", type=int, default=50, help="Base candidatos heurística")
    p.add_argument("--candidatos-por-contenedor", type=int, default=15, help="Candidatos por contenedor (heurística)")
    p.add_argument("--timeout-ms", type=int, default=15000, help="Timeout del flow/LLM en ms")
    p.add_argument("--model", default="gemini-2.5-flash", help="Modelo Gemini a usar")
    p.add_argument("--use-llm-for-formatting", action="store_true", help="Si se quiere usar LLM para formateo de fuerza bruta (opcional)")
    p.add_argument("--formating-attempts", type=int, default=3, help="Reintentos del LLM para formateo")
    p.add_argument("--max-intentos-heuristica", type=int, default=3, help="Número de reintentos para el agente heurístico (generar JSON)")
    p.add_argument("--max-eval", type=int, default=None, help="Límite de evaluaciones en fuerza bruta (safety)")
    return p.parse_args()

# -----------------------
# Logging / métricas
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("solver")

metrics = {
    "time_read": 0.0,
    "time_analysis": 0.0,
    "time_bruteforce": 0.0,
    "time_llm_format": 0.0,
    "time_heuristic": 0.0,
    "llm_calls": 0,
    "combinaciones_evaluadas": 0,
}

# -----------------------
# Utilidades y reglas (Python)
# -----------------------
DELTAS_8_DIR = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0),  ( 1, 1)]

def encontrar_posiciones_candidatas(mapa: List[List[str]]) -> List[Tuple[int,int]]:
    if not mapa or not mapa[0]: return []
    filas, cols = len(mapa), len(mapa[0])
    posiciones_validas = set()
    for r in range(filas):
        for c in range(cols):
            if mapa[r][c] != '-': continue
            for dr, dc in DELTAS_8_DIR:
                nr, nc = r + dr, c + dc
                if 0 <= nr < filas and 0 <= nc < cols and mapa[nr][nc] == 'X':
                    posiciones_validas.add((r, c)); break
    return list(posiciones_validas)

def distancia_manhattan(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
    (r1, c1), (r2, c2) = p1, p2
    return abs(r1 - r2) + abs(c1 - c2)

def calcular_distancia_total(casas: Iterable[Tuple[int,int]], contenedores_actuales: Iterable[Tuple[int,int]]) -> float:
    contenedores_actuales = list(contenedores_actuales)
    if not contenedores_actuales: return float('inf')
    total = 0
    for casa in casas:
        total += min(distancia_manhattan(casa, cont) for cont in contenedores_actuales)
    return total

def ponderar_posiciones(mapa: List[List[str]], posiciones_candidatas: List[Tuple[int,int]]) -> List[Tuple[int, Tuple[int,int]]]:
    filas, cols = len(mapa), len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    if not casas: return [(0, pos) for pos in posiciones_candidatas]
    puntuaciones = []
    for pos in posiciones_candidatas:
        puntuaciones.append((sum(distancia_manhattan(pos, casa) for casa in casas), pos))
    puntuaciones.sort(key=lambda item: item[0])
    return puntuaciones

def validar_solucion_python(mapa_original_lista: List[List[str]], mapa_generado_str: str, num_c_esperado: int) -> bool:
    if not mapa_generado_str:
        logger.debug("Validación: Mapa vacío o None.")
        return False
    mapa_gen_lineas = [line for line in mapa_generado_str.splitlines() if line.strip()]
    if not mapa_gen_lineas: logger.debug("Validación: Mapa vacío (filtrado)."); return False
    filas_orig, cols_orig = len(mapa_original_lista), len(mapa_original_lista[0])
    if len(mapa_gen_lineas) != filas_orig:
        logger.debug(f"Validación: Filas {len(mapa_gen_lineas)} != {filas_orig}."); return False
    mapa_gen_lista = []
    for r in range(filas_orig):
        linea = list(mapa_gen_lineas[r])
        if len(linea) != cols_orig:
            logger.debug(f"Validación: Cols {len(linea)} != {cols_orig} en fila {r}."); return False
        mapa_gen_lista.append(linea)
    count_c = 0
    for r in range(filas_orig):
        for c in range(cols_orig):
            celda_orig, celda_gen = mapa_original_lista[r][c], mapa_gen_lista[r][c]
            if celda_gen == 'C':
                count_c += 1
                if celda_orig != '-':
                    logger.debug(f"Validación: 'C' sobre '{celda_orig}' en ({r},{c})."); return False
                adyacente_a_X = False
                for dr, dc in DELTAS_8_DIR:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < filas_orig and 0 <= nc < cols_orig and mapa_original_lista[nr][nc] == 'X':
                        adyacente_a_X = True; break
                if not adyacente_a_X:
                    logger.debug(f"Validación: 'C' no adyacente a 'X' en ({r},{c})."); return False
            elif celda_gen != celda_orig:
                logger.debug(f"Validación: Celda ({r},{c}) modificada."); return False
    if count_c != num_c_esperado:
        logger.debug(f"Validación: {count_c} 'C' encontrados, se esperaban {num_c_esperado}."); return False
    return True

def format_map_from_coords(mapa_original: List[List[str]], coords: Iterable[Tuple[int,int]]) -> str:
    """Formatea localmente el mapa (100% fiable). Si coords contienen listas las convertimos."""
    mapa = [row[:] for row in mapa_original]
    for rc in coords:
        try:
            r, c = int(rc[0]), int(rc[1])
        except Exception:
            logger.warning(f"Formateador: coordenada inválida {rc}; la ignoro.")
            continue
        if 0 <= r < len(mapa) and 0 <= c < len(mapa[0]):
            mapa[r][c] = 'C'
        else:
            logger.warning(f"Formateador: Coordenada {(r,c)} fuera de límites, ignorada.")
    return '\n'.join(''.join(row) for row in mapa)

# -----------------------
# Algoritmos Heurísticos Locales (Fallback)
# -----------------------
def greedy_select(mapa: List[List[str]], candidatos: List[Tuple[int,int]], N: int) -> List[Tuple[int,int]]:
    filas = len(mapa); cols = len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    seleccion: List[Tuple[int,int]] = []
    candidatos_disponibles = candidatos.copy()
    for _ in range(min(N, len(candidatos_disponibles))):
        best = None; best_score = float('inf')
        for c in candidatos_disponibles:
            trial = seleccion + [c]
            dist = calcular_distancia_total(casas, trial)
            if dist < best_score:
                best_score = dist; best = c
        if best is None: break
        seleccion.append(best); candidatos_disponibles.remove(best)
    return seleccion

def hill_climb_swap(mapa: List[List[str]], current_coords: List[Tuple[int,int]], pool: List[Tuple[int,int]], max_iters: int = 200) -> List[Tuple[int,int]]:
    filas = len(mapa); cols = len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    current = current_coords.copy()
    pool_set = set(pool) - set(current)
    best_score = calcular_distancia_total(casas, current)
    it = 0; improved = True
    while improved and it < max_iters:
        improved = False; it += 1
        for i, sel in enumerate(current):
            for cand in list(pool_set):
                candidate_list = current.copy()
                candidate_list[i] = cand
                score = calcular_distancia_total(casas, candidate_list)
                if score < best_score:
                    best_score = score
                    pool_set.add(sel); pool_set.remove(cand)
                    current[i] = cand; improved = True; break
            if improved: break
    return current

# -----------------------
# Fuerza bruta (iterador)
# -----------------------
def encontrar_solucion_optima(mapa_lista: List[List[str]], posiciones_validas: List[Tuple[int,int]], num_c_a_colocar: int, max_evaluaciones: Optional[int]=None) -> Optional[Tuple[Tuple[int,int], ...]]:
    logger.info("Iniciando optimización por fuerza bruta (generador).")
    t0 = time.perf_counter()
    filas, cols = len(mapa_lista), len(mapa_lista[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa_lista[i][j] == 'O']
    if not casas:
        logger.warning("No se encontraron casas. Colocando en las primeras posiciones válidas.")
        return tuple(posiciones_validas[:num_c_a_colocar])

    best_distance = float('inf'); best_combo = None; evaluated = 0
    combos_iter = itertools.combinations(posiciones_validas, num_c_a_colocar)
    for combo in combos_iter:
        evaluated += 1; metrics["combinaciones_evaluadas"] += 1
        dist = calcular_distancia_total(casas, combo)
        if dist < best_distance:
            best_distance = dist; best_combo = combo
            if best_distance == 0: break
        if max_evaluaciones is not None and evaluated >= max_evaluaciones:
            logger.info(f"Parando búsqueda tras {evaluated} evaluaciones (límite)."); break

    metrics["time_bruteforce"] += time.perf_counter() - t0
    logger.info(f"Fuerza bruta finalizada. Evaluadas: {evaluated}. Mejor distancia: {best_distance}")
    return best_combo

# -----------------------
# Workflow V6: LLM-first (Razonamiento) + Python (Formateo/Fallback)
# -----------------------
class MyStartEvent(StartEvent):
    query: str
    mapa_original_lista: list
    num_c: int
    posiciones_validas: List[Tuple[int,int]]
    lista_ponderada: List[Tuple[int, Tuple[int,int]]]

class CoordenadasEvent(Event):
    coordenadas_elegidas: list
    mapa_original_lista: list
    num_c: int
    posiciones_validas: List[Tuple[int,int]]
    lista_ponderada: List[Tuple[int, Tuple[int,int]]]

class SolucionEvent(Event):
    salida: str
    mapa_original_lista: list

class ProblemaFlow(Workflow):
    llm = None
    max_intentos = 3

    @step
    async def generar_lista_coordenadas(self, ev: MyStartEvent) -> CoordenadasEvent:
        logger.info("WF: Entró en 'generar_lista_coordenadas' (LLM-first)")
        prompt_llm = f"""
{ev.query}

---
TAREA: Elige el MEJOR conjunto de {ev.num_c} posiciones de la lista ponderada del prompt anterior.
Devuelve SOLO una lista JSON: [[r1,c1],[r2,c2],...,[rN,cN]]
"""
        intentos = 1
        while intentos <= self.max_intentos:
            logger.info(f"Intento LLM {intentos}/{self.max_intentos} (generar coords)")
            metrics["llm_calls"] += 1
            texto = ""
            try:
                respuesta = await self.llm.acomplete(prompt_llm)
                texto = respuesta.text.strip().replace("`", "")
                # intento de extraer JSON puro (puede venir con texto)
                # buscamos primera aparicion de '[' y la última ']' para intentar parsear
                first = texto.find('[')
                last = texto.rfind(']')
                if first != -1 and last != -1 and last > first:
                    candidate = texto[first:last+1]
                else:
                    candidate = texto
                coords = json.loads(candidate)
                # normalizar coords a listas de enteros
                if isinstance(coords, list) and len(coords) == ev.num_c:
                    coords_norm = []
                    ok = True
                    for item in coords:
                        if not (isinstance(item, (list,tuple)) and len(item) == 2):
                            ok = False; break
                        coords_norm.append([int(item[0]), int(item[1])])
                    if ok:
                        logger.info("LLM devolvió JSON con la longitud esperada.")
                        return CoordenadasEvent(coordenadas_elegidas=coords_norm,
                                                mapa_original_lista=ev.mapa_original_lista,
                                                num_c=ev.num_c,
                                                posiciones_validas=ev.posiciones_validas,
                                                lista_ponderada=ev.lista_ponderada)
                    else:
                        logger.warning("JSON parseado no tiene pares [r,c] correctos.")
                else:
                    logger.warning("LLM devolvió JSON pero con longitud distinta o tipo no esperado.")
            except Exception as e:
                logger.warning(f"Fallo al parsear JSON del LLM: {e}. Texto (trunc): {texto[:300]}")
            intentos += 1
            prompt_llm += "\nRECUERDA: devuelve SOLO la lista JSON [[r,c],...]."
        logger.error("LLM no pudo generar una lista JSON válida tras los intentos.")
        return CoordenadasEvent(coordenadas_elegidas=[], mapa_original_lista=ev.mapa_original_lista, num_c=ev.num_c, posiciones_validas=ev.posiciones_validas, lista_ponderada=ev.lista_ponderada)

    @step
    def formatear_y_validar(self, ev: CoordenadasEvent) -> SolucionEvent:
        """Paso 2 (Python): Formatea la lista del LLM o activa el fallback local.
        Ahora: si el LLM devuelve coords válidas, comparamos:
        - solución LLM (formateada y validada)
        - solución local (greedy + hill-climb)
        y elegimos la que tenga menor suma de distancias (total_pasos).
        """
        logger.info("WF: Entró en 'formatear_y_validar' (Python) - comparación LLM vs greedy")

        mapa_final_str = ""
        valid_set = set(ev.posiciones_validas)

        # Preparar lista de casas para calcular distancias
        filas = len(ev.mapa_original_lista)
        cols = len(ev.mapa_original_lista[0])
        casas = [(i, j) for i in range(filas) for j in range(cols) if ev.mapa_original_lista[i][j] == 'O']

        # VALIDACIÓN ESTRUCTURAL de las coords LLM (tipo y pertenencia al conjunto válido)
        llm_coords_ok_structure = (
            isinstance(ev.coordenadas_elegidas, list) and
            len(ev.coordenadas_elegidas) == ev.num_c and
            all(isinstance(c, list) and len(c) == 2 and tuple(c) in valid_set for c in ev.coordenadas_elegidas)
        )

        # Transformar coords LLM a lista de tuplas si pasa la validación estructural
        llm_coords_tuples: List[Tuple[int,int]] = []
        if llm_coords_ok_structure:
            llm_coords_tuples = [ (int(c[0]), int(c[1])) for c in ev.coordenadas_elegidas ]

        # 1) Si LLM propone coords estructuralmente válidas, formateamos y validamos el mapa resultante
        llm_map_str = ""
        llm_validated = False
        if llm_coords_tuples:
            llm_map_str = format_map_from_coords(ev.mapa_original_lista, llm_coords_tuples)
            llm_validated = validar_solucion_python(ev.mapa_original_lista, llm_map_str, ev.num_c)
            if not llm_validated:
                logger.warning("Las coordenadas del LLM pasan estructura pero la VALIDACIÓN del mapa ha fallado.")
            else:
                logger.info("Coordenadas LLM válidas y mapa LLM pasó la validación.")
        mapa_final_str = llm_map_str

        return SolucionEvent(
            salida=mapa_final_str,
            mapa_original_lista=ev.mapa_original_lista
        )


    @step
    async def finalizar(self, ev: SolucionEvent) -> StopEvent:
        logger.info("WF: Entró en 'finalizar' -> StopEvent")
        return StopEvent(result=ev.salida)

# -----------------------
# Main / Orquestador
# -----------------------
async def main_async(args):
    t_start = time.perf_counter()

    # LLM setup
    api_key = "AIzaSyDbWiFK8ia8xbhgmfODNtvrCh5LbVSg7RU"
    if not api_key:
        raise ValueError("La variable de entorno 'GOOGLE_API_KEY' no está configurada. Exporta GOOGLE_API_KEY y vuelve a ejecutar.")
    local_llm = GoogleGenAI(model=args.model, api_key=api_key)
    Settings.llm = local_llm

    # Lectura
    t0 = time.perf_counter()
    ruta_entrada = args.input
    try:
        with open(ruta_entrada, "r") as f:
            mapa_lista = [list(line.rstrip("\n")) for line in f.readlines() if line.strip() != ""]
            
            # suponiendo que tenemos una matriz cuadrada
            numero_filas = len(mapa_lista)
            numero_columnas = len(mapa_lista[0])
            densidades_dict = {}
            min_tamaño = 3
            total_casas = sum(cell == 'O' for row in mapa_lista for cell in row)
            if numero_filas == numero_columnas:
                min_tamaño = 3
                tamaños_matriz = []
                
                for i in range(1, numero_filas+1):
                    if numero_filas % i == 0 and i >= min_tamaño:
                        tamaños_matriz.append(i)

                total_casas = sum(cell == 'O' for row in mapa_lista for cell in row)

                for x in tamaños_matriz:
                    densidades = []
                    step = x  # tamaño de submatriz
                    for i in range(0, numero_filas, step):
                        for j in range(0, numero_columnas, step):
                            # extraemos la submatriz
                            sub = [row[j:j+step] for row in mapa_lista[i:i+step]]
                            # contamos casas en la submatriz
                            casas_sub = sum(cell == 'O' for row_sub in sub for cell in row_sub)
                            densidad = casas_sub / total_casas if total_casas > 0 else 0
                            densidades.append(densidad)
                    densidades_dict[x] = densidades
            else:
                # Matriz rectangular: subdivisión adaptada
                # Posibles tamaños de submatriz para filas y columnas
                tamaños_filas = [i for i in range(min_tamaño, numero_filas+1) if numero_filas % i == 0]
                tamaños_columnas = [j for j in range(min_tamaño, numero_columnas+1) if numero_columnas % j == 0]

                for alto in tamaños_filas:
                    for ancho in tamaños_columnas:
                        densidades = []
                        for i in range(0, numero_filas, alto):
                            for j in range(0, numero_columnas, ancho):
                            # Extraemos la submatriz
                                sub = [row[j:j+ancho] for row in mapa_lista[i:i+alto]]
                                # Contamos casas en la submatriz
                                casas_sub = sum(cell == 'O' for row_sub in sub for cell in row_sub)
                                densidad = casas_sub / total_casas if total_casas > 0 else 0
                                densidades.append(densidad)
                        densidades_dict[(alto, ancho)] = densidades
            print(densidades_dict)


    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de entrada en {ruta_entrada}"); return
    metrics["time_read"] += time.perf_counter() - t0

    if not mapa_lista: logger.error("El mapa está vacío."); return

    mapa_str = '\n'.join(''.join(fila) for fila in mapa_lista)
    t_analysis = time.perf_counter()
    posiciones_validas = encontrar_posiciones_candidatas(mapa_lista)
    metrics["time_analysis"] += time.perf_counter() - t_analysis

    if not posiciones_validas: logger.error("No se encontraron posiciones válidas."); return

    k = len(posiciones_validas)
    N = args.N
    try:
        num_combos = math.comb(k, N) if k >= N else 0
    except Exception:
        num_combos = float('inf')

    logger.info("--- ANÁLISIS ---")
    logger.info(f"Posiciones candidatas (k): {k}")
    logger.info(f"Contenedores a colocar (N): {N}")
    logger.info(f"Combinaciones totales (k choose N): {num_combos:,}")

    resultado_texto = ""
    solucion_es_optima = False

    # ROUTER
    if num_combos > 0 and num_combos <= args.umbral_combinaciones:
        logger.info(f"Estrategia: Fuerza Bruta (Combinaciones <= {args.umbral_combinaciones:,}).")
        mejor_coords = encontrar_solucion_optima(mapa_lista, posiciones_validas, N, max_evaluaciones=args.max_eval)
        solucion_es_optima = True
        if not mejor_coords:
            logger.error("La fuerza bruta no encontró una solución."); return
        resultado_texto = format_map_from_coords(mapa_lista, mejor_coords)

        if args.use_llm_for_formatting:
            logger.info("Usando LLM para formateo (opcional)...")
            prompt_formateo = f"""
TAREA: Coloca el carácter 'C' en estas coordenadas del mapa.
Debes cambiar SOLO las posiciones indicadas a 'C' y NADA MÁS. Mantén el formato original.
MAPA ORIGINAL:
{mapa_str}
COORDENADAS:
{list(mejor_coords)}
Responde SOLO con el mapa modificado, sin explicaciones.
"""
            for i in range(args.formating_attempts):
                metrics["llm_calls"] += 1
                t_llm = time.perf_counter()
                try:
                    resp = await local_llm.acomplete(prompt_formateo)
                    candidate = resp.text.strip()
                except Exception as e:
                    logger.error(f"LLM formatting error: {e}"); candidate = ""
                metrics["time_llm_format"] += time.perf_counter() - t_llm

                if candidate and validar_solucion_python(mapa_lista, candidate, N):
                    resultado_texto = candidate
                    logger.info("LLM ha formateado correctamente."); break
                else:
                    logger.warning("LLM formateó mal la salida; reintentando (fallback local ya está listo).")

    else:
        logger.info(f"Estrategia: Heurística con Workflow LLM (Combinaciones > {args.umbral_combinaciones:,}).")
        max_candidatos_heuristicos = args.base_candidatos + (N * args.candidatos_por_contenedor)
        logger.info(f"Calculando lista ponderada (Top {max_candidatos_heuristicos}) para el agente...")
        lista_ponderada = ponderar_posiciones(mapa_lista, posiciones_validas)
        top_candidatos = lista_ponderada[:max_candidatos_heuristicos]
        texto_ponderado = "\n".join([f"  - Score: {score}, Pos: {pos}" for score, pos in top_candidatos])

        query = f"""
Tengo un mapa:
{mapa_str}

Objetivo: Colocar {N} contenedores 'C' para minimizar la distancia total desde todas las casas.

Reglas OBLIGATORIAS:
1. Contenedor SOLO en posiciones '-'
2. Cada contenedor debe estar ADYACENTE (en las 8 direcciones) de al menos una 'X'
3. Minimizar la suma de pasos desde todas las casas 'O' a su contenedor más cercano.
4. NO modificar el mapa excepto para colocar las 'C'

ANÁLISIS (Python): El problema es demasiado complejo para fuerza bruta.
Estas son las {len(top_candidatos)} mejores posiciones válidas (ordenadas de mejor a peor):
{texto_ponderado}

IMPORTANTE:
- Usa esta lista como guía principal.
- Elige el *conjunto* de {N} posiciones que MEJOR cobertura den JUNTAS.
- No te limites a coger las {N} primeras de la lista (podrían estar agrupadas).
- Te doy además un recurso que es una diccionario de listas de densidades por tamaños de submatrices {densidades_dict} para que puedas analizar zonas con más casas 'O'.
- Las submatrices van de izquierda a derecha y luego de arriba a abajo.
- Trata de poner contenedores allí donde la densidad de casas 'O' sea mayor.
"""

        flow = ProblemaFlow(timeout=max(450, int(args.timeout_ms/1000)), verbose=True)
        flow.llm = local_llm
        flow.max_intentos = args.max_intentos_heuristica

        start_event = MyStartEvent(
            query=query,
            mapa_original_lista=mapa_lista,
            num_c=N,
            posiciones_validas=posiciones_validas,
            lista_ponderada=lista_ponderada
        )

        try:
            logger.info("Lanzando flow.run(start_event=MyStartEvent(...))")
            t_heur = time.perf_counter()
            result = await flow.run(start_event=start_event)
            metrics["time_heuristic"] += time.perf_counter() - t_heur
            # Extraer resultado de forma robusta
            if isinstance(result, str):
                resultado_texto = result
            else:
                # si es StopEvent o similar con .result
                try:
                    # algunos wrappers retornan el StopEvent directamente
                    if hasattr(result, "result"):
                        resultado_texto = result.result
                    else:
                        resultado_texto = str(result)
                except Exception:
                    resultado_texto = str(result)
            if resultado_texto is None:
                resultado_texto = ""
            logger.info(f"Workflow returned (trunc): {repr(resultado_texto)[:300]}")
        except Exception as e:
            logger.exception(f"Error crítico ejecutando el flujo heurístico (flow.run)."); resultado_texto = ""

        if not resultado_texto:
            logger.warning("El workflow heurístico no devolvió NINGÚN resultado (se aplicará fallback final).")

    # VALIDACIÓN FINAL Y FALLBACK FINAL
    if not resultado_texto or not validar_solucion_python(mapa_lista, resultado_texto, N):
        logger.warning("Solución final inválida o vacía; aplicando fallback heurístico local como última opción.")
        lista_ponderada = ponderar_posiciones(mapa_lista, posiciones_validas)
        top_candidatos = lista_ponderada[: (args.base_candidatos + N * args.candidatos_por_contenedor)]
        fallback_coords = [pos for _, pos in top_candidatos[:N]]
        resultado_texto = format_map_from_coords(mapa_lista, fallback_coords)

    valid_final = validar_solucion_python(mapa_lista, resultado_texto, N)
    if not valid_final:
        logger.error("ERROR FINAL: solución inválida incluso después del fallback.")
    else:
        logger.info("Validación final OK.")

    # Guardado
    os.makedirs(args.output_dir, exist_ok=True)
    ruta_salida = os.path.join(args.output_dir, os.path.basename(args.input))
    with open(ruta_salida, "w") as f:
        f.write(resultado_texto)

    # métricas y resumen
    cols_orig = len(mapa_lista[0])
    mapa_llm_lineas = [line for line in resultado_texto.splitlines() if line.strip()]
    mapa_llm = [list(line.ljust(cols_orig)) for line in mapa_llm_lineas]
    casas = [(i,j) for i in range(len(mapa_llm)) for j in range(len(mapa_llm[0])) if mapa_llm[i][j] == 'O']
    contenedores = [(i,j) for i in range(len(mapa_llm)) for j in range(len(mapa_llm[0])) if mapa_llm[i][j] == 'C']
    total_pasos = calcular_distancia_total(casas, contenedores)

    metrics["total_runtime"] = time.perf_counter() - t_start
    metrics["puntuacion_final"] = total_pasos

    logger.info("="*60)
    logger.info(f"Salida guardada en: {ruta_salida}")
    logger.info(f"Solución óptima calculada: {solucion_es_optima}")
    logger.info(f"Puntuación final (total_pasos): {total_pasos}")
    logger.info("Métricas:")
    logger.info(json.dumps(metrics, indent=2))
    logger.info("="*60)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))