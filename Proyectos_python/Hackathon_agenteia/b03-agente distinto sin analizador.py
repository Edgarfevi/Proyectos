import asyncio
import os
import math
import itertools
import time
import argparse
import logging
from typing import List, Tuple, Iterable, Optional

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step


entrada = "entradas/1c.txt"

def parse_args():
    p = argparse.ArgumentParser(description="Solver Contenedores")
    p.add_argument("--input", "-i", default=entrada, help="Archivo de entrada")
    p.add_argument("--output-dir", "-o", default="salidas", help="Directorio de salida")
    p.add_argument("--N", type=int, default=10, help="Número de contenedores a colocar")
    p.add_argument("--umbral-combinaciones", type=int, default=10_000_000, help="Umbral para decidir fuerza bruta")
    p.add_argument("--base-candidatos", type=int, default=50, help="Base candidatos heurística")
    p.add_argument("--candidatos-por-contenedor", type=int, default=15, help="Candidatos por contenedor (heurística)")
    p.add_argument("--timeout-ms", type=int, default=5000, help="Timeout del flow/LLM en ms")
    p.add_argument("--model", default="gemini-2.5-flash", help="Modelo Gemini a usar")
    p.add_argument("--use-llm-for-formatting", action="store_true", help="Si se quiere usar LLM para formateo de fuerza bruta (opcional)")
    p.add_argument("--formating-attempts", type=int, default=3, help="Reintentos del LLM para formateo")
    p.add_argument("--max-eval", type=int, default=None, help="Límite de evaluaciones en fuerza bruta (safety)")
    p.add_argument("--max-evaluations-safety", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max_eval", type=int, default=None, help="Alias: límite de evaluaciones (safety)")
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
# Funciones utilitarias / reglas
# -----------------------
DELTAS_8_DIR = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),           ( 0, 1),
                ( 1, -1), ( 1, 0),  ( 1, 1)]

def encontrar_posiciones_candidatas(mapa: List[List[str]]) -> List[Tuple[int,int]]:
    if not mapa or not mapa[0]:
        return []
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
    if not contenedores_actuales:
        return float('inf')
    total = 0
    for casa in casas:
        total += min(distancia_manhattan(casa, cont) for cont in contenedores_actuales)
    return total

def ponderar_posiciones(mapa: List[List[str]], posiciones_candidatas: List[Tuple[int,int]]) -> List[Tuple[int, Tuple[int,int]]]:
    filas, cols = len(mapa), len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    if not casas:
        return [(0, pos) for pos in posiciones_candidatas]
    puntuaciones = []
    for pos in posiciones_candidatas:
        puntuaciones.append((sum(distancia_manhattan(pos, casa) for casa in casas), pos))
    puntuaciones.sort(key=lambda item: item[0])
    return puntuaciones

def validar_solucion_python(mapa_original_lista: List[List[str]], mapa_generado_str: str, num_c_esperado: int) -> bool:
    mapa_gen_lineas = [line for line in mapa_generado_str.splitlines() if line.strip()]
    if not mapa_gen_lineas:
        logger.debug("Validación: Mapa vacío.")
        return False
    filas_orig, cols_orig = len(mapa_original_lista), len(mapa_original_lista[0])
    if len(mapa_gen_lineas) != filas_orig:
        logger.debug(f"Validación: Filas {len(mapa_gen_lineas)} != {filas_orig}.")
        return False
    mapa_gen_lista = []
    for r in range(filas_orig):
        linea = list(mapa_gen_lineas[r])
        if len(linea) != cols_orig:
            logger.debug(f"Validación: Cols {len(linea)} != {cols_orig} en fila {r}.")
            return False
        mapa_gen_lista.append(linea)
    count_c = 0
    for r in range(filas_orig):
        for c in range(cols_orig):
            celda_orig, celda_gen = mapa_original_lista[r][c], mapa_gen_lista[r][c]
            if celda_gen == 'C':
                count_c += 1
                if celda_orig != '-':
                    logger.debug(f"Validación: 'C' sobre '{celda_orig}' en ({r},{c}).")
                    return False
                adyacente_a_X = False
                for dr, dc in DELTAS_8_DIR:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < filas_orig and 0 <= nc < cols_orig and mapa_original_lista[nr][nc] == 'X':
                        adyacente_a_X = True; break
                if not adyacente_a_X:
                    logger.debug(f"Validación: 'C' no adyacente a 'X' en ({r},{c}).")
                    return False
            elif celda_gen != celda_orig:
                logger.debug(f"Validación: Celda ({r},{c}) modificada.")
                return False
    if count_c != num_c_esperado:
        logger.debug(f"Validación: {count_c} 'C' encontrados, se esperaban {num_c_esperado}.")
        return False
    return True

def format_map_from_coords(mapa_original: List[List[str]], coords: Iterable[Tuple[int,int]]) -> str:
    """Formatea localmente el mapa colocando 'C' en coords (fallback sin LLM)."""
    mapa = [row[:] for row in mapa_original]
    for r, c in coords:
        mapa[r][c] = 'C'
    return '\n'.join(''.join(row) for row in mapa)

# -----------------------
# Fuerza bruta (generador)
# -----------------------
def encontrar_solucion_optima(mapa_lista: List[List[str]], posiciones_validas: List[Tuple[int,int]], num_c_a_colocar: int, max_evaluaciones: Optional[int]=None) -> Optional[Tuple[Tuple[int,int], ...]]:
    """
    Itera combinaciones sin crear la lista completa. Devuelve la mejor combinación encontrada.
    max_evaluaciones: si se pasa, corta tras evaluar ese número de combinaciones (útil como safety).
    """
    logger.info("Iniciando optimización por fuerza bruta (generador).")
    t0 = time.perf_counter()
    filas, cols = len(mapa_lista), len(mapa_lista[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa_lista[i][j] == 'O']
    if not casas:
        logger.warning("No se encontraron casas. Colocando en las primeras posiciones válidas.")
        return tuple(posiciones_validas[:num_c_a_colocar])

    best_distance = float('inf')
    best_combo = None
    evaluated = 0

    combos_iter = itertools.combinations(posiciones_validas, num_c_a_colocar)
    for combo in combos_iter:
        evaluated += 1
        metrics["combinaciones_evaluadas"] += 1
        dist = calcular_distancia_total(casas, combo)
        if dist < best_distance:
            best_distance = dist
            best_combo = combo
            # Early exit if perfect 0 distance (rare)
            if best_distance == 0:
                break
        if max_evaluaciones is not None and evaluated >= max_evaluaciones:
            logger.info(f"Parando búsqueda tras {evaluated} evaluaciones (límite).")
            break

    metrics["time_bruteforce"] += time.perf_counter() - t0
    logger.info(f"Fuerza bruta finalizada. Evaluadas: {evaluated}. Mejor distancia: {best_distance}")
    return best_combo

# -----------------------
# Workflow LLM (heurístico)
# -----------------------
class SolucionEvent(Event):
    texto: str
    salida: str
    intentos: int
    mapa_original_lista: list

class ProblemaFlow(Workflow):
    llm = None  # se seteará al instanciar
    max_intentos = 2

    @step
    async def generar_solucion(self, ev: StartEvent) -> SolucionEvent:
        mejor_solucion = None
        intento = 1
        prev_attempts = []
        while intento <= self.max_intentos:
            logger.info(f"Intento heurístico {intento}/{self.max_intentos}...")
            prompt = ev.query
            # Añadimos contexto de intentos previos al prompt para mejorar reintentos
            if prev_attempts:
                prompt += "\n\nHISTORIAL_INTENTOS_PREVIOS:\n" + "\n".join(prev_attempts[-3:])
                prompt += "\n\nPor favor corrige los errores detectados en los intentos previos."

            try:
                metrics["llm_calls"] += 1
                respuesta = await self.llm.acomplete(prompt)
                solucion = respuesta.text.strip()
            except Exception as e:
                logger.error(f"Error llamando al LLM: {e}")
                solucion = ""

            es_valida = False
            try:
                es_valida = validar_solucion_python(ev.mapa_original_lista, solucion, ev.num_c)  # type: ignore
            except Exception as e:
                logger.exception("Error en validación de la solución del LLM.")

            prev_attempts.append(solucion if solucion else f"[ERROR: {str(e) if 'e' in locals() else 'LLM_EMPTY'}]")

            if mejor_solucion is None or es_valida:
                mejor_solucion = solucion

            if es_valida:
                return SolucionEvent(texto=ev.query, salida=mejor_solucion, intentos=intento, mapa_original_lista=ev.mapa_original_lista)

            logger.warning("Solución heurística inválida, reintentando...")
            intento += 1

        logger.warning("Agente heurístico: no se encontró solución válida. Devolviendo mejor intento.")
        return SolucionEvent(texto=ev.query, salida=mejor_solucion or "", intentos=self.max_intentos, mapa_original_lista=ev.mapa_original_lista)

    @step
    async def finalizar(self, ev: SolucionEvent) -> StopEvent:
        return StopEvent(result=ev.salida)

# -----------------------
# Main / orquestador
# -----------------------
async def main_async(args):
    t_start = time.perf_counter()

    # LLM setup
    api_key = "AIzaSyDbWiFK8ia8xbhgmfODNtvrCh5LbVSg7RU"
    if api_key is None:
        raise ValueError("❌ Error: La variable de entorno GOOGLE_GENAI_API_KEY no está establecida.")

    local_llm = GoogleGenAI(model=args.model, api_key=api_key)
    Settings.llm = local_llm

    # Lectura
    t0 = time.perf_counter()
    ruta_entrada = args.input
    try:
        with open(ruta_entrada, "r") as f:
            mapa_lista = [list(line.strip()) for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de entrada en {ruta_entrada}")
        return
    metrics["time_read"] += time.perf_counter() - t0

    if not mapa_lista:
        logger.error("El mapa está vacío.")
        return

    mapa_str = '\n'.join(''.join(fila) for fila in mapa_lista)
    t_analysis = time.perf_counter()
    posiciones_validas = encontrar_posiciones_candidatas(mapa_lista)
    metrics["time_analysis"] += time.perf_counter() - t_analysis

    if not posiciones_validas:
        logger.error("No se encontraron posiciones válidas.")
        return

    k = len(posiciones_validas)
    N = args.N
    try:
        num_combos = math.comb(k, N) if k >= N else 0
    except Exception:
        num_combos = float('inf')

    logger.info("--- ANÁLISIS ---")
    logger.info(f"Posiciones candidatas (k): {k}")
    logger.info(f"Contenedores a colocar (N): {N}")
    logger.info(f"Combinaciones totales (k choose N): {num_combos if num_combos != float('inf') else 'inf'}")

    resultado_texto = ""
    solucion_es_optima = False

    # DECISIÓN ESTRATEGIA
    if num_combos > 0 and num_combos <= args.umbral_combinaciones:
        # FUERZA BRUTA
        logger.info("Estrategia: Fuerza Bruta (óptima).")
        t_bf = time.perf_counter()
        # max_evaluaciones opcional: si se desea un safety
        mejor_coords = encontrar_solucion_optima(mapa_lista, posiciones_validas, N, max_evaluaciones=args.max_eval)
        metrics["time_bruteforce"] += time.perf_counter() - t_bf
        solucion_es_optima = True
        if not mejor_coords:
            logger.error("La fuerza bruta no encontró una solución.")
            return

        # Formateo — preferimos hacerlo localmente. LLM solo si se solicita.
        mapa_contingencia = format_map_from_coords(mapa_lista, mejor_coords)
        resultado_texto = mapa_contingencia

        # Si se pide explícitamente usar LLM para formateo (mantengo la opción)
        if args.use_llm_for_formatting:
            logger.info("Usando LLM para formateo (opcional).")
            prompt_formateo = f"""
TAREA: Coloca el carácter 'C' en estas coordenadas del mapa.
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
                    logger.error(f"LLM formatting error: {e}")
                    candidate = ""
                metrics["time_llm_format"] += time.perf_counter() - t_llm

                if candidate and validar_solucion_python(mapa_lista, candidate, N):
                    resultado_texto = candidate
                    logger.info("LLM ha formateado correctamente.")
                    break
                else:
                    logger.warning("LLM formateó mal la salida; reintentando (o fallback local).")
            # si no produjo resultado válido, resultado_texto ya contiene el mapa local.

    else:
        # HEURÍSTICA + AGENTE LLM
        logger.info("Estrategia: Heurística con Agente LLM.")
        # Poda dinámica
        max_candidatos_heuristicos = args.base_candidatos + (N * args.candidatos_por_contenedor)
        lista_ponderada = ponderar_posiciones(mapa_lista, posiciones_validas)
        top_candidatos = lista_ponderada[:max_candidatos_heuristicos]
        texto_ponderado = "\n".join([f"  - Score: {score}, Pos: {pos}" for score, pos in top_candidatos])

        query = f"""
Tengo un mapa:
{mapa_str}

Objetivo: Colocar {N} contenedores 'C' para minimizar la distancia total desde todas las casas.

Reglas OBLIGATORIAS:
1. Contenedor SOLO en posiciones '-'
2. Cada contenedor debe estar ADYACENTE (en las 8) de al menos una 'X'
3. Minimizar la suma de pasos desde todas las casas 'O' a su contenedor más cercano.
4. NO modificar el mapa excepto para colocar las 'C'

ANÁLISIS (Python): Estas son las {len(top_candidatos)} mejores posiciones válidas (ordenadas de mejor a peor):
{texto_ponderado}

IMPORTANTE: Usa la lista como guía pero elige el conjunto de {N} que mejor cobertura conjunta ofrezca.
Devuélveme SOLO el mapa completo con los {N} contenedores 'C' colocados.
"""

        # Ejecutar workflow LLM
        flow = ProblemaFlow(timeout=args.timeout_ms, verbose=True)
        # Seteo del llm y un pequeño hack para pasar num_c al evento (usamos atributos dinámicos)
        flow.llm = local_llm
        # Inyectamos atributos adicionales en el StartEvent: mapa_original_lista y num_c
        # El run() del workflow entregará resultado (string)
        t_heur = time.perf_counter()
        try:
            # el run recibe los keywords que el Workflow espera; dependemos de la implementación del Workflow en llama_index
            resultado_texto = await flow.run(query=query, mapa_original_lista=mapa_lista, num_c=N)
        except Exception as e:
            logger.exception(f"Error ejecutando el flujo heurístico: {e}")
            resultado_texto = ""

        metrics["time_heuristic"] += time.perf_counter() - t_heur

        # Si el LLM no devuelve nada válido, intentamos fallback simple: coger las N mejores posiciones individuales
        if not resultado_texto or not validar_solucion_python(mapa_lista, resultado_texto, N):
            logger.warning("LLM heurístico no produjo solución válida; aplicando fallback heurístico local.")
            # fallback: elegir las N posiciones con mejor puntuación individual y formatear localmente
            fallback_coords = [pos for _, pos in top_candidatos[:N]]
            resultado_texto = format_map_from_coords(mapa_lista, fallback_coords)

    # VALIDACIÓN FINAL
    valid_final = validar_solucion_python(mapa_lista, resultado_texto, N)
    if not valid_final:
        logger.error("ERROR FINAL: Solución producida es inválida. Salvando salida de diagnóstico.")
    else:
        logger.info("Validación final OK.")

    # Guardado
    os.makedirs(args.output_dir, exist_ok=True)
    ruta_salida = os.path.join(args.output_dir, os.path.basename(args.input))
    with open(ruta_salida, "w") as f:
        f.write(resultado_texto)

    # Cálculo métricas finales
    cols_orig = len(mapa_lista[0])
    mapa_llm_lineas = [line for line in resultado_texto.splitlines() if line.strip()]
    mapa_llm = [list(line.ljust(cols_orig)) for line in mapa_llm_lineas]
    casas = [(i,j) for i in range(len(mapa_llm)) for j in range(len(mapa_llm[0])) if mapa_llm[i][j] == 'O']
    contenedores = [(i,j) for i in range(len(mapa_llm)) for j in range(len(mapa_llm[0])) if mapa_llm[i][j] == 'C']
    total_pasos = calcular_distancia_total(casas, contenedores)

    metrics["total_runtime"] = time.perf_counter() - t_start
    # print resumen
    logger.info("="*60)
    logger.info(f"Salida guardada en: {ruta_salida}")
    logger.info(f"Solución óptima calculada: {solucion_es_optima}")
    logger.info(f"Suma total de pasos: {total_pasos}")
    logger.info("Métricas:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("="*60)



if __name__ == "__main__":
    args = parse_args()
    # Propago algunos args al namespace para compatibilidad con variables usadas
    # Normalizamos nombres
    if getattr(args, "base_candidatos", None) is None:
        args.base_candidatos = 50
    if getattr(args, "candidatos_por_contenedor", None) is None:
        args.candidatos_por_contenedor = 15

    # Ejecutamos main asincrónico
    asyncio.run(main_async(args))
