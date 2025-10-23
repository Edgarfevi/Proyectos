from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
import asyncio
from collections import deque
from llama_index.llms.google_genai import GoogleGenAI
import os

# --- VARIABLES GLOBALES ---
NOMBRE_MAPA = "1.txt"
CANTIDAD_C = 2

local_llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyCzbWPsAw6sxcnQzIgZgbkF3u2zON2Ffz4"
)

Settings.llm = local_llm

# --- Eventos simplificados ---
class AnalisisEvent(Event):
    texto: str
    analisis: str

class SolucionEvent(Event):
    texto: str
    analisis: str
    salida: str
    intentos: int

# --- Workflow simplificado (3 steps) ---
class ProblemaFlow(Workflow):
    llm = local_llm
    max_intentos = 2

    # Step 1: Analizar problema completo (reglas + restricciones + formato)
    @step
    async def analizar_problema(self, ev: StartEvent) -> AnalisisEvent:
        prompt = f"""
Analiza este problema y extrae:
1. Las reglas que deben cumplirse
2. Las restricciones existentes
3. El formato de salida esperado

Enunciado:
{ev.query}

Responde de forma estructurada y concisa.
"""
        response = await self.llm.acomplete(prompt)
        return AnalisisEvent(texto=ev.query, analisis=response.text)

    # Step 2: Generar y validar solución (con reintentos)
    @step
    async def generar_solucion(self, ev: AnalisisEvent) -> SolucionEvent:
        mejor_solucion = None
        intento = 1
        
        while intento <= self.max_intentos:
            print(f"Intento {intento} de {self.max_intentos}...")
            
            prompt = f"""
Problema original:
{ev.texto}

Análisis del problema:
{ev.analisis}

{"GENERA" if intento == 1 else f"REINTENTO {intento}: Genera una NUEVA"} solución que cumpla TODAS las reglas.

IMPORTANTE:
- Devuelve SOLO el mapa completo con los {CANTIDAD_C} contenedores 'C' colocados
- NO incluyas explicaciones, comentarios ni texto adicional
- Mantén las mismas dimensiones del mapa original
- Cada línea debe tener exactamente el mismo formato que el original
"""
            
            respuesta = await self.llm.acomplete(prompt)
            solucion = respuesta.text.strip()
            
            # Validación simple
            prompt_validacion = f"""
Solución:
{solucion}

Análisis del problema:
{ev.analisis}

¿Cumple TODAS las reglas? Responde SOLO: SI o NO
"""
            validacion = await self.llm.acomplete(prompt_validacion)
            es_valida = "SI" in validacion.text.upper().strip()
            
            if mejor_solucion is None or es_valida:
                mejor_solucion = solucion
            
            print(f"Validación: {validacion.text.strip()}")
            
            if es_valida:
                print(f"✅ Solución válida en intento {intento}")
                return SolucionEvent(
                    texto=ev.texto,
                    analisis=ev.analisis,
                    salida=mejor_solucion,
                    intentos=intento
                )
            
            intento += 1
        
        print(f"⚠️ No se encontró solución válida. Usando mejor intento.")
        return SolucionEvent(
            texto=ev.texto,
            analisis=ev.analisis,
            salida=mejor_solucion,
            intentos=self.max_intentos
        )

    # Step 3: Finalizar
    @step
    async def finalizar(self, ev: SolucionEvent) -> StopEvent:
        return StopEvent(result=ev.salida)

# --- 4. Función para calcular distancia mínima ---
def distancia_minima(mapa, start, targets):
    filas, cols = len(mapa), len(mapa[0])
    visitado = [[False]*cols for _ in range(filas)]
    queue = deque([(start[0], start[1], 0)])
    
    while queue:
        i, j, d = queue.popleft()
        if (i, j) in targets:
            return d
        if visitado[i][j]:
            continue
        visitado[i][j] = True
        
        # Moverse a las 4 direcciones (arriba, abajo, izquierda, derecha)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            # Solo verificar que esté dentro de los límites
            # TODAS las casillas son transitables (-, X, O, C)
            if 0 <= ni < filas and 0 <= nj < cols:
                queue.append((ni, nj, d+1))
    
    return float('inf')

# --- 5. Función distancia_total ---
def distancia_total(mapa):
    filas, cols = len(mapa), len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    contenedores = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'C']
    
    total = 0
    for casa in casas:
        total += distancia_minima(mapa, casa, contenedores)
    return total

# --- Ejecución ---
async def main():
    ruta_entrada = f"entradas/{NOMBRE_MAPA}"
    
    with open(ruta_entrada, "r") as f:
        mapa = [list(line.strip()) for line in f.readlines()]
    
    os.makedirs("salidas", exist_ok=True)
    
    mapa_str = '\n'.join([''.join(fila) for fila in mapa])
    query = f"""
Tengo un mapa:
{mapa_str}

Este mapa es una representación de una ciudad donde:
- 'X' son carreteras
- 'O' son casas
- '-' son espacios en blanco

Objetivo: Colocar {CANTIDAD_C} contenedores 'C' para minimizar la distancia total desde todas las casas.

Reglas OBLIGATORIAS:
1. Contenedor SOLO en posiciones '-'
2. Cada contenedor debe estar ADYACENTE (al lado) de al menos una 'X'
3. Minimizar la suma de pasos desde todas las casas 'O' a su contenedor más cercano
4. NO modificar el mapa excepto para colocar las 'C'
5. Mantener las mismas dimensiones

Devuélveme SOLO el mapa completo con los {CANTIDAD_C} contenedores 'C' colocados.
"""

    flow = ProblemaFlow(timeout=2000, verbose=True)
    result = await flow.run(query=query)

    resultado_texto = result
    print("\n" + "="*50)
    print("MAPA GENERADO POR LLM:")
    print("="*50)
    print(resultado_texto)
    print("="*50 + "\n")

    ruta_salida = f"salidas/{NOMBRE_MAPA}"
    with open(ruta_salida, "w") as f:
        f.write(resultado_texto)
    
    print(f"✅ Mapa guardado en: {ruta_salida}\n")

    ancho = max(len(line) for line in resultado_texto.splitlines())
    mapa_llm = [list(line.ljust(ancho)) for line in resultado_texto.splitlines() if line.strip()]
    print(mapa_llm)

    total_pasos = distancia_total(mapa_llm)
    print(f"📊 Suma total de pasos: {total_pasos}\n")


if __name__ == "__main__":
    asyncio.run(main())