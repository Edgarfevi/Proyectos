from llama_index.llms.ollama import Ollama
from collections import deque
from llama_index.llms.google_genai import GoogleGenAI	
from google.genai import types
import os

# --- 1. VARIABLE GLOBAL CON EL NOMBRE DEL MAPA ---
NOMBRE_MAPA = "1.txt"

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyD9E2JHMibZVU56Hs-rr1M7SmRs7diCGY0",
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)  # deshabilita thinking
    )
)

# --- 2. Leer mapa usando la variable global ---
ruta_entrada = f"entradas/{NOMBRE_MAPA}"
with open(ruta_entrada, "r") as f:
    mapa = [list(line.strip()) for line in f.readlines()]

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

# --- 5. Pedir al LLM la colocación de contenedores ---
mapa_str = '\n'.join([''.join(fila) for fila in mapa])
query = f"""
Tengo un mapa:
{mapa_str}

Este mapa es una representación de una ciudad donde las 'X' son carreteras, 'O' son casas y '-' 
son espacios en blanco. Quiero colocar 2 contenedores 'C' para que las casas puedan tirar la basura, quiero
que permitan que desde las casas haya la minima distancia a los contenedores.

Reglas para colocar 2 contenedores 'C':
1. Contenedor solo en '-'
2. Debe estar al lado de 'X'
3. Minimizar pasos desde todas las casas 'O'

No modifiques el mapa inicial más que para colocar las 'C', que ocuparán el lugar de '-' debe conservar las mismas dimensiones y elementos

Devuélveme SOLO el mapa completo con los 2 contenedores 'C' colocados respetando estas reglas.
"""
resultado = llm.complete(query)

# --- 6. Convertir mapa LLM a lista de listas, normalizando ancho ---
resultado_texto = resultado.text
print("Mapa devuelto por LLM:")
print(resultado_texto)

# ✅ --- Guardar mapa generado usando la variable global ---
ruta_salida = f"salidas/{NOMBRE_MAPA}"
os.makedirs("salidas", exist_ok=True)
with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(resultado_texto)
print(f"\n✅ Mapa guardado en {ruta_salida}")

ancho = max(len(line) for line in resultado_texto.splitlines())
mapa_llm = [list(line.ljust(ancho)) for line in resultado_texto.splitlines() if line.strip()]

# --- 7. Calcular suma de pasos ---
total_pasos = distancia_total(mapa_llm)
print("\nSuma total de pasos de todas las casas al contenedor más cercano:", total_pasos)