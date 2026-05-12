
# FAEA — Física de Altas Energías y Aplicaciones

Breve repositorio de análisis y ejercicios relacionados con Física de Altas Energías (FAEA). Contiene notebooks, scripts, datos en formato ROOT y resultados reproducibles usados como apoyo para ejercicios, prácticas y pequeños estudios.

---

**Contenido rápido**

- **Resumen**: este repositorio agrupa código, datos y resultados para análisis sencillos con archivos ROOT y visualización en Python.
- **Dónde mirar primero**: [Archivos/](Archivos/) (scripts y notebooks), [Datos/](Datos/) (archivos .root), [Resultados/](Resultados/) (yields, resultados finales), [Informe.tex](Informe.tex) (documento principal).

---

**Estructura del proyecto**

- [Informe.tex](Informe.tex) — Documento LaTeX principal.
- [Archivos/](Archivos/) — Scripts y notebooks de trabajo
	- [Archivos/Introduccion.ipynb](Archivos/Introduccion.ipynb)
	- [Archivos/Plotter.py](Archivos/Plotter.py)
	- [Archivos/primer_ejercicio.ipynb](Archivos/primer_ejercicio.ipynb)
	- [Archivos/segundo_ejercicio.ipynb](Archivos/segundo_ejercicio.ipynb)
	- [Archivos/Selector.py](Archivos/Selector.py)
	- [Archivos/test.py](Archivos/test.py)
- [Datos/](Datos/) — Archivos de entrada en formato ROOT (.root)
	- `data.root`, `dy.root`, `qcd.root`, `single_top.root`, `ttbar.root`, `wjets.root`, `ww.root`, `wz.root`, `zz.root`
- [Imagenes/](Imagenes/) y [img/](img/) — Imágenes y recursos gráficos usados en el informe
- [Programacion_objetos/](Programacion_objetos/) — Ejemplos sobre POO y herencia en Python
	- [Programacion_objetos/Aeroplane.py](Programacion_objetos/Aeroplane.py)
	- [Programacion_objetos/Ejemplo_herencia.py](Programacion_objetos/Ejemplo_herencia.py)
- [Resultados/](Resultados/) — Resultados procesados y tablas
	- [Resultados/yields_MuonPt.txt](Resultados/yields_MuonPt.txt)
	- [Resultados/resultados_finales.txt](Resultados/resultados_finales.txt)
	- [Resultados/seccion_eficaz_final.txt](Resultados/seccion_eficaz_final.txt)

---

Requisitos sugeridos

- Python 3.8+ (recomendado)
- Paquetes habituales: `numpy`, `scipy`, `matplotlib`, `pandas`, `uproot` (o `ROOT` si se prefiere), `jupyter`.

Ejemplo de instalación rápida (entorno virtual recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pandas uproot jupyter
```

Si usas ROOT (PyROOT) en lugar de `uproot`, instala ROOT siguiendo su guía oficial.

---

Uso y reproducción de los análisis

1. Abrir los notebooks:
	 - Ejecuta `jupyter lab` o `jupyter notebook` y abre los notebooks en [Archivos/](Archivos/).
	 - Los notebooks `primer_ejercicio.ipynb` y `segundo_ejercicio.ipynb` contienen pasos reproducibles del análisis.

2. Scripts de utilidades:
	 - [Archivos/Plotter.py](Archivos/Plotter.py): script para generar las figuras principales a partir de archivos ROOT o ficheros intermedios.
		 - Ejemplo: `python3 Archivos/Plotter.py --input Datos/data.root --output Imagenes/fig1.png`
	 - [Archivos/Selector.py](Archivos/Selector.py): funciones de selección y filtrado de eventos que se reutilizan en los notebooks.
	 - [Archivos/test.py](Archivos/test.py): pruebas básicas y ejemplos de ejecución.

3. Datos:
	 - Los archivos RAW están en [Datos/](Datos/). Si trabajan con archivos muy grandes, considera usar una copia reducida para pruebas rápidas.

4. Resultados y tablas:
	 - Salidas numéricas y tablas se guardan en [Resultados/](Resultados/). Revisa `yields_MuonPt.txt` para los yields por bin.

---

Buenas prácticas y consejos

- Versiona solo los scripts y resultados resumidos; los archivos ROOT grandes pueden mantenerse fuera del repo si son muy pesados.
- Usa entornos virtuales para asegurar reproducibilidad.
- Documenta en los notebooks los pasos de preprocesado y calibración (si corresponde).

---

Contacto y autoría

Si quieres colaborar, mejorar un script o preguntar algo, abre una issue o contacta al autor del repositorio.

---

Licencia

Por defecto: añade aquí la licencia que prefieras (MIT, GPLv3, etc.). Si no quieres publicar una licencia, indica "Todos los derechos reservados".

---

¿Qué hago ahora?

- Puedo añadir un `requirements.txt` automático con las dependencias detectadas.
- Puedo limpiar y anotar los notebooks para facilitar su lectura.

Si quieres, procedo con cualquiera de estas tareas.

## Archivos que se entregan

- `Archivos/Selector.py`: lee los ficheros ROOT, aplica las selecciones y rellena histogramas.
- `Archivos/Plotter.py`: organiza los histogramas, calcula conteos y genera salidas.
- `Archivos/test.py`: ejecuta el análisis completo y calcula la sección eficaz final.

## Estructura mínima reproducible

Para reproducir los resultados, el proyecto debe mantenerse con esta organización:

```text
FAEA/
├── Archivos/
│   ├── Selector.py
│   ├── Plotter.py
│   └── test.py
├── Datos/
│   ├── data.root
│   ├── qcd.root
│   ├── wjets.root
│   ├── ww.root
│   ├── wz.root
│   ├── zz.root
│   ├── dy.root
│   ├── single_top.root
│   └── ttbar.root
├── results/
└── Resultados/
```

## Condiciones para reproducirlo

- Los nombres de los ficheros ROOT deben conservarse.
- La carpeta `Datos/` debe estar en la raíz del proyecto.
- Los tres scripts de `Archivos/` deben permanecer juntos.
- El orden de las muestras en el script principal debe respetarse, porque el cálculo final depende de él.

## Resultado esperado

Al ejecutar el análisis se generan las figuras, los conteos y el resumen final de la sección eficaz en `results/`.