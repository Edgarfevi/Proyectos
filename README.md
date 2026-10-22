# 🚀 Proyectos

> Repositorio personal de proyectos académicos, experimentos y material de aprendizaje continuo.

[![GitHub](https://img.shields.io/badge/GitHub-Edgarfevi-181717?style=flat&logo=github)](https://github.com/Edgarfevi)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

Este repositorio reúne trabajos de universidad, hackathons, simulaciones,
análisis de datos, inteligencia artificial y desarrollo web. Cada carpeta puede
tener sus propias dependencias y documentación.

## 📋 Tabla de contenidos

- [📁 Estructura del repositorio](#-estructura-del-repositorio)
- [⭐ Proyectos destacados](#-proyectos-destacados)
- [🛠️ Tecnologías](#️-tecnologías)
- [🚀 Instalación y uso](#-instalación-y-uso)
- [🤝 Contribuciones](#-contribuciones)
- [🔐 Seguridad](#-seguridad)
- [📚 Citación](#-citación)
- [📄 Licencia](#-licencia)
- [📧 Contacto](#-contacto)

## 📁 Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| [`Apuntes/`](Apuntes/) | Apuntes, ejercicios y notebooks de clase. |
| [`Cursos/`](Cursos/) | Material práctico de cursos y asignaturas. |
| [`Datathon/`](Datathon/) | Análisis exploratorio, limpieza de datos y modelos predictivos. |
| [`Datathon_2026/`](Datathon_2026/) | Material, datos y código del Datathon 2026. |
| [`Fisica/`](Fisica/) | Prácticas y proyectos de física aplicada, métodos numéricos, electrónica y óptica. |
| [`Hack_arcelor/`](Hack_arcelor/) | Soluciones, datos, informes y benchmarks del hackathon de Arcelor. |
| [`Hackathon_agenteia/`](Hackathon_agenteia/) | Formación y experimentos sobre agentes, LLM y pipelines RAG. |
| [`Hackathon_salud/`](Hackathon_salud/) | Análisis y modelado de datos relacionados con salud. |
| [`IA/`](IA/) | Laboratorios de inteligencia artificial, redes neuronales y utilidades. |
| [`Latex/`](Latex/) | Plantillas, portadas y recursos para documentos académicos en LaTeX. |
| [`Mates/`](Mates/) | Material y ejercicios de matemáticas, incluyendo prácticas en R. |
| [`Proyectos_python/`](Proyectos_python/) | Proyectos personales y experimentos desarrollados en Python. |
| [`spaceapp/`](spaceapp/) | Código y recursos del proyecto de NASA Space Apps. |
| [`web_asturismo/`](web_asturismo/) | Aplicación web de turismo en Asturias, con frontend React/Vite y backend FastAPI. |

## ⭐ Proyectos destacados

### 🎢 Simulador de montaña rusa

Simulación física con cálculos y visualizaciones en 2D y 3D.

- [`Fisica/MNAF2025/Matemática_aplicada/montaña_rusa/`](Fisica/MNAF2025/Matemática_aplicada/montaña_rusa/)

### 🤖 Agentes de inteligencia artificial

Experimentos con agentes inteligentes, modelos de lenguaje y recuperación
aumentada mediante generación (RAG).

- [`Hackathon_agenteia/`](Hackathon_agenteia/)

### 🏥 Análisis de salud

Exploración de datos y modelos relacionados con la enfermedad renal crónica.

- [`Hackathon_salud/`](Hackathon_salud/)

### 🌐 Web Asturismo

Aplicación full-stack orientada a la planificación turística en Asturias.

- [`web_asturismo/frontend/`](web_asturismo/frontend/)
- [`web_asturismo/backend/`](web_asturismo/backend/)

## 🛠️ Tecnologías

### Lenguajes y entornos

- Python
- JavaScript
- R
- LaTeX
- Jupyter Notebook/JupyterLab

### Librerías y frameworks

- NumPy, Pandas y Matplotlib para análisis y visualización.
- SciPy para cálculo científico.
- Manim para animaciones matemáticas.
- React y Vite para el frontend web.
- FastAPI y Uvicorn para la API de Asturismo.

### Herramientas

- Git y Git LFS para el control de versiones y archivos grandes.
- Visual Studio Code.
- Entornos virtuales de Python (`venv`).

## 🚀 Instalación y uso

### Requisitos

- Git.
- Python 3.12 o una versión compatible con el proyecto que se vaya a ejecutar.
- Node.js y npm para [`web_asturismo/frontend/`](web_asturismo/frontend/).
- Jupyter Notebook o JupyterLab para los notebooks.
- Git LFS si se necesitan archivos gestionados con LFS.

### Clonar el repositorio

```bash
git clone https://github.com/Edgarfevi/Proyectos.git
cd Proyectos
```

### Ejecutar un proyecto Python

No existe un único archivo de dependencias para todo el repositorio. Cada
proyecto Python puede requerir paquetes distintos. Se recomienda crear un
entorno virtual dentro del proyecto correspondiente:

```bash
cd ruta/al/proyecto
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
python -m pip install --upgrade pip
```

Si el proyecto incluye un archivo de dependencias, instálalo desde esa carpeta:

```bash
python -m pip install -r requirements.txt
```

### Abrir notebooks

Desde la raíz del repositorio o desde el proyecto que se quiera explorar:

```bash
jupyter lab
```

### Ejecutar el frontend de Web Asturismo

```bash
cd web_asturismo/frontend
npm install
npm run dev
```

Otros comandos disponibles en el frontend:

```bash
npm run build    # Genera la compilación de producción
npm run lint     # Ejecuta ESLint
npm run preview  # Sirve la compilación localmente
```

El backend de [`web_asturismo/backend/`](web_asturismo/backend/) usa FastAPI y
Uvicorn. Antes de ejecutarlo hay que instalar sus dependencias y revisar la
configuración de datos de [`main.py`](web_asturismo/backend/API/archivos_py/main.py),
ya que actualmente contiene rutas locales absolutas a los archivos de datos.

## 🤝 Contribuciones

Este es principalmente un repositorio personal de aprendizaje, pero las
sugerencias y mejoras son bienvenidas:

1. Abre un [issue](https://github.com/Edgarfevi/Proyectos/issues) describiendo
   el cambio o problema.
2. Haz un fork y crea una rama descriptiva.
3. Realiza cambios pequeños y documenta cualquier comportamiento nuevo.
4. Abre una pull request con el contexto y las comprobaciones realizadas.

Consulta [`CONTRIBUTING.md`](CONTRIBUTING.md) para conocer las buenas prácticas
del repositorio.

## 🔐 Seguridad

No publiques credenciales, tokens, datos privados ni información sensible.
Para informar de una vulnerabilidad, sigue el proceso descrito en
[`SECURITY.md`](SECURITY.md) y evita abrir un issue público con los detalles.

## 📚 Citación

Si utilizas este repositorio en un trabajo académico, consulta los metadatos de
[`CITATION.cff`](CITATION.cff).

## 📄 Licencia

El contenido se distribuye bajo la licencia **MIT**. Consulta
[`LICENSE`](LICENSE) para conocer los términos completos.

## 📧 Contacto

**Edgar Fernández Vigil**

- GitHub: [Edgarfevi](https://github.com/Edgarfevi)
- Email: [Edgarfevi@outlook.es](mailto:Edgarfevi@outlook.es)

<div align="center">

⭐ Si encuentras útil algún proyecto, considera darle una estrella ⭐

**Hecho con ❤️ y mucho esfuerzo**

</div>
