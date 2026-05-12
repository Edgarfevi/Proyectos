# FAEA - Entrega de scripts del análisis de $t\bar{t}$

Este directorio se entrega solo con los tres scripts del análisis y con la estructura mínima necesaria para reproducir los resultados de la sección eficaz.

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