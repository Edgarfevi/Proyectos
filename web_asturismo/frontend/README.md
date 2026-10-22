# Frontend de Web Asturismo

Aplicación React + Vite para visualizar un mapa interactivo de Asturias con datos de masificación turística, condiciones meteorológicas y eventos.

## Requisitos

- Node.js (versión LTS recomendada)
- npm
- Backend FastAPI de este repositorio ejecutándose en `http://localhost:8000`

## Instalación

```bash
npm install
```

## Ejecución en desarrollo

```bash
npm run dev
```

## Scripts disponibles

- `npm run dev`: inicia el servidor de desarrollo de Vite.
- `npm run build`: genera la build de producción.
- `npm run lint`: ejecuta ESLint sobre el proyecto.
- `npm run preview`: sirve localmente la build generada.

## Notas de integración

- El frontend consume datos del backend en `http://localhost:8000`.
- También obtiene geometrías GeoJSON de municipios de Asturias desde un servicio externo de `visorasturias.es`.
