import React, { useState, useEffect } from 'react';
import { MapPin, Users, TrendingUp, X } from 'lucide-react';

const AsturiasTourismMap = () => {
  const [selectedConcejo, setSelectedConcejo] = useState(null);
  const [geojsonData, setGeojsonData] = useState(null);
  const [bounds, setBounds] = useState({ minX: 0, minY: 0, maxX: 0, maxY: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [concejoData, setConcejoData] = useState({});
  const [init, setInit] = useState(false);
  const [events, setEvents] = useState([]);
  const [currentEventIndex, setCurrentEventIndex] = useState(0);

  // Función para normalizar nombres de municipios
  const normalizarNombre = (nombre) => {
    return nombre
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .trim();
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Cargar datos de la API local
        const apiResponse = await fetch('http://localhost:8000');
        if (!apiResponse.ok) {
          throw new Error(`Error API: ${apiResponse.status}`);
        }
        const apiData = await apiResponse.json();
        console.log('Datos de la API recibidos:', apiData);
        
        // Procesar datos de la API
        const processedData = {};
        apiData.forEach(item => {
          if (item.municipio) {
            const nombreNormalizado = normalizarNombre(item.municipio);
            
            processedData[nombreNormalizado] = {
              nombreOriginal: item.municipio,
              masificacion: item.masificacion,
              sitio: [item.monumento,item.descripcion] || 'Sin descripción disponible',
              categoria: item.categoria || 'General',
              idealidad: item.idealidad,
              temperatura: item.temperatura,
              viento: item.viento,
              cielo: item.cielo,
              precipitaciones: item.precipitaciones
            };
          } else {
            setEvents(item.eventos || []);
          }
        });
        
        setConcejoData(processedData);
        console.log('Datos de municipios procesados:', processedData);
        
        // Cargar GeoJSON
        const geoResponse = await fetch('https://visorasturias.es/arcgis/rest/services/Asturias/MunicipiosETRS/MapServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson');
        
        if (!geoResponse.ok) {
          throw new Error(`Error HTTP: ${geoResponse.status}`);
        }
        
        const data = await geoResponse.json();
        console.log('GeoJSON cargado:', data);
        console.log('Nombres de municipios en GeoJSON:', data.features.map(f => f.properties.NAMEUNIT));
        
        // Calcular los límites del mapa
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        data.features.forEach(feature => {
          if (feature.geometry.type === 'Polygon') {
            feature.geometry.coordinates[0].forEach(coord => {
              minX = Math.min(minX, coord[0]);
              maxX = Math.max(maxX, coord[0]);
              minY = Math.min(minY, coord[1]);
              maxY = Math.max(maxY, coord[1]);
            });
          } else if (feature.geometry.type === 'MultiPolygon') {
            feature.geometry.coordinates.forEach(polygon => {
              polygon[0].forEach(coord => {
                minX = Math.min(minX, coord[0]);
                maxX = Math.max(maxX, coord[0]);
                minY = Math.min(minY, coord[1]);
                maxY = Math.max(maxY, coord[1]);
              });
            });
          }
        });
        
        setBounds({ minX, minY, maxX, maxY });
        setGeojsonData(data);
        setLoading(false);
      } catch (err) {
        setError('Error al cargar los datos: ' + err.message);
        setLoading(false);
        console.error(err);
      }
    };

    fetchData();
  }, []);

  // Efecto para rotar los eventos cada 5 segundos
  useEffect(() => {
    if (events.length > 0) {
      const interval = setInterval(() => {
        setCurrentEventIndex((prevIndex) => (prevIndex + 1) % events.length);
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [events.length]);

  const getColor = (masificacion) => {
    // Interpolación de verde a amarillo a rojo
    if (masificacion < 50) {
      const ratio = masificacion / 50;
      return `rgb(${Math.round(34 + ratio * 221)}, ${Math.round(197 + ratio * 58)}, 34)`;
    } else {
      const ratio = (masificacion - 50) / 50;
      return `rgb(255, ${Math.round(255 - ratio * 102)}, ${Math.round(34 - ratio * 34)})`;
    }
  };

  const getMasificacionLabel = (masificacion) => {
    if (masificacion < 40) return 'Baja';
    if (masificacion < 60) return 'Media';
    if (masificacion < 75) return 'Alta';
    return 'Muy Alta';
  };

  const projectToSVG = (lon, lat) => {
    if (!bounds.minX || !bounds.maxX) return { x: 0, y: 0 };
    
    const x = ((lon - bounds.minX) / (bounds.maxX - bounds.minX)) * 100;
    const y = ((bounds.maxY - lat) / (bounds.maxY - bounds.minY)) * 100;
    
    return { x, y };
  };

  if (!init) {
    return (
      <div className="relative w-full h-screen overflow-hidden">
        {/* Logo/Empresa */}
        <div className="absolute top-8 left-8 bg-white/90 backdrop-blur rounded-lg shadow-lg p-4">
          <div className="w-32 h-32 rounded-lg flex items-center justify-center">
            <img 
              src="../public/empresa.jpg"
              alt="Logo Empresa"
              className="w-full h-full object-contain"
            /> 
          </div>
          <p className="text-center mt-2 text-sm font-semibold text-gray-700">Empresa Organizadora</p>
        </div>

        {/* Contenido principal */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center px-8 max-w-4xl animate-fade-in bg-white/90 p-40 rounded-lg">
            <h1 className="text-6xl md:text-8xl font-bold text-black mb-4 drop-shadow-lg">
              Bienvenidos
            </h1>
            <h2 className="text-3xl md:text-5xl font-semibold text-black/90 mb-2">
              al corazón de
            </h2>
            <h3 className="text-5xl md:text-7xl font-extrabold text-yellow-300 mb-8 drop-shadow-lg">
              ASTURISMO
            </h3>
            <p className="text-xl md:text-2xl text-black/95 mb-4 font-medium">
              Tu brújula para explorar la esencia geográfica del Paraíso Natural.
            </p>
            <p className="text-lg md:text-xl text-black/90 mb-8">
              ¿Listo para descubrir los secretos de Asturias en un mapa interactivo?
            </p>
            <button
              onClick={() => setInit(true)}
              className="px-8 py-4 bg-yellow-400 hover:bg-yellow-500 text-gray-900 font-bold text-lg rounded-full shadow-2xl transform hover:scale-105 transition-all duration-300 hover:shadow-yellow-400/50 cursor-pointer"
            >
              ¡EXPLORA EL MAPA AHORA!
            </button>
          </div>
        </div>

        {/* Footer */}
        <footer className="absolute bottom-4 left-0 right-0 text-center text-white/70 text-sm">
          &copy; 2025 HACKATHON - Autores: Pablo García, Edgar Fernández, Sergio Fernández, Pablo Mojardín, Juan Rodríguez
        </footer>
      </div>
    );
  }

  return (
    <div className="w-full h-screen p-6">
      <div className="h-full flex flex-col">
        <div className="text-center bg-white/80">
          <h1 className="text-4xl font-bold text-black mb-2">
            Mapa de turismo verde de Asturias
          </h1>
          <p className="text-gray-700">
            Haz clic en un concejo para ver más detalles del mismo.
          </p>
        </div>

        <div style={{height: 'calc(100vh - 150px)'}} className="flex-1">
          
          {/* Mapa */}
          <div className="w-full h-full bg-white/50 p-6 relative overflow-hidden">
            {/* Banner de eventos rotativo */}
            {events.length > 0 && (
              <div className="absolute top-6 right-6 z-10 w-96 max-w-md">
                <div className="bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-400 rounded-lg shadow-2xl p-4 border-2 border-yellow-500">
                  <div className="flex flex-col gap-3">
                    <div className="text-center">
                      <p className="text-gray-900 font-bold text-base animate-pulse">
                        🎉 {events[currentEventIndex]}
                      </p>
                    </div>
                    <div className="flex gap-1 justify-center">
                      {events.map((_, index) => (
                        <button
                          key={index}
                          onClick={() => setCurrentEventIndex(index)}
                          className={`w-2 h-2 rounded-full transition-all ${
                            index === currentEventIndex 
                              ? 'bg-gray-900 w-4' 
                              : 'bg-gray-600 hover:bg-gray-700'
                          }`}
                          aria-label={`Ver evento ${index + 1}`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-white">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Cargando mapa de Asturias...</p>
                </div>
              </div>
            )}
            
            {error && (
              <div className="absolute inset-0 flex items-center justify-center bg-white">
                <div className="text-center text-red-600">
                  <p className="mb-4">{error}</p>
                  <button 
                    onClick={() => window.location.reload()} 
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  >
                    Reintentar
                  </button>
                </div>
              </div>
            )}
            
            {!loading && !error && geojsonData && (
              <div className="relative w-full h-full m-4">
                <svg 
                  viewBox="0 0 100 100" 
                  className="w-full h-full"
                  preserveAspectRatio="none"
                  style={{ display: 'block' }}
                >
                  {geojsonData.features.map((feature, idx) => {
                    const pathData = [];
                    
                    const processPolygon = (coordinates) => {
                      const points = coordinates.map(coord => {
                        const { x, y } = projectToSVG(coord[0], coord[1]);
                        return `${x},${y}`;
                      }).join(' ');
                      return points;
                    };
                    
                    if (feature.geometry.type === 'Polygon') {
                      feature.geometry.coordinates.forEach(ring => {
                        pathData.push(processPolygon(ring));
                      });
                    } else if (feature.geometry.type === 'MultiPolygon') {
                      feature.geometry.coordinates.forEach(polygon => {
                        polygon.forEach(ring => {
                          pathData.push(processPolygon(ring));
                        });
                      });
                    }
                    
                    const nombreGeoJSON = feature.properties.NAMEUNIT || feature.properties.NOMBRE;
                    const nombreNormalizado = normalizarNombre(nombreGeoJSON);
                    const datosDelConcejo = concejoData[nombreNormalizado];
                    
                    return (
                      <g key={idx}>
                        {pathData.map((points, i) => (
                          <polygon
                            key={i}
                            points={points}
                            fill={datosDelConcejo 
                              ? getColor(datosDelConcejo.masificacion)
                              : "#60a5fa"}
                            stroke="#1e40af"
                            strokeWidth="0.2"
                            className="cursor-pointer hover:opacity-80 transition-opacity"
                            onClick={() => {
                              console.log('Clicked:', nombreGeoJSON, '-> normalizado:', nombreNormalizado);
                              if (datosDelConcejo) {
                                setSelectedConcejo({
                                  nombre: datosDelConcejo.nombreOriginal || nombreGeoJSON,
                                  ...datosDelConcejo
                                });
                              } else {
                                console.log('No hay datos para:', nombreNormalizado);
                              }
                            }}
                          />
                        ))}
                      </g>
                    );
                  })}
                </svg>
              </div>
            )}

            {/* Leyenda */}
            <div className="absolute bottom-6 right-6 bg-white/95 backdrop-blur rounded-lg shadow-lg p-4">
              <h3 className="text-sm font-bold text-gray-700 mb-2">Calificación general</h3>
              <div className="space-y-1">
                {[
                  { color: 'rgb(34, 197, 34)', label: 'Muy Alta (75%+)' },
                  { color: 'rgb(255, 255, 34)', label: 'Alta (60-75%)' },
                  { color: 'rgb(255, 200, 17)', label: 'Media (40-60%)' },
                  { color: 'rgb(255, 153, 0)', label: 'Baja (0-40%)' },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-xs text-gray-600">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Modal de información */}
          {selectedConcejo && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl shadow-2xl w-full max-w-md max-h-[90vh] overflow-y-auto">
                <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex justify-between items-start">
                  <div className="flex-1">
                    <h2 className="text-3xl font-bold text-gray-800 mb-2">
                      {selectedConcejo.nombre}
                    </h2>
                    <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                      {selectedConcejo.categoria}
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedConcejo(null)}
                    className="cursor-pointer ml-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <X size={24} className="text-gray-600" />
                  </button>
                </div>

                <div className="p-6 space-y-6">
                  {/* Sitio de interés */}
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <MapPin className="text-indigo-600" size={20} />
                      <h3 className="font-bold text-gray-800">{selectedConcejo.sitio[0]}</h3>
                    </div>
                    <p className="text-gray-700">
                      {selectedConcejo.sitio[1]}
                    </p>
                  </div>

                  {/* Nivel de masificación */}
                  <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Users className="text-orange-600" size={20} />
                      <h3 className="font-bold text-gray-800">Nivel de Masificación</h3>
                    </div>
                    
                    <div className="mb-3">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-2xl font-bold text-gray-800">
                          {selectedConcejo.masificacion.toFixed(1)}%
                        </span>
                        <span className="text-sm font-semibold text-gray-600 px-3 py-1 bg-white rounded-full">
                          {getMasificacionLabel(selectedConcejo.masificacion)}
                        </span>
                      </div>
                      
                      <div className="w-full bg-gray-200 rounded-full h-6 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                          style={{
                            width: `${Math.min(selectedConcejo.masificacion, 100)}%`,
                            backgroundColor: getColor(selectedConcejo.masificacion),
                          }}
                        >
                          <span className="text-xs font-bold text-white drop-shadow">
                            {selectedConcejo.masificacion.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2 text-sm text-gray-600">
                      <div className="flex items-center gap-2">
                        <TrendingUp size={16} className="text-gray-500" />
                        <span>
                          {selectedConcejo.masificacion > 75 
                            ? 'Zona de alta afluencia turística'
                            : selectedConcejo.masificacion > 60
                            ? 'Zona con afluencia turística considerable'
                            : selectedConcejo.masificacion > 40
                            ? 'Zona con afluencia turística moderada'
                            : 'Zona poco masificada, ideal para turismo tranquilo'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Estadísticas adicionales */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-green-50 rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-green-700">
                        {selectedConcejo.temperatura}°C
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Temperatura
                      </div>
                    </div>
                    <div className="bg-purple-50 rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-purple-700">
                        {selectedConcejo.viento} km/h
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Viento
                      </div>
                    </div>
                    <div className="bg-blue-50 rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-blue-700">
                        {((1-selectedConcejo.idealidad) * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Idealidad
                      </div>
                    </div>
                    <div className="bg-yellow-50 rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-yellow-700">
                        {(selectedConcejo.cielo * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Nubosidad
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AsturiasTourismMap;