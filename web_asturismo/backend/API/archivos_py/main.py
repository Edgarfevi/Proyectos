'''
Implementación Api
'''
###
#Imports
###
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from logica import idealidad
import json
from datetime import datetime


###
#Ficheros
###

nombrefichero = '/home/edgar/GitHub/Proyectos/web_asturismo/backend/API/Archivos_info/datos3.xlsx'
fichero2 = '/home/edgar/GitHub/Proyectos/web_asturismo/backend/API/Archivos_info/prediccion_12h_normalizada.csv'
densidad = '/home/edgar/GitHub/Proyectos/web_asturismo/backend/API/Archivos_info/masificacion.csv'
fichero = '/home/edgar/GitHub/Proyectos/web_asturismo/backend/API/Archivos_info/turismo-asturias.json'

###
#Separamos datos
###
datos = pd.read_excel(nombrefichero)
datos = datos.replace({np.nan: None})
apis = pd.read_csv(fichero2)
apis = apis.replace({np.nan: None})
#masificacion = pd.read_csv(densidad)

masificacionl = pd.read_csv('indice_masificacion_asturias.csv')
masificacionl = masificacionl.replace({np.nan: None})
municipios = datos['nombre']
descripcion = datos['descripcion']
monumentos = datos['elemento']

temperatura = apis['temperatura']
viento = apis['viento_velocidad']
cielo = apis['cielo_normalizado']
precipitaciones = apis['prob_precipitacion']


masificacion = masificacionl['indice_masificacion']
with open(fichero, 'r', encoding='utf-8') as archivo:
        # json.load() lee el archivo y lo convierte en un diccionario de Python
        datoscal = json.load(archivo)

#print(masificacion)
#valor = datos['val']
#print(municipios, descripcion)

retorno = []
for i in range(len(municipios)):
    ideal = idealidad(temperatura[i], viento[i], cielo[i], precipitaciones[i], masificacion[i])
    retorno.append({'municipio': municipios[i],'monumento':monumentos[i], 'descripcion': descripcion[i], 'idealidad': float(ideal), 'temperatura': float(temperatura[i]), 'viento': float(viento[i]), 'cielo': float(cielo[i]), 'precipitaciones': float(precipitaciones[i]), 'masificacion': float(masificacion[i])})
    #retorno.append({'municipio': municipios[i], 'descripcion': descripcion[i],'masificacion': masificacion[i]})

datoscal = datoscal['CalendarBookings']['CalendarBooking']
fecha_actual = datetime.now()
fecha_actual = fecha_actual.date()
eventosdehoy = []
for evento in datoscal:
    fechainicio = datetime.strptime(evento['StartDate'], '%Y-%m-%d %H:%M:%S')
    fechafin = datetime.strptime(evento['StartDate'], '%Y-%m-%d %H:%M:%S')
    #print(fechainicio)
    if fechainicio.date()<=fecha_actual and fechafin.date()>=fecha_actual:
        eventosdehoy.append(evento['Title'])
retorno.append({'eventos': eventosdehoy})
print(retorno)


# Crea la aplicación FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las origenes (para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)



# Crea un endpoint en la ruta principal ("/")
# Cuando alguien visite la URL raíz, se ejecutará esta función.
@app.get("/")
def ruta_principal():
    return retorno
