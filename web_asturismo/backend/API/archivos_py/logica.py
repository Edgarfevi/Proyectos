'''
Este modulo contiene la lógica principal de la aplicación.
'''
###
#Imports
###
import numpy as np

def desviacionTideal(T):
    '''
    Calcula y normaliza la desviación térmica de la ideal (22 grados).
    '''
    desviacion = np.abs(T - 22)
    desviacion_normalizada = min(desviacion / 22,1)  # Normaliza entre 0 y  1
    return desviacion_normalizada

def vientonorm(v):
    '''
    Normaliza la velocidad del viento.
    '''
    v_normalizada = min(v / 160, 1)  # Normaliza entre 0 y 1
    return v_normalizada

def cielonorm(c):
    '''
    Normaliza el estado del cielo.
    '''
    c_normalizada = min(c / 4, 1)  # Normaliza entre 0 y 1
    return c_normalizada

def indice_normalizado(valor):
    '''
    Normaliza un valor dado un valor máximo.
    '''
    return min(valor / 100, 1)

def idealidad(T, v, c, prec, masificacion):
    '''
    Calcula la idealidad del clima en función de la temperatura, viento, cielo y precipitaciones.
    0 ideal y 1 horrible
    La masificación tiene un peso dominante (85% del total)
    '''
    masificacion_norm = indice_normalizado(masificacion)
    desviacion_T = desviacionTideal(T)
    v_normalizada = vientonorm(v)
    
    # Pesos: masificación domina con 85%, el resto comparte el 15%
    peso_clima = 0.15  # 15% para todos los factores climáticos combinados
    peso_masificacion = 0.85  # 85% para masificación
    
    # Promedio de factores climáticos
    factor_clima = (desviacion_T + v_normalizada + c + prec) / 4
    
    # Idealidad final
    idealidad = (factor_clima * peso_clima) + (masificacion_norm * peso_masificacion)
    
    return idealidad