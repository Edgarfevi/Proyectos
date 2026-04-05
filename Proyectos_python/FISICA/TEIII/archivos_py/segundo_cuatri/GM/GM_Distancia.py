# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as SO
import matplotlib.pyplot as plt

medidas=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Distancia.tsv",usecols=2,skiprows=11)

particulas=[(medidas[i:i+5]).mean() for i in range(0,len(medidas),5)]
distancia=np.array([0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11]) #m
eparticulas=[(medidas[i:i+5]).std() for i in range(0,len(medidas),5)]
epart=np.sqrt(particulas)

#Tomo como incertidumbre el máximo entre la poissoniana o la desviación estandar
epart=[max(epart[i],eparticulas[i]) for i in range(len(epart))]

#print(particulas)

#Funciones de ajuste
def puntual(d,N):
    return N/(d*d)

def geom(d,N):
    return 0.5*N*(1-d/(0.0175**2+d*d)**0.5)

np.set_printoptions(precision=2)
sigma=epart
sigma[-1]=0.9
parametro1,covarianza1=SO.curve_fit(puntual,distancia,particulas, sigma=sigma)
parametro2,covarianza2=SO.curve_fit(geom,distancia,particulas)


print('Para el ajuste puntual el parámetro es: %5.1f' % parametro1.item(), \
'+- %5.1f' % np.sqrt((np.diag(covarianza1)).item()))
print('Para el ajuste geométrico el parámetro es: %5.1f' % parametro2.item(), \
'+- %5.1f' % np.sqrt((np.diag(covarianza2)).item()))

#Representación gráfica
t=np.linspace(0.02,0.11,101)

plt.errorbar(distancia,particulas,yerr = epart, ls = '', marker = '.', c='darkslateblue',label='datos')
plt.plot(t,puntual(t,parametro1),'g-',label='puntual')
plt.plot(t,geom(t,parametro2[0]),'c-',label='no puntual')
plt.xlabel('Distancia (m)')
plt.ylabel(u'Partículas por intervalo de tiempo')
plt.title(u'Cuentas en función de la distancia')
plt.legend(loc='best')
plt.grid()
plt.show()

