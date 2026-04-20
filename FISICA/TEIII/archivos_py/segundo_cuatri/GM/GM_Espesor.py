# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as SO
import matplotlib.pyplot as plt

medidas=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Espesor.tsv",usecols=2,skiprows=11)

particulas=np.array([(medidas[i:i+5]).mean() for i in range(0,len(medidas),5)])
espesor=np.array([0,4.5, 6.5, 129, 161, 206, 258, 328, 419, 516, 590, 645, 849]) #El cero medido sin placas
epart=np.sqrt(particulas)

print(particulas)

def exponencial(e,N,rho):
    return N*np.exp(-e*rho)

parametro,covarianza=SO.curve_fit(exponencial,espesor[2:],particulas[2:], p0 = [500,5e-3]) 
print(parametro)
print(np.sqrt(np.diag(covarianza))) 

#Representación gráfica
t=np.linspace(0,1000,101)

plt.figure()
plt.errorbar(espesor,particulas,yerr = epart, ls = '', marker = '.', c='darkslateblue',label='datos')
plt.plot(t,exponencial(t,parametro[0],parametro[1]),'g-',label='exponencial') 
plt.xlabel('Densidad superficial $mg/cm^2$')
plt.ylabel(u'Partículas por intervalo de tiempo')
plt.title(u'Cuentas en función de la densidad superficial del Aluminio')
plt.legend(loc='best')
plt.grid()
plt.show()