# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit

data=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Fondo.tsv",usecols=2,skiprows=11)

nbins=np.array(range(0,int(data.max()+3)))
n,bins,patches=plt.hist(data,bins=nbins,rwidth=0.5,align='left',density=True,label='Datos')

errorn=np.sqrt(n*len(data))/len(data)


plt.errorbar(bins[:-1],n,yerr = errorn, ls = '', marker = '.', c='darkslateblue')
plt.xlabel('Cuentas / 2 segundos')
plt.ylabel('Probabilidad')
plt.title('Ajuste')

def mipoisson(x,mu):
    return poisson.pmf(x,mu)

###estimacion de parametros en base a la muestra
mu = np.mean(data)
print("Media es: %.2f cuentas/2s" % mu)
std = np.std(data)
print("Std es: %.2f cuentas/2s" % std)

popt_p, pcov_p = curve_fit(mipoisson, bins[:-1], n, p0=[mu])

# Dibuja Poisson 
x = range(0,int(data.max()+3))
x = np.linspace(bins[0], bins[-1], len(nbins))
plt.plot(x, mipoisson(x, *popt_p), label='Poisson fit', color='blue')

print("Ajuste da un parametro mu de %.2f +- %.2f cuentas/2s" %(popt_p,pcov_p))
print("Desviation for Poisson is %.2f +- %.2f cuentas/2s" %(np.sqrt(popt_p),np.sqrt(pcov_p)))

plt.legend(loc='best')
plt.show()
