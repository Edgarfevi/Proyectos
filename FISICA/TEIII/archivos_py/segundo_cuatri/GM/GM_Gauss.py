import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
from scipy.optimize import curve_fit

#Lectura de datos
datosSr=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Cs90gauss.tsv",usecols=2,skiprows=11)
n_intervalos=7
nbins=np.array(range(int(datosSr.min()-5),int(datosSr.max()+5),n_intervalos)) #Intervalos # alterar n_intervalos
#Disminuir el número de intervalos cambiando el 1 final por 2,3,4,5,...

#Construcción del histograma
n,bins,patches=plt.hist(datosSr,bins=nbins,rwidth=0.5,align='left',density=True,label='Datos')
#print(bins)

#Incertidumbres poissonianas
errorn=np.sqrt(n*len(datosSr))/len(datosSr)

#Gráfica
plt.errorbar(bins[:-1],n,yerr = errorn, ls = '', marker = '.', c='darkslateblue')
plt.xlabel('Cuentas por segundo')
plt.ylabel('Probabilidad')
plt.title('Ajuste')

#Funciones de ajuste
def gaussian(x,mu,sigma,A):
    return A*norm.pdf(x,mu,sigma)


def mipoisson(x,mu):
    return poisson.pmf(x,mu)


###estimacion de parametros en base a la muestra
mu = np.mean(datosSr)
sigma2 = np.var(datosSr)
print("Media: %.2f cuentas/s" % mu, '+- %.2f' % np.sqrt(sigma2))
print("Sigma2: %.2f cuentas/s\n" % sigma2)

# Ajuste a gaussiana y poissoniana de los datos del histograma
popt_g, pcov_g = curve_fit(gaussian, bins[:-1], n, p0=[mu, sigma2,1.])
popt_p, pcov_p = curve_fit(mipoisson, bins[:-1], n, p0=[mu])

# Grafica de las dos fucniones en el intervalo de los datos
x = range(int(datosSr.min()-5),int(datosSr.max()+5))
plt.plot(x, gaussian(x, *popt_g), label='Gauss', color='red')
plt.plot(x, mipoisson(x, *popt_p), label='Poisson', color='blue')

print("Ajuste Gaussiano")
print("Mu= is %.1f +- %.1f cuentas/s" %(popt_g[0],np.sqrt(np.diag(pcov_g)[0])))
print("Sigma= %.1f +- %.1f cuentas/s\n" %(popt_g[1],np.sqrt((np.diag(pcov_g))[1])))

print("Ajuste Poissoniano")
print("Mu is %.1f +- %.1f cuentas/s" %(popt_p[0],np.sqrt(np.diag(pcov_p)[0])))

plt.legend(loc='best')
plt.show()

