"""# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#%%

'''
ESTRONCIO 90
'''

plt.figure()

data=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_sr.tsv",usecols=(1,2),skiprows=11)
V=data[:,0]
Counts=data[:,1]
eCounts=np.sqrt(Counts)


R2 = Counts[-1]
V2 = V[-1]

m = [2*(R2-Counts[i])*1E4/(R2+Counts[i])/(V2-V[i]) for i in range(len(V)-1)]

print(f"analisis de Sr 90 m = {m}")

plt.errorbar(V,Counts,yerr = eCounts, ls = '', marker = '.', c='darkslateblue')
plt.xlabel("Voltaje(V)")
plt.ylabel("Counts/30s")
plt.title("Sr 90")
plt.show()

plt.figure()

data=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Co.tsv",usecols=(1,2),skiprows=11)

V=data[:,0]
Counts=data[:,1]
eCounts=np.sqrt(Counts)
print(V)

R2 = Counts[-1]
V2 = V[-1]

m = [2*(R2-Counts[i])*1E4/(R2+Counts[i])/(V2-V[i]) for i in range(len(V)-1)]

print(f"analisis de Co 60 m = {m}")

plt.errorbar(V,Counts,yerr = eCounts, ls = '', marker = '.', c='darkslateblue')
plt.xlabel("Voltaje(V)")
plt.ylabel("Counts/30s")
plt.title("Co 60")
plt.show()

# %%

'''
TALIO 204
'''

#Cargo datos
data=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Tl.tsv",usecols=(1,2),skiprows=11)
#print(data)
V=data[:,0]
Counts=data[:,1]

#Incertidumbre poissoniana
eCounts=np.sqrt(Counts)

#Tomo como referencia el último punto
R2 = Counts[-1]
V2 = V[-1]

#Cálculo de pendientes m
m = [2*(R2-Counts[i])*1E4/(R2+Counts[i])/(V2-V[i]) for i in range(len(V)-1)]

np.set_printoptions(precision=1)
print('Voltajes (V):', V)
print('Pendientes (Cuentas/V)', m)

#Gráfica
plt.errorbar(V,Counts,yerr = eCounts, ls = '', marker = '.', c='darkslateblue')
plt.xlabel("Voltaje(V)")
plt.ylabel("Counts/30s")
plt.title("Tl 204")
plt.show()


# %%

'''
COBALTO 60
'''

plt.figure()

data=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Co.tsv",usecols=(1,2),skiprows=11)
print(data)
V=data[:,0]
Counts=data[:,1]
eCounts=np.sqrt(Counts)
print(V)

R2 = Counts[-1]
V2 = V[-1]

m = [2*(R2-Counts[i])*1E4/(R2+Counts[i])/(V2-V[i]) for i in range(len(V)-1)]

print(f"analisis de Co 60 m = {m}")

plt.errorbar(V,Counts,yerr = eCounts, ls = '', marker = '.', c='darkslateblue')
plt.xlabel("Voltaje(V)")
plt.ylabel("Counts/30s")
plt.title("Co 60")
plt.show()

#%%

'''
PLUTONIO 238
'''


plt.figure()

data=np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Pu.tsv",usecols=(1,2),skiprows=11)
print(data)
V=data[:,0]
Counts=data[:,1]
eCounts=np.sqrt(Counts)
print(V)

R2 = Counts[-1]
V2 = V[-1]

m = [2*(R2-Counts[i])*1E4/(R2+Counts[i])/(V2-V[i]) for i in range(len(V)-1)]

print(f"analisis de Pu 238 m = {m}")

plt.errorbar(V,Counts,yerr = eCounts, ls = '', marker = '.', c='darkslateblue')
plt.xlabel("Voltaje(V)")
plt.ylabel("Counts/30s")
plt.title("Pu 238")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.figure()

# --- TL 204 ---
data_Tl = np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Tl.tsv",usecols=(1,2),skiprows=11)
V_Tl = data_Tl[:,0]
C_Tl = data_Tl[:,1]

interp_Tl = interp1d(V_Tl, C_Tl, kind='linear')
V_Tl_fine = np.linspace(min(V_Tl), max(V_Tl), 300)

# --- CO 60 ---
data_Co = np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Co.tsv",usecols=(1,2),skiprows=11)
V_Co = data_Co[:,0]
C_Co = data_Co[:,1]

interp_Co = interp1d(V_Co, C_Co, kind='linear')
V_Co_fine = np.linspace(min(V_Co), max(V_Co), 300)
"""
#################################
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# 1. Definición de la función corregida
def analizar_plateau(V, R):
    sigma = np.sqrt(R)
    V2 = V[-1]
    R2 = R[-1]

    # Pendientes según el guion
    m = np.array([
        2*(R2 - R[i])/(R2 + R[i]) * 1e4/(V2 - V[i])
        for i in range(len(V)-1)
    ])

    # Región plateau
    idx = np.where(m < 10)[0]
    Vp = V[idx]
    Rp = R[idx]

    # Ajuste lineal meseta CON matriz de covarianza para obtener el error
    coef, cov = np.polyfit(Vp, Rp, 1, cov=True)
    err_m = np.sqrt(cov[0, 0]) # Error de la pendiente

    Vfit = np.linspace(min(Vp), max(Vp), 200)
    Rfit = np.polyval(coef, Vfit)

    return sigma, Vp, Rp, Vfit, Rfit, coef, err_m

# ==========================================
# 2. Lectura de datos y Gráficas
# ==========================================

# --- FIGURA 1: Tl 204 y Co 60 ---
plt.figure()

# Tl 204
data_Tl = np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Tl.tsv",usecols=(1,2),skiprows=11)
V_Tl, R_Tl = data_Tl[:,0], data_Tl[:,1]
sigma_Tl, Vp_Tl, Rp_Tl, Vfit_Tl, Rfit_Tl, coef_Tl, err_m_Tl = analizar_plateau(V_Tl, R_Tl)

plt.errorbar(V_Tl, R_Tl, yerr=sigma_Tl, fmt='o', label='Tl 204')
plt.plot(Vfit_Tl, Rfit_Tl, '-', label='Meseta Tl')

# Co 60
data_Co = np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Co.tsv", usecols=(1,2), skiprows=11)
V_Co, R_Co = data_Co[:,0], data_Co[:,1]
sigma_Co, Vp_Co, Rp_Co, Vfit_Co, Rfit_Co, coef_Co, err_m_Co = analizar_plateau(V_Co, R_Co)

plt.errorbar(V_Co, R_Co, yerr=sigma_Co, fmt='s', label='Co 60')
plt.plot(Vfit_Co, Rfit_Co, '--', label='Meseta Co')

plt.xlabel("Voltaje (V)")
plt.ylabel("Cuentas / 30 s")
plt.title("Curva Plateau: Tl 204 y Co 60")
plt.legend()
plt.show()

# --- FIGURA 2: Sr 90 y Pu 238 ---
plt.figure()

# Sr 90
data_Sr = np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_sr.tsv",usecols=(1,2),skiprows=11)
V_Sr, R_Sr = data_Sr[:,0], data_Sr[:,1]
sigma_Sr, Vp_Sr, Rp_Sr, Vfit_Sr, Rfit_Sr, coef_Sr, err_m_Sr = analizar_plateau(V_Sr, R_Sr)

plt.errorbar(V_Sr, R_Sr, yerr=sigma_Sr, fmt='o', label='Sr 90')
plt.plot(Vfit_Sr, Rfit_Sr, '-', label='Meseta Sr')

# Pu 238
data_Pu = np.loadtxt(f"/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/TEIII/archivos_py/segundo_cuatri/GM/archive/Plateau_Pu.tsv",usecols=(1,2),skiprows=11)
V_Pu, R_Pu = data_Pu[:,0], data_Pu[:,1]
sigma_Pu, Vp_Pu, Rp_Pu, Vfit_Pu, Rfit_Pu, coef_Pu, err_m_Pu = analizar_plateau(V_Pu, R_Pu)

plt.errorbar(V_Pu, R_Pu, yerr=sigma_Pu, fmt='s', label='Pu 238')
plt.plot(Vfit_Pu, Rfit_Pu, '--', label='Meseta Pu')

plt.xlabel("Voltaje (V)")
plt.ylabel("Cuentas / 30 s")
plt.title("Curva Plateau: Sr 90 y Pu 238")
plt.legend()
plt.show()

# ==========================================
# 3. Análisis textual en consola
# ==========================================
print("\n--- Análisis de Plateau según el Guion ---")

# Diccionario para automatizar el print de todos los elementos
resultados = {
    "Tl 204": (Vp_Tl, coef_Tl, err_m_Tl),
    "Co 60":  (Vp_Co, coef_Co, err_m_Co),
    "Sr 90":  (Vp_Sr, coef_Sr, err_m_Sr),
    "Pu 238": (Vp_Pu, coef_Pu, err_m_Pu)
}

for muestra, (Vp, coef, err) in resultados.items():
    inicio_pl = min(Vp)
    fin_pl = max(Vp)
    anchura = fin_pl - inicio_pl
    
    # Calcular el rango del 60-80% del plateau requerido en la práctica
    v_60 = inicio_pl + 0.60 * anchura
    v_80 = inicio_pl + 0.80 * anchura
    
    print(f"\n[{muestra}]")
    print(f"  Pendiente ajuste = {coef[0]:.2f} ± {err:.2f} cuentas/V")
    print(f"  Rango Plateau = de {inicio_pl:.0f}V a {fin_pl:.0f}V (Anchura: {anchura:.0f}V)")
    print(f"  Zona de trabajo óptima (60%-80%) = Entre {v_60:.0f}V y {v_80:.0f}V")
    
    # Comprobar si 1000V es un buen voltaje de trabajo según el guion
    if v_60 <= 1000 <= v_80:
        print("  -> ¡Éxito! 1000V cae DENTRO de la zona óptima del 60-80%.")
    else:
        print("  -> OJO: 1000V cae FUERA de la zona óptima del 60-80% sugerida.")
