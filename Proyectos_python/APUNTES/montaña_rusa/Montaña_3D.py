# Modulo de cálculos específicos de la montaña rusa
import calculo_montaña as cmr

# Module de gestión de fechas y rutas
import os

# Modulos de cálculo y gestión simbólica
import numpy as np
import sympy as sp

# Modulos de graficación
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Modulo de interpolación
import scipy.interpolate as scip
from scipy.interpolate import BSpline

# Funcion de resolución de EDOs
from scipy.integrate import solve_ivp

# Modulos de animación
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

# Supresión de warnings
import warnings
warnings.filterwarnings('ignore')

plt.ion()

# Función que genera la expresión paramétrica
def montaña_rusa_parametrica(X_data=None, Z_data=None, Y_data=None, grado=3):
    '''
    Construye la expresión paramétrica r(t) mediante B-splines.
    
    Parámetros:
    -----------
    X_data : array-like
        Coordenadas horizontales (distancia)
    Z_data : array-like
        Coordenadas verticales (altura)
    grado : int
        Grado del B-spline (default=3, cúbico)
    
    Retorna:
    --------
    curva_bspline: expresión parámetrica evaluable con un parámetro [0,1]
    '''
    
    # Como hacemos una función 3D, si falta alguna coordenada la llenamos con ceros
    if X_data is None:
        X_data = np.full(len(Z_data), 0)
    elif Z_data is None:
        Z_data = np.full(len(X_data), 0)
    elif Y_data is None:
        Y_data = np.full(len(X_data), 0)
    
    # definimos los Puntos de control, serán los mismos datos
    puntos_control = np.array([X_data, Y_data, Z_data])  # Cada fila es una coordenada (X, Y, Z)

    # Definimos los nodos, debe cumplir la condición len(nodos) = len(puntos_control) + grado + 1
    nodos = np.array([0]*(grado) + list(np.linspace(0, 1, len(X_data)-grado)) + [1]*(grado)) #Ponemos los nodos al inicio y al final repetidos grado+1 veces para forzar a que la curva pase por los puntos extremos
    
    # Usamos la clase BSpline de scipy.interpolate
    curva_bspline = BSpline(nodos, puntos_control.T, grado)
    
    return curva_bspline




# Definimos la gran bajada de la montaña rusa
Gran_bajada = cmr.curva3d('s',np.linspace(-250,250,100),A=70,args=[0.02])

# giro antes del loop
giro1 = cmr.curva3d('h',np.linspace(0,np.pi/2,20),A=140,C=[Gran_bajada[0][-1]+10,-140,Gran_bajada[2][-1]],args=[0],plano='xy')

# Definimos el primer loop de la montaña rusa
Loop1 = cmr.curva3d('l',np.linspace(0,1.79,20),A=33,C=[giro1[0][0],-giro1[1][0]+40,0],args=[0],plano='yz',paso=-30)

# Definimos montañita (gaussina)
Montañita = cmr.curva3d('g',np.linspace(-250,250,50),A=10,C=[Loop1[0][-1],Loop1[1][-1]+250,0],args=[95],plano='yz')

# giro antes de la pequeña bajada
giro2 = cmr.curva3d('h',np.linspace(np.pi/2,0,20),A=100,C=[Montañita[0][-1]-100,Montañita[1][-1]+20,Montañita[2][-1]],args=[0],plano='xy')

# Definimos la pequeña bajada
Pequeña_bajada = cmr.curva3d('s',np.linspace(-100,100,30),A=30,C=[-giro2[0][0]+110,giro2[1][0],Montañita[2][-1]-30],args=[0.1])

# Definimos el segundo loop de la montaña rusa
Loop2 = cmr.curva3d('l',np.linspace(0,1.79,20),A=30,C=[Pequeña_bajada[0][0]+265,Pequeña_bajada[1][0]+30,Pequeña_bajada[2][-1]],args=[0],paso=-30,plano='xz')

# Recta final frenado
Recta_final = [Loop2[0][0]-np.linspace(1,201,20),np.full(20,Loop2[1][0]),np.full(20,Loop2[2][-1])]

# =============================================
# VISUALIZACIÓN 3D DE LA MONTAÑA RUSA
# ==============================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(Gran_bajada[0], Gran_bajada[1], Gran_bajada[2], 'o', linewidth=2, label='Gran Bajada')

ax.plot(giro1[0], -giro1[1], giro1[2], 'o', linewidth=2, label='Giro antes del Loop 1')

ax.plot(Loop1[0], Loop1[1], Loop1[2], 'o', linewidth=2, label='Loop 1')

ax.plot(Montañita[0], Montañita[1], Montañita[2], 'o', linewidth=2, label='Montañita')

ax.plot(-Pequeña_bajada[0], Pequeña_bajada[1], Pequeña_bajada[2], 'o', linewidth=2, label='Pequeña Bajada')

ax.plot(giro2[0], giro2[1], giro2[2], 'o', linewidth=2, label='Giro antes del Loop 2')

ax.plot(Loop2[0], Loop2[1], Loop2[2], 'o', linewidth=2, label='Loop 2')

ax.plot(Recta_final[0], Recta_final[1], Recta_final[2], 'o', linewidth=2, label='Recta Final')  
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Montaña Rusa 3D')
ax.legend()
plt.show()

# Visualización usando la expresión paramétrica r(t)

#Construimos la expresión
Curva_parametrica = montaña_rusa_parametrica(
    X_data=np.concatenate([Gran_bajada[0],giro1[0][::-1],Loop1[0],Montañita[0],giro2[0][::-1],-Pequeña_bajada[0],Loop2[0][::-1],Recta_final[0]]),
    Z_data=np.concatenate([Gran_bajada[2],giro1[2][::-1],Loop1[2],Montañita[2],giro2[2][::-1],Pequeña_bajada[2],Loop2[2][::-1],Recta_final[2]]),
    Y_data=np.concatenate([Gran_bajada[1],-giro1[1][::-1],Loop1[1],Montañita[1],giro2[1][::-1],Pequeña_bajada[1],Loop2[1][::-1],Recta_final[1]]),
    grado=3
)
# Evaluamos
t_eval = np.linspace(0, 1, 1000)

fig=plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Gráfica comparativa
ax.plot(Curva_parametrica(t_eval)[:,0], Curva_parametrica(t_eval)[:,1], Curva_parametrica(t_eval)[:,2], 'r-', linewidth=2, label='B-spline Paramétrica')
ax.plot(np.concatenate([Gran_bajada[0],giro1[0][::-1],Loop1[0],Montañita[0],giro2[0][::-1],-Pequeña_bajada[0],Loop2[0][::-1],Recta_final[0]]),
        np.concatenate([Gran_bajada[1],-giro1[1][::-1],Loop1[1],Montañita[1],giro2[1][::-1],Pequeña_bajada[1],Loop2[1][::-1],Recta_final[1]]),
        np.concatenate([Gran_bajada[2],giro1[2][::-1],Loop1[2],Montañita[2],giro2[2][::-1],Pequeña_bajada[2],Loop2[2][::-1],Recta_final[2]]),
        'o', markersize=2, label='Puntos de Datos',alpha=0.4)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Montaña Rusa 3D - Curva Paramétrica B-spline')
ax.legend()
plt.show()


g = 9.8 #m/s^2 #aceleración de la gravedad
m = 800 #kg #masa del vagón con 4 pasajeros a bordo
mu = 0.0 #coeficiente de fricción
c_a = 0 #coeficiente de resistencia aerodinámica
S_f = 2 #m^2 #superficie frontal del vagón
ro_a = 1.225 #kg/m^3 #densidad del aire
c_v = (c_a*S_f*ro_a)/(2*m) #m^{-1} #coeficiente de la fuerza viscosa

# Condiciones iniciales [parámetro u0, velocidad v0]
u0 = 0.0              # inicio de la curva
v0 = 7             # m/s - velocidad inicial
y0 = [u0, v0]

#=============================================
# Sistema Sedo
#=============================================

# Calcular derivadas de la curva usando pkgmrusa UNA SOLA VEZ
derivadas_curva = cmr.trayec_der(Curva_parametrica)


def edo_montaña_rusa_conservativa(t, y):
    '''
    Define el sistema de EDOs para la montaña rusa.
    
    Parámetros:
    -----------
    t : float
        Tiempo (no se usa explícitamente en este sistema)
    y : list
        Estado del sistema [posición u, velocidad v]
    
    Retorna:
    --------
    dydt : list
        Derivadas [du/dt, dv/dt]
    '''
    
    return cmr.edofun_mr(t, y, derivadas_curva, mu, c_v, g)


#Información del sistema
print("="*70)
print("SISTEMA EDO - CASO CONSERVATIVO")
print("="*70)
print(f"Parámetros físicos:")
print(f"  • Masa: {m} kg")
print(f"  • Coef. rozamiento μ: {mu} (SIN FRICCIÓN)")
print(f"  • Coef. arrastre ca: {c_a} (SIN ROZAMIENTO)")
print(f"  • Superficie frontal Sf: {S_f} m²")
print(f"  • Coeficiente viscoso cvefVis: {c_v} 1/m")
print(f"  • Gravedad: {g} m/s² (positivo)")
print(f"\nCondiciones iniciales:")
print(f"  • Posición inicial: u₀ = {u0}")
print(f"  • Velocidad inicial: v₀ = {v0} m/s")

# t_eval para solución densa
t_eval = np.linspace(0, 200, 1000)

# Resolver EDO no conservativa
solución_3D = solve_ivp(edo_montaña_rusa_conservativa,[0,200],y0,method="Radau",t_eval=t_eval,dense_output=True,events=[cmr.finalVia,cmr.paradaVagon])

# ==============================================
# GRÁFICAS COMBINADAS: Análisis del movimiento con fricción
# ==============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis Completo de la Montaña Rusa - Caso Conservativo (sin fricción)', fontsize=16, fontweight='bold')

# Generar puntos densos para cálculos
t_dense = np.linspace(solución_3D.t[0], solución_3D.t[-1], 1000)
datos = solución_3D.sol(t_dense)
u_vals = datos[0]
v_vals = datos[1]

# ==============================================
# 1. VELOCIDAD vs TIEMPO (subplot superior izquierdo)
# ==============================================
ax1 = axes[0, 0]
ax1.plot(t_dense, v_vals, linewidth=2, color='darkblue', alpha=0.8, label='Radau')
ax1.set_xlabel('Tiempo (s)', fontsize=11)
ax1.set_ylabel('Velocidad (m/s)', fontsize=11)
ax1.set_title('Velocidad vs Tiempo', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# ==============================================
# 2. ACELERACIÓN TOTAL vs TIEMPO (subplot superior derecho)
# ==============================================
ax2 = axes[0, 1]

aceleraciones = []
for i in range(len(u_vals)):
    fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_vals[i], v_vals[i], derivadas_curva, g)
    ace_total, ace_tg, ace_nr = cmr.aceleracion(v_vals[i], baseLocal, mu, c_v, fuerzaN, ctes[1], g)
    aceleraciones.append(ace_total)

ax2.plot(t_dense, aceleraciones, linewidth=2, color='crimson', alpha=0.8, label='Radau')
ax2.set_xlabel('Tiempo (s)', fontsize=11)
ax2.set_ylabel('Aceleración total (m/s²)', fontsize=11)
ax2.set_title('Aceleración Total vs Tiempo', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# ==============================================
# 3. ENERGÍA vs TIEMPO (subplot inferior izquierdo)
# ==============================================
ax3 = axes[1, 0]

energias = []
for i in range(len(u_vals)):
    E = cmr.energia(u_vals[i], v_vals[i], derivadas_curva[0], g)
    energias.append(E)

ax3.plot(t_dense, energias, linewidth=2, color='green', alpha=0.8, label='Radau')
ax3.set_xlabel('Tiempo (s)', fontsize=11)
ax3.set_ylabel('Energía (J/kg)', fontsize=11)
ax3.set_title('Energía vs Tiempo (constante en caso conservativo)', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# ==============================================
# 4. FUERZA NORMAL vs TIEMPO en unidades de G (subplot inferior derecho)
# ==============================================
ax4 = axes[1, 1]

fuerzas_normales_G = []

for i in range(len(u_vals)):
    fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_vals[i], v_vals[i], derivadas_curva, g)
    # Convertir a unidades de G (1 G = 9.81 m/s²)
    fuerzaN_G = fuerzaN / g
    fuerzas_normales_G.append(fuerzaN_G)

ax4.plot(t_dense, fuerzas_normales_G, linewidth=2, color='darkorange', alpha=0.8, label='Radau')
ax4.set_xlabel('Tiempo (s)', fontsize=11)
ax4.set_ylabel('Fuerza Normal (G)', fontsize=11)
ax4.set_title('Fuerza Normal vs Tiempo', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='F_N = 0')
ax4.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='1 G')
ax4.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# ==============================================
# ANÁLISIS DE PÉRDIDA DE ENERGÍA
# ==============================================
print("\n" + "="*80)
print("ANÁLISIS DE PÉRDIDA DE ENERGÍA - CASO CONSERVATIVO")
print("="*80)
E_inicial = energias[0]
E_final = energias[-1]
perdida_energia = E_inicial - E_final
porcentaje_perdida = (perdida_energia / E_inicial) * 100

print(f"\nEnergía inicial:  {E_inicial:.2f} J/kg")
print(f"Energía final:    {E_final:.2f} J/kg")
print(f"Pérdida total:    {perdida_energia:.2f} J/kg ({porcentaje_perdida:.2f}%)")
print(f"\nVelocidad inicial: {v_vals[0]:.2f} m/s")
print(f"Velocidad final:   {v_vals[-1]:.2f} m/s")
print(f"\n💡 La energía disminuye gradualmente debido a:")
print(f"   • Fricción con la pista (μ = {mu})")
print(f"   • Resistencia aerodinámica (c_v = {c_v:.6f} m⁻¹)")
print("="*80 + "\n")


fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Curva_parametrica(solución_3D.sol(t_dense)[0])[:,0], Curva_parametrica(solución_3D.sol(t_dense)[0])[:,1], Curva_parametrica(solución_3D.sol(t_dense)[0])[:,2], 'b-', linewidth=2, label='Montaña Rusa')
ax.set_title('Trayectoria de la Montaña Rusa - Caso Conservativo (sin fricción)', fontsize=14, fontweight='bold')
ax.set_xlabel('Distancia (m)', fontsize=12)
ax.set_ylabel('Profundidad (m)', fontsize=12)
ax.set_zlabel('Altura (m)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()



g = 9.8 #m/s^2 #aceleración de la gravedad
m = 800 #kg #masa del vagón con 4 pasajeros a bordo
mu = 0.015 #coeficiente de fricción
c_a = 0.4 #coeficiente de resistencia aerodinámica
S_f = 2 #m^2 #superficie frontal del vagón
ro_a = 1.225 #kg/m^3 #densidad del aire
c_v = (c_a*S_f*ro_a)/(2*m) #m^{-1} #coeficiente de la fuerza viscosa

# Condiciones iniciales [parámetro u0, velocidad v0]
u0 = 0.0              # inicio de la curva
v0 = 7             # m/s - velocidad inicial
y0 = [u0, v0]

#=============================================
# Sistema Sedo
#=============================================

# Calcular derivadas de la curva usando pkgmrusa UNA SOLA VEZ
derivadas_curva = cmr.trayec_der(Curva_parametrica)


def edo_montaña_rusa_no_conservativa(t, y):
    '''
    Define el sistema de EDOs para la montaña rusa.
    
    Parámetros:
    -----------
    t : float
        Tiempo (no se usa explícitamente en este sistema)
    y : list
        Estado del sistema [posición u, velocidad v]
    
    Retorna:
    --------
    dydt : list
        Derivadas [du/dt, dv/dt]
    '''
    
    return cmr.edofun_mr(t, y, derivadas_curva, mu, c_v, g)


#Información del sistema
print("="*70)
print("SISTEMA EDO - CASO NO CONSERVATIVO")
print("="*70)
print(f"Parámetros físicos:")
print(f"  • Masa: {m} kg")
print(f"  • Coef. rozamiento μ: {mu} (SIN FRICCIÓN)")
print(f"  • Coef. arrastre ca: {c_a} (SIN ROZAMIENTO)")
print(f"  • Superficie frontal Sf: {S_f} m²")
print(f"  • Coeficiente viscoso cvefVis: {c_v} 1/m")
print(f"  • Gravedad: {g} m/s² (positivo)")
print(f"\nCondiciones iniciales:")
print(f"  • Posición inicial: u₀ = {u0}")
print(f"  • Velocidad inicial: v₀ = {v0} m/s")


# t_eval para solución densa
t_eval = np.linspace(0, 200, 1000)

# Resolver EDO no conservativa
solución_3D_fric = solve_ivp(edo_montaña_rusa_no_conservativa,[0,200],y0,method="Radau",t_eval=t_eval,dense_output=True,events=[cmr.finalVia,cmr.paradaVagon])


# ==============================================
# GRÁFICAS COMBINADAS: Análisis del movimiento con fricción
# ==============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis Completo de la Montaña Rusa - Caso NO Conservativo (con fricción)', fontsize=16, fontweight='bold')

# Generar puntos densos para cálculos
t_dense = np.linspace(solución_3D_fric.t[0], solución_3D_fric.t[-1], 1000)
datos = solución_3D_fric.sol(t_dense)
u_vals = datos[0]
v_vals = datos[1]

# ==============================================
# 1. VELOCIDAD vs TIEMPO (subplot superior izquierdo)
# ==============================================
ax1 = axes[0, 0]
ax1.plot(t_dense, v_vals, linewidth=2, color='darkblue', alpha=0.8, label='Radau')
ax1.set_xlabel('Tiempo (s)', fontsize=11)
ax1.set_ylabel('Velocidad (m/s)', fontsize=11)
ax1.set_title('Velocidad vs Tiempo', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# ==============================================
# 2. ACELERACIÓN TOTAL vs TIEMPO (subplot superior derecho)
# ==============================================
ax2 = axes[0, 1]

aceleraciones = []
for i in range(len(u_vals)):
    fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_vals[i], v_vals[i], derivadas_curva, g)
    ace_total, ace_tg, ace_nr = cmr.aceleracion(v_vals[i], baseLocal, mu, c_v, fuerzaN, ctes[1], g)
    aceleraciones.append(ace_total)

ax2.plot(t_dense, aceleraciones, linewidth=2, color='crimson', alpha=0.8, label='Radau')
ax2.set_xlabel('Tiempo (s)', fontsize=11)
ax2.set_ylabel('Aceleración total (m/s²)', fontsize=11)
ax2.set_title('Aceleración Total vs Tiempo', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# ==============================================
# 3. ENERGÍA vs TIEMPO (subplot inferior izquierdo)
# ==============================================
ax3 = axes[1, 0]

energias = []
for i in range(len(u_vals)):
    E = cmr.energia(u_vals[i], v_vals[i], derivadas_curva[0], g)
    energias.append(E)

ax3.plot(t_dense, energias, linewidth=2, color='green', alpha=0.8, label='Radau')
ax3.set_xlabel('Tiempo (s)', fontsize=11)
ax3.set_ylabel('Energía (J/kg)', fontsize=11)
ax3.set_title('Energía vs Tiempo (disminuye por fricción)', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# ==============================================
# 4. FUERZA NORMAL vs TIEMPO en unidades de G (subplot inferior derecho)
# ==============================================
ax4 = axes[1, 1]

fuerzas_normales_G = []

for i in range(len(u_vals)):
    fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_vals[i], v_vals[i], derivadas_curva, g)
    # Convertir a unidades de G (1 G = 9.81 m/s²)
    fuerzaN_G = fuerzaN / g
    fuerzas_normales_G.append(fuerzaN_G)

ax4.plot(t_dense, fuerzas_normales_G, linewidth=2, color='darkorange', alpha=0.8, label='Radau')
ax4.set_xlabel('Tiempo (s)', fontsize=11)
ax4.set_ylabel('Fuerza Normal (G)', fontsize=11)
ax4.set_title('Fuerza Normal vs Tiempo', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='F_N = 0')
ax4.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='1 G')
ax4.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# ==============================================
# ANÁLISIS DE PÉRDIDA DE ENERGÍA
# ==============================================
print("\n" + "="*80)
print("ANÁLISIS DE PÉRDIDA DE ENERGÍA - CASO CON FRICCIÓN")
print("="*80)
E_inicial = energias[0]
E_final = energias[-1]
perdida_energia = E_inicial - E_final
porcentaje_perdida = (perdida_energia / E_inicial) * 100

print(f"\nEnergía inicial:  {E_inicial:.2f} J/kg")
print(f"Energía final:    {E_final:.2f} J/kg")
print(f"Pérdida total:    {perdida_energia:.2f} J/kg ({porcentaje_perdida:.2f}%)")
print(f"\nVelocidad inicial: {v_vals[0]:.2f} m/s")
print(f"Velocidad final:   {v_vals[-1]:.2f} m/s")
print(f"\n💡 La energía disminuye gradualmente debido a:")
print(f"   • Fricción con la pista (μ = {mu})")
print(f"   • Resistencia aerodinámica (c_v = {c_v:.6f} m⁻¹)")
print("="*80 + "\n")


fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Curva_parametrica(solución_3D_fric.sol(t_dense)[0])[:,0], Curva_parametrica(solución_3D_fric.sol(t_dense)[0])[:,1], Curva_parametrica(solución_3D_fric.sol(t_dense)[0])[:,2], 'b-', linewidth=2, label='Montaña Rusa')
ax.set_title('Trayectoria de la Montaña Rusa - Caso NO Conservativo (con fricción)', fontsize=14, fontweight='bold')
ax.set_xlabel('Distancia (m)', fontsize=12)
ax.set_ylabel('Profundidad (m)', fontsize=12)
ax.set_zlabel('Altura (m)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()


# ==============================================
# CASO VIABLE
# ==============================================
# Definimos la gran bajada de la montaña rusa
Gran_bajada = cmr.curva3d('s',np.linspace(-250,250,100),A=70,args=[0.02])

# giro antes del loop
giro1 = cmr.curva3d('h',np.linspace(0,np.pi/2,20),A=140,C=[Gran_bajada[0][-1]+10,-140,Gran_bajada[2][-1]],args=[0],plano='xy')

# Definimos el primer loop de la montaña rusa
Loop1 = cmr.curva3d('l',np.linspace(0,1.79,20),A=33,C=[giro1[0][0],-giro1[1][0]+37,0],args=[0],plano='yz',paso=-10)

# Definimos montañita (gaussina)
Montañita = cmr.curva3d('g',np.linspace(-250,250,50),A=10,C=[Loop1[0][-1],Loop1[1][-1]+250,0],args=[95],plano='yz')

# giro antes de la pequeña bajada
giro2 = cmr.curva3d('h',np.linspace(np.pi/2,0,20),A=100,C=[Montañita[0][-1]-100,Montañita[1][-1]+20,Montañita[2][-1]],args=[0],plano='xy')

# Definimos la pequeña bajada
Pequeña_bajada = cmr.curva3d('s',np.linspace(-100,100,30),A=30,C=[-giro2[0][0]+110,giro2[1][0],Montañita[2][-1]-30],args=[0.1])

# Definimos el segundo loop de la montaña rusa
Loop2 = cmr.curva3d('l',np.linspace(0,1.79,20),A=25,C=[Pequeña_bajada[0][0]+265,Pequeña_bajada[1][0]+30,Pequeña_bajada[2][-1]+2],args=[0],paso=-10,plano='xz')

# Recta final frenado
Recta_final = [Loop2[0][0]-np.linspace(1,201,20),np.full(20,Loop2[1][0]),np.full(20,Loop2[2][-1])]


# Visualización usando la expresión paramétrica r(t)

#Construimos la expresión
Curva_parametrica = montaña_rusa_parametrica(
    X_data=np.concatenate([Gran_bajada[0],giro1[0][::-1],Loop1[0],Montañita[0],giro2[0][::-1],-Pequeña_bajada[0],Loop2[0][::-1],Recta_final[0]]),
    Z_data=np.concatenate([Gran_bajada[2],giro1[2][::-1],Loop1[2],Montañita[2],giro2[2][::-1],Pequeña_bajada[2],Loop2[2][::-1],Recta_final[2]]),
    Y_data=np.concatenate([Gran_bajada[1],-giro1[1][::-1],Loop1[1],Montañita[1],giro2[1][::-1],Pequeña_bajada[1],Loop2[1][::-1],Recta_final[1]]),
    grado=3
)
# Evaluamos
t_eval = np.linspace(0, 1, 1000)

fig=plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Gráfica comparativa
ax.plot(Curva_parametrica(t_eval)[:,0], Curva_parametrica(t_eval)[:,1], Curva_parametrica(t_eval)[:,2], 'r-', linewidth=2, label='B-spline Paramétrica')

ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Montaña Rusa 3D - Curva Paramétrica B-spline')
ax.legend()
plt.show()


g = 9.8 #m/s^2 #aceleración de la gravedad
m = 800 #kg #masa del vagón con 4 pasajeros a bordo
mu = 0.015 #coeficiente de fricción
c_a = 0.4 #coeficiente de resistencia aerodinámica
S_f = 2 #m^2 #superficie frontal del vagón
ro_a = 1.225 #kg/m^3 #densidad del aire
c_v = (c_a*S_f*ro_a)/(2*m) #m^{-1} #coeficiente de la fuerza viscosa

# Condiciones iniciales [parámetro u0, velocidad v0]
u0 = 0.0              # inicio de la curva
v0 = 11.8        # m/s - velocidad inicial
y0 = [u0, v0]

#=============================================
# Sistema Sedo
#=============================================

# Calcular derivadas de la curva usando pkgmrusa UNA SOLA VEZ
derivadas_curva = cmr.trayec_der(Curva_parametrica)


def edo_montaña_rusa_no_conservativa(t, y):
    '''
    Define el sistema de EDOs para la montaña rusa.
    
    Parámetros:
    -----------
    t : float
        Tiempo (no se usa explícitamente en este sistema)
    y : list
        Estado del sistema [posición u, velocidad v]
    
    Retorna:
    --------
    dydt : list
        Derivadas [du/dt, dv/dt]
    '''
    
    return cmr.edofun_mr(t, y, derivadas_curva, mu, c_v, g)


#Información del sistema
print("="*70)
print("SISTEMA EDO - CASO NO CONSERVATIVO")
print("="*70)
print(f"Parámetros físicos:")
print(f"  • Masa: {m} kg")
print(f"  • Coef. rozamiento μ: {mu} (SIN FRICCIÓN)")
print(f"  • Coef. arrastre ca: {c_a} (SIN ROZAMIENTO)")
print(f"  • Superficie frontal Sf: {S_f} m²")
print(f"  • Coeficiente viscoso cvefVis: {c_v} 1/m")
print(f"  • Gravedad: {g} m/s² (positivo)")
print(f"\nCondiciones iniciales:")
print(f"  • Posición inicial: u₀ = {u0}")
print(f"  • Velocidad inicial: v₀ = {v0} m/s")

# Tiempo de evaluación
t_eval = np.linspace(0, 300, 600)

# Solución del sistema EDOs
sol_viable = solve_ivp(edo_montaña_rusa_no_conservativa,[0,300],y0,method="Radau",t_eval=t_eval,dense_output=True,events=[cmr.finalVia,cmr.paradaVagon],max_step=0.1)


# ==============================================
# GRÁFICAS COMBINADAS: Análisis del movimiento con fricción
# ==============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Análisis Completo de la Montaña Rusa - Caso NO Conservativo (con fricción)', fontsize=16, fontweight='bold')

# Generar puntos densos para cálculos
t_dense = np.linspace(sol_viable.t[0], sol_viable.t[-1], 1000)
datos = sol_viable.sol(t_dense)
u_vals = datos[0]
v_vals = datos[1]

# ==============================================
# 1. VELOCIDAD vs TIEMPO (subplot superior izquierdo)
# ==============================================
ax1 = axes[0, 0]
ax1.plot(t_dense, v_vals, linewidth=2, color='darkblue', alpha=0.8, label='Radau')
ax1.set_xlabel('Tiempo (s)', fontsize=11)
ax1.set_ylabel('Velocidad (m/s)', fontsize=11)
ax1.set_title('Velocidad vs Tiempo', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# ==============================================
# 2. ACELERACIÓN TOTAL vs TIEMPO (subplot superior derecho)
# ==============================================
ax2 = axes[0, 1]

aceleraciones = []
for i in range(len(u_vals)):
    fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_vals[i], v_vals[i], derivadas_curva, g)
    ace_total, ace_tg, ace_nr = cmr.aceleracion(v_vals[i], baseLocal, mu, c_v, fuerzaN, ctes[1], g)
    aceleraciones.append(ace_total)

ax2.plot(t_dense, aceleraciones, linewidth=2, color='crimson', alpha=0.8, label='Radau')
ax2.set_xlabel('Tiempo (s)', fontsize=11)
ax2.set_ylabel('Aceleración total (m/s²)', fontsize=11)
ax2.set_title('Aceleración Total vs Tiempo', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# ==============================================
# 3. ENERGÍA vs TIEMPO (subplot inferior izquierdo)
# ==============================================
ax3 = axes[1, 0]

energias = []
for i in range(len(u_vals)):
    E = cmr.energia(u_vals[i], v_vals[i], derivadas_curva[0], g)
    energias.append(E)

ax3.plot(t_dense, energias, linewidth=2, color='green', alpha=0.8, label='Radau')
ax3.set_xlabel('Tiempo (s)', fontsize=11)
ax3.set_ylabel('Energía (J/kg)', fontsize=11)
ax3.set_title('Energía vs Tiempo (disminuye por fricción)', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# ==============================================
# 4. FUERZA NORMAL vs TIEMPO en unidades de G (subplot inferior derecho)
# ==============================================
ax4 = axes[1, 1]

fuerzas_normales_G = []

for i in range(len(u_vals)):
    fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_vals[i], v_vals[i], derivadas_curva, g)
    # Convertir a unidades de G (1 G = 9.81 m/s²)
    fuerzaN_G = fuerzaN / g
    fuerzas_normales_G.append(fuerzaN_G)

ax4.plot(t_dense, fuerzas_normales_G, linewidth=2, color='darkorange', alpha=0.8, label='Radau')
ax4.set_xlabel('Tiempo (s)', fontsize=11)
ax4.set_ylabel('Fuerza Normal (G)', fontsize=11)
ax4.set_title('Fuerza Normal vs Tiempo', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='F_N = 0')
ax4.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='1 G')
ax4.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# ==============================================
# ANÁLISIS DE PÉRDIDA DE ENERGÍA
# ==============================================
print("\n" + "="*80)
print("ANÁLISIS DE PÉRDIDA DE ENERGÍA - CASO CON FRICCIÓN")
print("="*80)
E_inicial = energias[0]
E_final = energias[-1]
perdida_energia = E_inicial - E_final
porcentaje_perdida = (perdida_energia / E_inicial) * 100

print(f"\nEnergía inicial:  {E_inicial:.2f} J/kg")
print(f"Energía final:    {E_final:.2f} J/kg")
print(f"Pérdida total:    {perdida_energia:.2f} J/kg ({porcentaje_perdida:.2f}%)")
print(f"\nVelocidad inicial: {v_vals[0]:.2f} m/s")
print(f"Velocidad final:   {v_vals[-1]:.2f} m/s")
print(f"\n💡 La energía disminuye gradualmente debido a:")
print(f"   • Fricción con la pista (μ = {mu})")
print(f"   • Resistencia aerodinámica (c_v = {c_v:.6f} m⁻¹)")
print("="*80 + "\n")


fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Curva_parametrica(sol_viable.sol(t_dense)[0])[:,0], Curva_parametrica(sol_viable.sol(t_dense)[0])[:,1], Curva_parametrica(sol_viable.sol(t_dense)[0])[:,2], 'b-', linewidth=2, label='Montaña Rusa')
ax.set_title('Trayectoria de la Montaña Rusa - Caso NO Conservativo (con fricción)', fontsize=14, fontweight='bold')
ax.set_xlabel('Distancia (m)', fontsize=12)
ax.set_ylabel('Profundidad (m)', fontsize=12)
ax.set_zlabel('Altura (m)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()


# ==============================================
# ESTUDIO DE PARÁMETROS VÁLIDOS
# ==============================================

def simular_montaña(m_val, mu_val, ca_val, Sf_val, v0_val, tiempo_max=300):
    """
    Simula la montaña rusa con los parámetros dados y verifica las condiciones.
    
    Retorna:
    --------
    dict con:
        - 'valido': bool (si cumple ambas condiciones)
        - 'completa_recorrido': bool
        - 'FN_max_G': float (fuerza normal máxima en unidades de G)
        - 'u_final': float (posición final alcanzada)
        - 'sol': objeto solución (si fue exitosa)
    """
    
    # Calcular coeficiente viscoso
    c_v_sim = (ca_val * Sf_val * ro_a) / (2 * m_val)
    

    # Condiciones iniciales
    y0_sim = [0.0, v0_val]
    
    # Sistema EDO
    def edo_sim(t, y):
        return cmr.edofun_mr(t, y, derivadas_curva, mu_val, c_v_sim, g)
    
    # Resolver
    try:
        sol_sim = solve_ivp(edo_sim, [0, tiempo_max], y0_sim, 
                            method="Radau", dense_output=True, 
                            events=[cmr.finalVia,cmr.paradaVagon],max_step=0.1,t_eval=np.linspace(0, tiempo_max, 600))  # Se detiene si velocidad llega a 0
        
        # Verificar si completó el recorrido (u_final >= 0.95)
        u_final = sol_sim.sol(sol_sim.t[-1])[0]
        completa = u_final >= 0.95
        
        # Calcular fuerza normal máxima
        t_check = np.linspace(sol_sim.t[0], sol_sim.t[-1], 500)
        datos_check = sol_sim.sol(t_check)
        u_check = datos_check[0]
        v_check = datos_check[1]
        
        FN_max = 0
        for i in range(len(u_check)):
            if u_check[i] >= 1.0:  # No revisar más allá del recorrido
                break
            fuerzaN, _, _ = cmr.fuerzaNormal(u_check[i], v_check[i], derivadas_curva, g)
            FN_G = fuerzaN / g
            if FN_G > FN_max:
                FN_max = FN_G
        
        # Verificar condiciones
        FN_ok = FN_max <= 5.0
        valido = completa and FN_ok
        
        return {
            'valido': valido,
            'completa_recorrido': completa,
            'FN_max_G': FN_max,
            'u_final': u_final,
            'sol': sol_sim,
            'FN_ok': FN_ok
        }
        
    except Exception as e:
        return {
            'valido': False,
            'completa_recorrido': False,
            'FN_max_G': np.inf,
            'u_final': 0,
            'sol': None,
            'FN_ok': False,
            'error': str(e)
        }

print("="*80)
print("FUNCIÓN DE SIMULACIÓN CREADA")
print("="*80)
print("Condiciones para validez:")
print("  1. Completar el recorrido: u_final ≥ 0.95")
print("  2. Fuerza normal máxima: FN_max ≤ 5g")
print("="*80)


# ==============================================
# ESTUDIO 1: Variación de MASA (m)
# ==============================================

# Parámetros base (del caso viable)
mu_base = 0.015
ca_base = 0.4
Sf_base = 2.0
v0_base = 11.8

# Rango de masas a explorar (kg)
masas = np.linspace(200, 2000, 50)

resultados_masa = []
for m_test in masas:
    resultado = simular_montaña(m_test, mu_base, ca_base, Sf_base, v0_base)
    resultados_masa.append({
        'm': m_test,
        'valido': resultado['valido'],
        'completa': resultado['completa_recorrido'],
        'FN_max': resultado['FN_max_G'],
        'u_final': resultado['u_final']
    })

# Análisis de resultados
masas_validas = [r['m'] for r in resultados_masa if r['valido']]
masas_completan = [r['m'] for r in resultados_masa if r['completa']]
masas_FN_ok = [r['m'] for r in resultados_masa if r['FN_max'] <= 5.0]

print("\n" + "="*80)
print("ESTUDIO 1: VARIACIÓN DE MASA (m)")
print("="*80)
print(f"Parámetros fijos: μ={mu_base}, c_a={ca_base}, S_f={Sf_base} m², v₀={v0_base} m/s")
print(f"\nRango explorado: {masas.min():.0f} - {masas.max():.0f} kg")
if masas_validas:
    print(f"✅ Rango válido: {min(masas_validas):.1f} - {max(masas_validas):.1f} kg")
else:
    print("❌ No hay valores válidos en el rango explorado")
print("="*80)

# Gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FN_max vs masa
FN_values = [r['FN_max'] for r in resultados_masa]
masas_array = [r['m'] for r in resultados_masa]
ax1.plot(masas_array, FN_values, 'b-', linewidth=2)
ax1.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Límite F_N = 5g')
ax1.fill_between(masas_array, 0, 5, alpha=0.2, color='green', label='Zona válida F_N')
ax1.set_xlabel('Masa (kg)', fontsize=11)
ax1.set_ylabel('F_N máxima (g)', fontsize=11)
ax1.set_title('Fuerza Normal Máxima vs Masa', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posición final vs masa
u_final_values = [r['u_final'] for r in resultados_masa]
ax2.plot(masas_array, u_final_values, 'g-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Mínimo para completar (0.95)')
ax2.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, label='Final del recorrido (1.0)')
ax2.fill_between(masas_array, 0.95, 1.1, alpha=0.2, color='green', label='Zona válida')
ax2.set_xlabel('Masa (kg)', fontsize=11)
ax2.set_ylabel('Posición final (u)', fontsize=11)
ax2.set_title('Posición Final vs Masa', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================================
# ESTUDIO 2: Variación de COEFICIENTE DE FRICCIÓN (μ)
# ==============================================

# Parámetros base
m_base = 800
ca_base = 0.4
Sf_base = 2.0
v0_base = 11.8

# Rango de μ a explorar
mu_values = np.linspace(0.005, 0.04, 50)

resultados_mu = []
for mu_test in mu_values:
    resultado = simular_montaña(m_base, mu_test, ca_base, Sf_base, v0_base)
    resultados_mu.append({
        'mu': mu_test,
        'valido': resultado['valido'],
        'completa': resultado['completa_recorrido'],
        'FN_max': resultado['FN_max_G'],
        'u_final': resultado['u_final']
    })

# Análisis
mu_validos = [r['mu'] for r in resultados_mu if r['valido']]

print("\n" + "="*80)
print("ESTUDIO 2: VARIACIÓN DE COEFICIENTE DE FRICCIÓN (μ)")
print("="*80)
print(f"Parámetros fijos: m={m_base} kg, c_a={ca_base}, S_f={Sf_base} m², v₀={v0_base} m/s")
print(f"\nRango explorado: {mu_values.min():.4f} - {mu_values.max():.4f}")
if mu_validos:
    print(f"✅ Rango válido: {min(mu_validos):.4f} - {max(mu_validos):.4f}")
else:
    print("❌ No hay valores válidos en el rango explorado")
print("="*80)

# Gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FN_max vs μ
FN_values = [r['FN_max'] for r in resultados_mu]
mu_array = [r['mu'] for r in resultados_mu]
ax1.plot(mu_array, FN_values, 'b-', linewidth=2)
ax1.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Límite F_N = 5g')
ax1.fill_between(mu_array, 0, 5, alpha=0.2, color='green', label='Zona válida F_N')
ax1.set_xlabel('Coeficiente de fricción μ', fontsize=11)
ax1.set_ylabel('F_N máxima (g)', fontsize=11)
ax1.set_title('Fuerza Normal Máxima vs μ', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posición final vs μ
u_final_values = [r['u_final'] for r in resultados_mu]
ax2.plot(mu_array, u_final_values, 'g-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Mínimo para completar (0.95)')
ax2.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, label='Final del recorrido (1.0)')
ax2.fill_between(mu_array, 0.95, 1.1, alpha=0.2, color='green', label='Zona válida')
ax2.set_xlabel('Coeficiente de fricción μ', fontsize=11)
ax2.set_ylabel('Posición final (u)', fontsize=11)
ax2.set_title('Posición Final vs μ', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================================
# ESTUDIO 3: Variación de COEFICIENTE AERODINÁMICO (c_a)
# ==============================================

# Parámetros base
m_base = 800
mu_base = 0.015
Sf_base = 2.0
v0_base = 11.8

# Rango de c_a a explorar
ca_values = np.linspace(0.1, 0.6, 50)

resultados_ca = []
for ca_test in ca_values:
    resultado = simular_montaña(m_base, mu_base, ca_test, Sf_base, v0_base)
    resultados_ca.append({
        'ca': ca_test,
        'valido': resultado['valido'],
        'completa': resultado['completa_recorrido'],
        'FN_max': resultado['FN_max_G'],
        'u_final': resultado['u_final']
    })

# Análisis
ca_validos = [r['ca'] for r in resultados_ca if r['valido']]

print("\n" + "="*80)
print("ESTUDIO 3: VARIACIÓN DE COEFICIENTE AERODINÁMICO (c_a)")
print("="*80)
print(f"Parámetros fijos: m={m_base} kg, μ={mu_base}, S_f={Sf_base} m², v₀={v0_base} m/s")
print(f"\nRango explorado: {ca_values.min():.3f} - {ca_values.max():.3f}")
if ca_validos:
    print(f"✅ Rango válido: {min(ca_validos):.3f} - {max(ca_validos):.3f}")
else:
    print("❌ No hay valores válidos en el rango explorado")
print("="*80)

# Gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FN_max vs c_a
FN_values = [r['FN_max'] for r in resultados_ca]
ca_array = [r['ca'] for r in resultados_ca]
ax1.plot(ca_array, FN_values, 'b-', linewidth=2)
ax1.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Límite F_N = 5g')
ax1.fill_between(ca_array, 0, 5, alpha=0.2, color='green', label='Zona válida F_N')
ax1.set_xlabel('Coeficiente aerodinámico c_a', fontsize=11)
ax1.set_ylabel('F_N máxima (g)', fontsize=11)
ax1.set_title('Fuerza Normal Máxima vs c_a', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posición final vs c_a
u_final_values = [r['u_final'] for r in resultados_ca]
ax2.plot(ca_array, u_final_values, 'g-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Mínimo para completar (0.95)')
ax2.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, label='Final del recorrido (1.0)')
ax2.fill_between(ca_array, 0.95, 1.1, alpha=0.2, color='green', label='Zona válida')
ax2.set_xlabel('Coeficiente aerodinámico c_a', fontsize=11)
ax2.set_ylabel('Posición final (u)', fontsize=11)
ax2.set_title('Posición Final vs c_a', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================================
# ESTUDIO 4: Variación de SUPERFICIE FRONTAL (S_f)
# ==============================================

# Parámetros base
m_base = 800
mu_base = 0.015
ca_base = 0.4
v0_base = 11.8

# Rango de S_f a explorar (m²)
Sf_values = np.linspace(0.5, 3.0, 50)

resultados_Sf = []
for Sf_test in Sf_values:
    resultado = simular_montaña(m_base, mu_base, ca_base, Sf_test, v0_base)
    resultados_Sf.append({
        'Sf': Sf_test,
        'valido': resultado['valido'],
        'completa': resultado['completa_recorrido'],
        'FN_max': resultado['FN_max_G'],
        'u_final': resultado['u_final']
    })

# Análisis
Sf_validos = [r['Sf'] for r in resultados_Sf if r['valido']]

print("\n" + "="*80)
print("ESTUDIO 4: VARIACIÓN DE SUPERFICIE FRONTAL (S_f)")
print("="*80)
print(f"Parámetros fijos: m={m_base} kg, μ={mu_base}, c_a={ca_base}, v₀={v0_base} m/s")
print(f"\nRango explorado: {Sf_values.min():.2f} - {Sf_values.max():.2f} m²")
if Sf_validos:
    print(f"✅ Rango válido: {min(Sf_validos):.2f} - {max(Sf_validos):.2f} m²")
else:
    print("❌ No hay valores válidos en el rango explorado")
print("="*80)

# Gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FN_max vs S_f
FN_values = [r['FN_max'] for r in resultados_Sf]
Sf_array = [r['Sf'] for r in resultados_Sf]
ax1.plot(Sf_array, FN_values, 'b-', linewidth=2)
ax1.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Límite F_N = 5g')
ax1.fill_between(Sf_array, 0, 5, alpha=0.2, color='green', label='Zona válida F_N')
ax1.set_xlabel('Superficie frontal S_f (m²)', fontsize=11)
ax1.set_ylabel('F_N máxima (g)', fontsize=11)
ax1.set_title('Fuerza Normal Máxima vs S_f', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posición final vs S_f
u_final_values = [r['u_final'] for r in resultados_Sf]
ax2.plot(Sf_array, u_final_values, 'g-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Mínimo para completar (0.95)')
ax2.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, label='Final del recorrido (1.0)')
ax2.fill_between(Sf_array, 0.95, 1.1, alpha=0.2, color='green', label='Zona válida')
ax2.set_xlabel('Superficie frontal S_f (m²)', fontsize=11)
ax2.set_ylabel('Posición final (u)', fontsize=11)
ax2.set_title('Posición Final vs S_f', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================================
# ESTUDIO 5: Variación de VELOCIDAD INICIAL (v_0)
# ==============================================

# Parámetros base
m_base = 800
mu_base = 0.015
ca_base = 0.4
Sf_base = 2.0

# Rango de v_0 a explorar (m/s)
v0_values = np.linspace(1.0, 50, 100)

resultados_v0 = []
for v0_test in v0_values:
    resultado = simular_montaña(m_base, mu_base, ca_base, Sf_base, v0_test)
    resultados_v0.append({
        'v0': v0_test,
        'valido': resultado['valido'],
        'completa': resultado['completa_recorrido'],
        'FN_max': resultado['FN_max_G'],
        'u_final': resultado['u_final']
    })

# Análisis
v0_validos = [r['v0'] for r in resultados_v0 if r['valido']]

print("\n" + "="*80)
print("ESTUDIO 5: VARIACIÓN DE VELOCIDAD INICIAL (v_0)")
print("="*80)
print(f"Parámetros fijos: m={m_base} kg, μ={mu_base}, c_a={ca_base}, S_f={Sf_base} m²")
print(f"\nRango explorado: {v0_values.min():.2f} - {v0_values.max():.2f} m/s")
if v0_validos:
    print(f"✅ Rango válido: {min(v0_validos):.2f} - {max(v0_validos):.2f} m/s")
else:
    print("❌ No hay valores válidos en el rango explorado")
print("="*80)

# Gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# FN_max vs v_0
FN_values = [r['FN_max'] for r in resultados_v0]
v0_array = [r['v0'] for r in resultados_v0]
ax1.plot(v0_array, FN_values, 'b-', linewidth=2)
ax1.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Límite F_N = 5g')
ax1.fill_between(v0_array, 0, 5, alpha=0.2, color='green', label='Zona válida F_N')
ax1.set_xlabel('Velocidad inicial v_0 (m/s)', fontsize=11)
ax1.set_ylabel('F_N máxima (g)', fontsize=11)
ax1.set_title('Fuerza Normal Máxima vs v_0', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posición final vs v_0
u_final_values = [r['u_final'] for r in resultados_v0]
ax2.plot(v0_array, u_final_values, 'g-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Mínimo para completar (0.95)')
ax2.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, label='Final del recorrido (1.0)')
ax2.fill_between(v0_array, 0.95, 1.1, alpha=0.2, color='green', label='Zona válida')
ax2.set_xlabel('Velocidad inicial v_0 (m/s)', fontsize=11)
ax2.set_ylabel('Posición final (u)', fontsize=11)
ax2.set_title('Posición Final vs v_0', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ======================================================================
# 1. PREPARAR DATOS PARA ANIMACIÓN
# ======================================================================

print("="*80)
print("PREPARANDO DATOS PARA ANIMACIÓN 3D")
print("="*80)

# Parámetros de interpolación
t_max_sim = sol_viable.t[-1]
n_frames = 300
t_anim = np.linspace(0, t_max_sim, n_frames)

# Datos interpolados de la solución
u_anim = sol_viable.sol(t_anim)[0]
v_anim = sol_viable.sol(t_anim)[1]

# Convertir u -> posiciones 3D (x, y, z)
pos_anim = np.array([Curva_parametrica(u) for u in u_anim])

# Calcular magnitudes para cada frame
energias_anim = []
fuerzas_G_anim = []
aceleraciones_anim = []

for i in range(n_frames):
    # Energía
    E_frame = cmr.energia(u_anim[i], v_anim[i], derivadas_curva[0], g)
    energias_anim.append(E_frame)
    
    # Fuerza Normal
    try:
        fuerzaN, baseLocal, ctes = cmr.fuerzaNormal(u_anim[i], v_anim[i], derivadas_curva, g)
        fuerzas_G_anim.append(fuerzaN / g)
    except:
        fuerzas_G_anim.append(0)
    
    # Aceleración
    aceleraciones_anim.append(v_anim[i])

print(f"✓ Frames totales: {n_frames}")
print(f"✓ Duración simulación: {t_max_sim:.2f} segundos")
print(f"✓ Rango de posición: u ∈ [0, {u_anim[-1]:.3f}]")
print(f"✓ Velocidad: v ∈ [{v_anim.min():.2f}, {v_anim.max():.2f}] m/s")
print(f"✓ F_N máxima: {max(fuerzas_G_anim):.2f} g")
print("="*80 + "\n")

# ======================================================================
# 2. CREAR FIGURA 3D CON SUBPLOTS
# ======================================================================

fig = plt.figure(figsize=(18, 11))
fig.suptitle('Montaña Rusa 3D - Animación Interactiva del Movimiento', 
             fontsize=16, fontweight='bold', y=0.98)

# Subplot principal: Vista 3D (ocupa 2x2)
ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')

# Dibujar pista completa
t_pista = np.linspace(0, 1, 1500)
pista_3d = np.array([Curva_parametrica(u) for u in t_pista])

ax_3d.plot(pista_3d[:, 0], pista_3d[:, 1], pista_3d[:, 2], 
           'k-', linewidth=2.5, alpha=0.3, label='Pista completa', zorder=1)

# Elementos animables en 3D
vagon_3d, = ax_3d.plot([], [], [], 'o', color='#FF0000', markersize=12, 
                       label='Vagón', zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)
estela_3d, = ax_3d.plot([], [], [], '--', color='#FF6B6B', linewidth=1.2, 
                        alpha=0.6, label='Trayectoria reciente', zorder=2)

# Configuración de ejes 3D
ax_3d.set_xlabel('X (m)', fontsize=10, fontweight='bold')
ax_3d.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
ax_3d.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
ax_3d.set_title('Vista 3D del Movimiento', fontsize=12, fontweight='bold', pad=15)
ax_3d.legend(loc='upper left', fontsize=9)

# Limites
margen_x = 40
margen_y = 40
margen_z = 15
ax_3d.set_xlim(pista_3d[:, 0].min() - margen_x, pista_3d[:, 0].max() + margen_x)
ax_3d.set_ylim(pista_3d[:, 1].min() - margen_y, pista_3d[:, 1].max() + margen_y)
ax_3d.set_zlim(pista_3d[:, 2].min() - margen_z, pista_3d[:, 2].max() + margen_z)

# ======================================================================
# Subplot 2: Velocidad vs Tiempo
# ======================================================================

ax_vel = fig.add_subplot(2, 3, 2)
ax_vel.plot(t_anim, v_anim, 'b-', linewidth=2.5, alpha=0.8, label='Velocidad')
ax_vel.fill_between(t_anim, 0, v_anim, alpha=0.15, color='blue')
linea_tiempo_vel, = ax_vel.plot([], [], 'o', color='red', markersize=8, label='Posición actual')
ax_vel.set_xlabel('Tiempo (s)', fontsize=10, fontweight='bold')
ax_vel.set_ylabel('Velocidad (m/s)', fontsize=10, fontweight='bold')
ax_vel.set_title('Velocidad vs Tiempo', fontsize=11, fontweight='bold')
ax_vel.grid(True, alpha=0.3, linestyle='--')
ax_vel.legend(fontsize=9, loc='best')
ax_vel.set_xlim(0, t_max_sim)

# ======================================================================
# Subplot 3: Fuerza Normal vs Tiempo
# ======================================================================

ax_fn = fig.add_subplot(2, 3, 3)
ax_fn.plot(t_anim, fuerzas_G_anim, 'g-', linewidth=2.5, alpha=0.8, label='Fuerza Normal (F_N)')
ax_fn.fill_between(t_anim, 0, fuerzas_G_anim, alpha=0.15, color='green')
ax_fn.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Límite (5g)', zorder=3)
ax_fn.fill_between(t_anim, 0, 5, alpha=0.08, color='red')
linea_tiempo_fn, = ax_fn.plot([], [], 'o', color='red', markersize=8, label='Posición actual')
ax_fn.set_xlabel('Tiempo (s)', fontsize=10, fontweight='bold')
ax_fn.set_ylabel('Fuerza Normal (g)', fontsize=10, fontweight='bold')
ax_fn.set_title('Fuerza Normal vs Tiempo', fontsize=11, fontweight='bold')
ax_fn.grid(True, alpha=0.3, linestyle='--')
ax_fn.legend(fontsize=9, loc='best')
ax_fn.set_xlim(0, t_max_sim)
ax_fn.set_ylim(min(fuerzas_G_anim) - 0.5, 6)

# ======================================================================
# Subplot 4: Energía vs Tiempo
# ======================================================================

ax_E = fig.add_subplot(2, 3, 5)
ax_E.plot(t_anim, energias_anim, 'm-', linewidth=2.5, alpha=0.8, label='Energía Total')
ax_E.fill_between(t_anim, 0, energias_anim, alpha=0.15, color='magenta')
linea_tiempo_E, = ax_E.plot([], [], 'o', color='red', markersize=8, label='Posición actual')
ax_E.set_xlabel('Tiempo (s)', fontsize=10, fontweight='bold')
ax_E.set_ylabel('Energía (J/kg)', fontsize=10, fontweight='bold')
ax_E.set_title('Energía Total vs Tiempo', fontsize=11, fontweight='bold')
ax_E.grid(True, alpha=0.3, linestyle='--')
ax_E.legend(fontsize=9, loc='best')
ax_E.set_xlim(0, t_max_sim)

# ======================================================================
# Subplot 5: Información General (texto)
# ======================================================================

ax_info = fig.add_subplot(2, 3, 6)
ax_info.axis('off')

texto_info = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                         fontsize=10, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='#FFFFCC', 
                                 alpha=0.85, pad=1, linewidth=1.5),
                         linespacing=1.8)

print("✓ Figura 3D multi-subplot creada")
# ======================================================================
# 3. FUNCIONES DE ANIMACIÓN
# ======================================================================

def init():
    """Inicializar animación"""
    vagon_3d.set_data([], [])
    vagon_3d.set_3d_properties([])
    estela_3d.set_data([], [])
    estela_3d.set_3d_properties([])
    linea_tiempo_vel.set_data([], [])
    linea_tiempo_fn.set_data([], [])
    linea_tiempo_E.set_data([], [])
    texto_info.set_text('')
    return vagon_3d, estela_3d, linea_tiempo_vel, linea_tiempo_fn, linea_tiempo_E, texto_info

def animate(frame):
    """Actualizar animación en cada frame"""
    
    # Posición del vagón en 3D
    x_vagon = pos_anim[frame, 0]
    y_vagon = pos_anim[frame, 1]
    z_vagon = pos_anim[frame, 2]
    
    vagon_3d.set_data([x_vagon], [y_vagon])
    vagon_3d.set_3d_properties([z_vagon])
    
    # Estela (últimos 80 frames)
    inicio_estela = max(0, frame - 80)
    estela_3d.set_data(pos_anim[inicio_estela:frame+1, 0], 
                      pos_anim[inicio_estela:frame+1, 1])
    estela_3d.set_3d_properties(pos_anim[inicio_estela:frame+1, 2])
    
    # Marcadores en gráficas
    linea_tiempo_vel.set_data([t_anim[frame]], [v_anim[frame]])
    linea_tiempo_fn.set_data([t_anim[frame]], [fuerzas_G_anim[frame]])
    linea_tiempo_E.set_data([t_anim[frame]], [energias_anim[frame]])
    
    # Rotación suave de vista 3D
    if frame % 3 == 0:
        azim = 30 + (frame / n_frames) * 330
        ax_3d.view_init(elev=20, azim=azim)
    
    # Texto informativo
    porcentaje = (u_anim[frame] / u_anim[-1]) * 100 if u_anim[-1] > 0 else 0
    energia_perdida = energias_anim[0] - energias_anim[frame]
    potencia = energia_perdida / (t_anim[frame] + 1e-6)
    
    info_text = (
        f"⏱️  TIEMPO\n"
        f"   {t_anim[frame]:.2f} s\n\n"
        f"🚄 VELOCIDAD\n"
        f"   {v_anim[frame]:.2f} m/s\n\n"
        f"⚡ ENERGÍA\n"
        f"   {energias_anim[frame]:.2f} J/kg\n"
        f"   Pérdida: {energia_perdida:.2f} J/kg\n\n"
        f"⚖️  FUERZA NORMAL\n"
        f"   {fuerzas_G_anim[frame]:.2f} g\n"
        f"   Límite: 5.0 g\n\n"
        f"📍 POSICIÓN\n"
        f"   {porcentaje:.1f}% recorrido\n"
        f"   u = {u_anim[frame]:.3f}"
    )
    texto_info.set_text(info_text)
    
    return vagon_3d, estela_3d, linea_tiempo_vel, linea_tiempo_fn, linea_tiempo_E, texto_info

# Crear animación
anim = FuncAnimation(fig, animate, init_func=init, 
                    frames=n_frames, interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()

print("✓ Animación 3D creada correctamente")

print("\n=== FIN DEL PROGRAMA, pulsa enter para salir ===")
input()