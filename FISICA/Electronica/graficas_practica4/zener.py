import matplotlib.pyplot as plt
import numpy as np

# Corriente de carga (IL) en amperios
IL = np.linspace(0, 0.03, 200)  # 0 a 30 mA

# Voltaje aproximado del Zener (VZ)
VZ = np.piecewise(IL, [IL < 0.005, (IL >= 0.005) & (IL <= 0.02), IL > 0.02],
                [lambda IL: 5 - 50*IL, 5, lambda IL: 5 - 50*(IL-0.02)])

plt.figure(figsize=(6,4))
plt.plot(IL*1000, VZ, color='blue', label='VZ vs IL')
plt.axhline(5, color='gray', linestyle='--', label='V_Z nominal')
plt.xlabel('Corriente de carga I_L [mA]')
plt.ylabel('Voltaje Zener V_Z [V]')
plt.title('Aproximación Voltaje vs Corriente del Zener')
plt.grid(True)
plt.legend()
plt.savefig("caracteristica_zener.png", dpi=300)