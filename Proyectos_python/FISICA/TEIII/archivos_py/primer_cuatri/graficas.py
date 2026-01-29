import matplotlib.pyplot as plt
import pandas as pd


# Abre el archivo Excel
df = pd.read_excel("/home/edgar/OneDrivePersonal/Universidad/excell.xlsx")

# Muestra las primeras filas
print(df.head())


angulos_45=[]
angulos_30=[]
paralela_30=[]
paralela_45=[]
perpendicular_30=[]
perpendicular_45=[]

for i in range(0,16):
    angulos_45.append(df.iloc[i+1,1])
    paralela_45.append(df.iloc[i+1,5])
    perpendicular_45.append(df.iloc[i+1,6])
    if df.iloc[i+16,1] != "NaN":
        angulos_30.append(df.iloc[i+16,1])
        paralela_30.append(df.iloc[i+16,5])
        perpendicular_30.append(df.iloc[i+16,6])


plt.plot(angulos_45,paralela_45,"o",label="horizontal")
plt.plot(angulos_45,perpendicular_45,"o",color="r",label="vertical")
plt.title(f"GRÁFICA INTENSIDAD FRENTE ÁNGULO")
plt.xlabel(r"Angulo $\alpha =45º$")
plt.ylabel(f"Intensidad (I)")
plt.legend()
plt.grid()
plt.savefig("/home/edgar/GitHub/Proyectos/Proyectos_python/TEIII/graficas/grafico_45.png")


plt.figure()
plt.plot(angulos_30, paralela_30, "o", label="horizontal")
plt.plot(angulos_30, perpendicular_30, "o", color="r", label="vertical")
plt.title(f"GRÁFICA INTENSIDAD FRENTE ÁNGULO")
plt.xlabel(r"Angulo $\alpha =30º$")
plt.ylabel(f"Intensidad (I)")
plt.legend()
plt.grid()
plt.savefig("/home/edgar/GitHub/Proyectos/Proyectos_python/TEIII/graficas/grafico_30.png")