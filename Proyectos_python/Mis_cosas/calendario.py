
import pandas as pd

# Leer el archivo Excel
df = pd.read_excel('/home/edgar/Documentos/GitHub/Proyectos/Proyectos_python/Mis_cosas/nombre.xls')  # o .xlsx

# Ver las primeras filas completas

print(df)
dia={}
dia_ = ""
subject = ""
calendario = {}
n=0
Start_date = ""
End_date = ""
Start_Time = "" 
End_Time = ""
Location = "" 
n = 0
for i in range (3,len(df)):
    for j in range (1, len(df.columns)):
        if (i-3) % 14 == 0 and type(df.iloc[i,j]) == str:
            dia_ = df.iloc[i,j].split(" ")[1]
            if "" != dia_:
                dia[j] = dia_
    if (i-3) % 14 != 0:            
        for k in range (1,15,3):
            if type(df.iloc[i,k]) == str and df.iloc[i,k] != "":
                Start_Time = df.iloc[i,k].split("-")[0]
                End_Time = df.iloc[i,k].split("-")[1]
            if type(df.iloc[i,k+1]) == str and df.iloc[i,k+1] != "":
                subject = df.iloc[i,k+1]
            if type(df.iloc[i,k+2]) == str and df.iloc[i,k+2] != "":
                Location = df.iloc[i,k+2]
            calendario[n]={'Subject': f'{subject}','Start Date': f'{dia[k]}','Start Time': f'{Start_Time}','End Date': f'{dia[k]}','End Time': f'{End_Time}','All Day Event': False,'Location': f'{Location}','Private': False}
            n+=1




df = pd.DataFrame.from_dict(calendario, orient='index')
print(df)

df.to_csv('calendario.csv', index=False)
