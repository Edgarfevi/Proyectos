
import pandas as pd
from ics import Calendar, Event
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

import pandas as pd
from datetime import datetime

# Supongamos que tu DataFrame se llama df
# df = pd.read_csv("tu_csv.csv")  # si lo tienes en CSV

# Abrir archivo ICS para escribir
with open("calendario.ics", "w") as f:
    f.write("BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Mi Calendario//EN\n")

    for index, row in df.iterrows():
        f.write("BEGIN:VEVENT\n")
        f.write(f"UID:{index}\n")  # ID único por fila
        f.write(f"SUMMARY:{row['Subject']}\n")
        
        # Convertir Start Date + Start Time a formato 24h para ICS
        if row['Start Time']:  # si tiene hora
            start = datetime.strptime(row['Start Date'] + " " + row['Start Time'], "%d/%m/%Y %H:%M")
            end = datetime.strptime(row['End Date'] + " " + row['End Time'], "%d/%m/%Y %H:%M")
            f.write(f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}\n")
            f.write(f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}\n")
        else:  # Evento de todo el día
            start = datetime.strptime(row['Start Date'], "%d/%m/%Y")
            end = datetime.strptime(row['End Date'], "%d/%m/%Y")
            f.write(f"DTSTART;VALUE=DATE:{start.strftime('%Y%m%d')}\n")
            f.write(f"DTEND;VALUE=DATE:{end.strftime('%Y%m%d')}\n")
        
        f.write(f"LOCATION:{row['Location']}\n")
        f.write(f"DESCRIPTION:\n")  # puedes añadir row['Description'] si tienes
        f.write(f"STATUS:CONFIRMED\n")
        f.write("END:VEVENT\n")

    f.write("END:VCALENDAR")
