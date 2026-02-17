import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace import sa
from sklearn.metrics import mean_squared_error

# Cargar CSV
df = pd.read_csv(r"C:\Users\Usuario\Downloads\datathon25_parte2_1z2x3c\datathon25_parte2\datos_turismos_predictivo.csv", sep=";")

# Asegurarse de que los datos sean numéricos
df["EJERCICIO"] = df["EJERCICIO"].astype(int)
df["MES"] = df["MES"].astype(int)
df["TURISMOS"] = df["TURISMOS"].astype(int)

# Crear columna MES_CONTINUO
df["MES_CONTINUO"] = range(len(df))

df.set_index('MES_CONTINUO', inplace=True)

# Si los datos tienen una alta variabilidad, transformamos logaritmicamente
df['TURISMOS'] = np.log1p(df['TURISMOS'])  # Transformación logarítmica

# === Entrenamiento del modelo SARIMA ===
# Probamos diferentes combinaciones de parámetros (p, d, q) y (P, D, Q, S)

# Para un grid search más amplio
best_aic = np.inf
best_order = None
best_seasonal_order = None
best_model = None

# Probamos diferentes combinaciones para los parámetros
for p in range(5):  # Aumentamos el rango de p
    for d in range(5):  # Rango para la diferencia
        for q in range(5):  # Aumentamos el rango de q
            for P in range(5):  # Aumentamos el rango de P
                for D in range(5):  # Rango para la diferencia estacional
                    for Q in range(5):  # Aumentamos el rango de Q
                        try:
                            # Definimos el modelo SARIMA con los parámetros actuales
                            model = SARIMAX(df['TURISMOS'],
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, 12),  # S=12 asumiendo estacionalidad anual
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                            
                            # Ajustamos el modelo
                            results = model.fit(disp=False)
                            
                            # Comparamos AIC para seleccionar el mejor modelo
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                best_seasonal_order = (P, D, Q, 12)
                                best_model = results
                        except:
                            continue

# Mostrar el mejor modelo encontrado
print(f"Mejor modelo encontrado: SARIMA{best_order}x{best_seasonal_order} - AIC: {best_aic}")

# Predicción para los próximos 6 meses
forecast = best_model.get_forecast(steps=6)
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Revertir la transformación logarítmica
forecast_values_exp = np.expm1(forecast_values)

# === Visualización de los resultados ===
plt.figure(figsize=(12, 6))

# Gráfico de los datos reales
plt.plot(df.index, np.expm1(df['TURISMOS']), label='Datos Reales', color='blue')

# Gráfico de la predicción
forecast_index = np.arange(df.index[-1] + 1, df.index[-1] + 7)
plt.plot(forecast_index, forecast_values_exp, label='Predicción SARIMA', linestyle='--', color='red')

# Añadir los intervalos de confianza
plt.fill_between(forecast_index, np.expm1(confidence_intervals.iloc[:, 0]), np.expm1(confidence_intervals.iloc[:, 1]), color='pink', alpha=0.3)

# Etiquetas y título
plt.xlabel('Mes Continuo')
plt.ylabel('Turismos')
plt.title('Predicción de Turismos usando SARIMA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%%


sum_forecast_values = forecast_values_exp.sum()

# Imprimir la suma de los valores predichos
print(f"Suma de los valores predichos: {sum_forecast_values:.2f}")
#%%
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# === Preparar datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Entrenar modelo SVR ===
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_scaled, y_scaled)

# === Predecir últimos 6 meses de 2022 ===
future_months = np.arange(df["MES_CONTINUO"].max() + 1, df["MES_CONTINUO"].max() + 7).reshape(-1, 1)
future_months_scaled = scaler_X.transform(future_months)
future_preds_scaled = model.predict(future_months_scaled)
future_preds = scaler_y.inverse_transform(future_preds_scaled.reshape(-1, 1)).ravel()

# === Mostrar resultados ===
meses = ["Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
print("Predicción de turismos (julio a diciembre 2022):")
for i, pred in enumerate(future_preds):
    print(f"{meses[i]} 2022: {pred:.0f} turismos")

# === Calcular MSE sobre datos de entrenamiento ===
y_train_pred_scaled = model.predict(X_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
mse = mean_squared_error(y, y_train_pred)
print(f"\nError cuadrático medio (MSE) en entrenamiento: {mse:.2f}")

# === Visualización ===
plt.figure(figsize=(12, 5))
plt.plot(df["MES_CONTINUO"], df["TURISMOS"], label="Datos reales")
plt.plot(future_months, future_preds, 'ro--', label="Predicción julio-diciembre 2022")
plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Predicción de turismos con SVR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# === Preparar datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Optimización de hiperparámetros ===
param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['rbf', 'poly']  # Probar con el kernel polinómico también
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y_scaled)

# Mejores parámetros encontrados
best_params = grid_search.best_params_
print(f"Mejores parámetros encontrados: {best_params}")

# === Entrenar el modelo con los mejores parámetros ===
best_model = grid_search.best_estimator_
best_model.fit(X_scaled, y_scaled)

# === Predecir últimos 6 meses de 2022 ===
future_months = np.arange(df["MES_CONTINUO"].max() + 1, df["MES_CONTINUO"].max() + 7).reshape(-1, 1)
future_months_scaled = scaler_X.transform(future_months)
future_preds_scaled = best_model.predict(future_months_scaled)
future_preds = scaler_y.inverse_transform(future_preds_scaled.reshape(-1, 1)).ravel()

# === Mostrar resultados ===
meses = ["Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
print("Predicción de turismos (julio a diciembre 2022):")
for i, pred in enumerate(future_preds):
    print(f"{meses[i]} 2022: {pred:.0f} turismos")

# === Calcular MSE sobre datos de entrenamiento ===
y_train_pred_scaled = best_model.predict(X_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
mse = mean_squared_error(y, y_train_pred)
print(f"\nError cuadrático medio (MSE) en entrenamiento: {mse:.2f}")

# === Evaluación con validación cruzada ===
cv_scores = cross_val_score(best_model, X_scaled, y_scaled, cv=5, scoring='neg_mean_squared_error')
print(f"\nError cuadrático medio en validación cruzada: {-cv_scores.mean():.2f}")

# === Visualización ===
plt.figure(figsize=(12, 5))
plt.plot(df["MES_CONTINUO"], df["TURISMOS"], label="Datos reales")
plt.plot(future_months, future_preds, 'ro--', label="Predicción julio-diciembre 2022")
plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Predicción de turismos con SVR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# === Preparar datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Identificación de outliers usando Rango Intercuartil (IQR)
q1 = np.percentile(df['TURISMOS'], 25)
q3 = np.percentile(df['TURISMOS'], 75)
iqr = q3 - q1  # Rango intercuartil

# Límites para outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Eliminar outliers
df_cleaned = df[(df['TURISMOS'] >= lower_bound) & (df['TURISMOS'] <= upper_bound)]

# Nuevos datos después de eliminar los outliers
X_cleaned = df_cleaned[["MES_CONTINUO"]].values
y_cleaned = df_cleaned["TURISMOS"].values

# Escalado robusto
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X_cleaned)
y_scaled = scaler_y.fit_transform(y_cleaned.reshape(-1, 1)).ravel()

# === Optimización de hiperparámetros ===
param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['rbf', 'poly']  # Probar con el kernel polinómico también
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y_scaled)

# Mejores parámetros encontrados
best_params = grid_search.best_params_
print(f"Mejores parámetros encontrados: {best_params}")

# === Entrenar el modelo con los mejores parámetros ===
best_model = grid_search.best_estimator_
best_model.fit(X_scaled, y_scaled)

# === Predecir últimos 6 meses de 2022 ===
future_months = np.arange(df["MES_CONTINUO"].max() + 1, df["MES_CONTINUO"].max() + 7).reshape(-1, 1)
future_months_scaled = scaler_X.transform(future_months)
future_preds_scaled = best_model.predict(future_months_scaled)
future_preds = scaler_y.inverse_transform(future_preds_scaled.reshape(-1, 1)).ravel()

# === Mostrar resultados ===
meses = ["Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
print("Predicción de turismos (julio a diciembre 2022):")
for i, pred in enumerate(future_preds):
    print(f"{meses[i]} 2022: {pred:.0f} turismos")

# === Calcular MSE sobre datos de entrenamiento ===
y_train_pred_scaled = best_model.predict(X_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
mse = mean_squared_error(y_cleaned, y_train_pred)
print(f"\nError cuadrático medio (MSE) en entrenamiento: {mse:.2f}")

# === Evaluación con validación cruzada ===
cv_scores = cross_val_score(best_model, X_scaled, y_scaled, cv=5, scoring='neg_mean_squared_error')
print(f"\nError cuadrático medio en validación cruzada: {-cv_scores.mean():.2f}")

# === Visualización ===
plt.figure(figsize=(12, 5))
plt.plot(df_cleaned["MES_CONTINUO"], df_cleaned["TURISMOS"], label="Datos reales")
plt.plot(future_months, future_preds, 'ro--', label="Predicción julio-diciembre 2022")
plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Predicción de turismos con SVR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

# === Preparar datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Normalizar los datos (escala entre 0 y 1)
from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Inicialización de parámetros ===
epochs = 50
learning_rate = 0.01  # Tasa de aprendizaje
m = len(X_scaled)  # Número de datos de entrenamiento

# Inicialización aleatoria de los pesos y el sesgo
w = np.random.randn()
b = np.random.randn()

# === Función de costo (MSE) ===
def compute_cost(X, y, w, b):
    predictions = X * w + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)  # MSE
    return cost

# === Gradiente Descendente ===
costs = []

for epoch in range(epochs):
    # Predicciones del modelo
    predictions = X_scaled * w + b

    # Gradientes de los parámetros (derivadas parciales)
    dw = (1/m) * np.sum((predictions - y_scaled) * X_scaled)
    db = (1/m) * np.sum(predictions - y_scaled)

    # Actualización de los parámetros
    w -= learning_rate * dw
    b -= learning_rate * db

    # Calcular y almacenar el costo
    cost = compute_cost(X_scaled, y_scaled, w, b)
    costs.append(cost)

    if epoch % 10 == 0:  # Mostrar cada 10 épocas
        print(f"Época {epoch + 1}/{epochs}, Costo: {cost:.4f}")

# === Predicción ===
y_pred_scaled = X_scaled * w + b
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# === Evaluación del modelo ===
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print(f"\nError cuadrático medio (MSE) final: {mse:.2f}")

# === Visualización del costo durante el entrenamiento ===
plt.plot(range(epochs), costs, label="Costo durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Costo (MSE)")
plt.title("Convergencia del gradiente descendente")
plt.grid(True)
plt.legend()
plt.show()

# === Visualización de la regresión ===
plt.scatter(X, y, label="Datos reales", color='blue')
plt.plot(X, y_pred, label="Modelo de regresión", color='red')
plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Modelo de regresión lineal con Gradiente Descendente")
plt.legend()
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# === Preparar datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Normalizar los datos (escala entre 0 y 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Crear modelo de red neuronal más profunda ===
model = Sequential()

# Capa de entrada (con 128 unidades y función de activación ReLU)
model.add(Dense(128, input_dim=1, activation='relu'))

# Capas ocultas (con 64 y 32 unidades, con activación ReLU)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Capa de salida (con 1 unidad y sin función de activación)
model.add(Dense(1))

# === Compilar el modelo ===
model.compile(optimizer='adam', loss='mean_squared_error')

# === Entrenar el modelo ===
history = model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, verbose=1)

# === Evaluación del modelo ===
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Error cuadrático medio (MSE) final: {mse:.2f}")

# === Visualización de la función de pérdida durante el entrenamiento ===
plt.plot(history.history['loss'], label="Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida (MSE)")
plt.title("Convergencia de la red neuronal")
plt.grid(True)
plt.legend()
plt.show()

# === Visualización de la regresión no lineal ===
plt.figure(figsize=(12, 6))

# Graficar datos reales
plt.scatter(X, y, label="Datos reales", color='blue', alpha=0.6)

# Graficar predicciones
plt.plot(X, y_pred, label="Predicción de la red neuronal", color='red', linewidth=2)

plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Predicción no lineal de turismos con red neuronal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# === Preparar datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Entrenar modelo SVR con ajustes para mayor no linealidad ===
model = SVR(kernel='poly', C=1000, gamma=0.1, epsilon=0.01)  # Aumento de C y gamma
model.fit(X_scaled, y_scaled)

# === Predecir últimos 6 meses de 2022 ===
future_months = np.arange(df["MES_CONTINUO"].max() + 1, df["MES_CONTINUO"].max() + 7).reshape(-1, 1)
future_months_scaled = scaler_X.transform(future_months)
future_preds_scaled = model.predict(future_months_scaled)
future_preds = scaler_y.inverse_transform(future_preds_scaled.reshape(-1, 1)).ravel()

# === Mostrar resultados ===
meses = ["Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
print("Predicción de turismos (julio a diciembre 2022):")
for i, pred in enumerate(future_preds):
    print(f"{meses[i]} 2022: {pred:.0f} turismos")

# === Calcular MSE sobre datos de entrenamiento ===
y_train_pred_scaled = model.predict(X_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
mse = mean_squared_error(y, y_train_pred)
print(f"\nError cuadrático medio (MSE) en entrenamiento: {mse:.2f}")

# === Visualización ===
plt.figure(figsize=(12, 5))
plt.plot(df["MES_CONTINUO"], df["TURISMOS"], label="Datos reales")
plt.plot(future_months, future_preds, 'ro--', label="Predicción julio-diciembre 2022")
plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Predicción de turismos con SVR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# === Cargar datos ===
# Asumiendo que ya tienes el DataFrame df con 'MES_CONTINUO' y 'TURISMOS'
# Si no es así, asegúrate de cargar o definir tu DataFrame `df`

# === Preparar los datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Normalizar los datos (escala entre 0 y 1)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# === Reestructurar los datos para LSTM ===
# Los datos deben ser de la forma [muestras, pasos de tiempo, características]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# === Crear el modelo LSTM ===
model = Sequential()

# Capa LSTM con 50 unidades y función de activación 'relu'
model.add(LSTM(units=50, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2])))

# Capa de salida con 1 neurona
model.add(Dense(1))

# === Compilar el modelo ===
model.compile(optimizer='adam', loss='mean_squared_error')

# === Entrenar el modelo ===
history = model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=1)

# === Evaluación del modelo ===
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Error cuadrático medio (MSE) final: {mse:.2f}")

# === Visualización de la regresión no lineal ===
plt.figure(figsize=(12, 6))

# Graficar datos reales
plt.scatter(X, y, label="Datos reales", color='blue', alpha=0.6)

# Graficar predicciones
plt.plot(X, y_pred, label="Predicción LSTM", color='red', linewidth=2)

plt.xlabel("Mes continuo desde enero 2015")
plt.ylabel("Turismos")
plt.title("Predicción no lineal con LSTM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# === Cargar datos ===
# Asumimos que tienes el DataFrame 'df' con 'MES_CONTINUO' y 'TURISMOS'
# Si no es así, asegúrate de cargar o definir tu DataFrame

# === Preparar los datos ===
X = df[["MES_CONTINUO"]].values
y = df["TURISMOS"].values

# Normalización de los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# === Crear y entrenar el modelo ===
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Predicción continua
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# === Entrenar el modelo 50 veces con variación en los datos ===
epochs = 50
predicciones = []
errores = []

for i in range(epochs):
    # En cada ciclo, dividimos los datos, omitiendo los primeros 6 meses de cada ciclo
    X_train = np.delete(X_scaled, slice(0, 6), axis=0)
    y_train = np.delete(y_scaled, slice(0, 6), axis=0)
    X_test = X_scaled[:6]  # Los primeros 6 meses para prueba
    y_test = y_scaled[:6]

    # Crear y entrenar el modelo
    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=3, verbose=0)

    # Hacer predicción para los primeros 6 meses
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Guardar las predicciones y calcular el error
    predicciones.append(y_pred.ravel())
    error = mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred)
    errores.append(error)

# === Mostrar las predicciones y errores ===
predicciones = np.array(predicciones)
errores = np.array(errores)

# Mostrar los errores promedio
print(f"Error cuadrático medio promedio (MSE) en {epochs} ciclos: {errores.mean():.2f}")

# Graficar las predicciones
plt.figure(figsize=(12, 6))
plt.plot(np.arange(6), df["TURISMOS"][:6], label="Datos reales", color='blue', marker='o')

for i in range(epochs):
    plt.plot(np.arange(6), predicciones[i], label=f"Predicción ciclo {i+1}", linestyle='--')

plt.xlabel("Meses de 2022 (Enero a Junio)")
plt.ylabel("Turismos")
plt.title(f"Predicciones con red neuronal en {epochs} ciclos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['MES_CONTINUO'], df['TURISMOS'], label="Turismos")
plt.xlabel("Mes Continuo")
plt.ylabel("Número de Turismos")
plt.title("Comportamiento de los Turismos a lo largo del tiempo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# === Cargar los datos ===
# Asumimos que 'df' contiene los datos con las columnas 'MES_CONTINUO' y 'TURISMOS'
# Aquí asignamos la columna 'MES_CONTINUO' como el índice temporal



# Asegurémonos de que 'MES_CONTINUO' esté en formato numérico
df['MES_CONTINUO'] = pd.to_numeric(df['MES_CONTINUO'], errors='coerce')

# Establecemos 'MES_CONTINUO' como índice temporal
df.set_index('MES_CONTINUO', inplace=True)

# === Entrenamiento del modelo SARIMA ===
# Usamos un modelo SARIMA con parámetros ajustados manualmente
# (p, d, q) = (1, 1, 1) para el componente ARIMA
# (P, D, Q, S) = (1, 1, 1, 12) para la estacionalidad anual

# Creamos el modelo SARIMA
sarima_model = SARIMAX(df['TURISMOS'], 
                       order=(1, 1, 1),  # ARIMA
                       seasonal_order=(1, 1, 1, 12),  # Estacionalidad anual
                       enforce_stationarity=False,  # Permite que el modelo sea estacionario
                       enforce_invertibility=False)  # Permite que el modelo sea invertible

# Ajustamos el modelo
sarima_results = sarima_model.fit()

# === Predicción para los próximos 6 meses ===
forecast = sarima_results.get_forecast(steps=6)
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# === Visualización de los resultados ===
plt.figure(figsize=(12, 6))

# Gráfico de los datos reales
plt.plot(df.index, df['TURISMOS'], label='Datos Reales', color='blue')

# Gráfico de la predicción
forecast_index = np.arange(df.index[-1] + 1, df.index[-1] + 7)
plt.plot(forecast_index, forecast_values, label='Predicción SARIMA', linestyle='--', color='red')

# Añadir los intervalos de confianza
plt.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)

# Etiquetas y título
plt.xlabel('Mes Continuo')
plt.ylabel('Turismos')
plt.title('Predicción de Turismos usando SARIMA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Evaluación del modelo ===
# Si tienes los valores reales de los próximos 6 meses, calcula el MSE (si no, este paso se omite)
# Asumimos que los últimos 6 meses corresponden a los valores reales a predecir
# En este caso, debes tener una columna en el DataFrame con los datos reales de los próximos 6 meses
real_values = df['TURISMOS'].iloc[-6:]  # Los últimos 6 meses como prueba (ajusta según tu caso)
mse = mean_squared_error(real_values, forecast_values)
print(f"Error cuadrático medio (MSE): {mse:.2f}")