# Importamos las bibliotecas necesarias
import os
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error


import warnings
warnings.filterwarnings('ignore')  # Para suprimir algunos avisos de statsmodels

#******************* 1. Introducción y lectura de datos
# Definimos nuestro entorno de trabajo.
os.chdir(r'C:\Users\MAICKEL\OneDrive\Documents\MASTER_UCM\MODULO 7 -P3- MINERIA DE DATOS (descomposición y suavizado)\Tarea') 
# df = pd.read_excel('datos_viajeros.xlsx')
df = pd.read_excel('datos_viajeros_bus.xlsx')
# Mostrar las primeras filas del dataframe
print(df.head())

# Renombrar columnas basándose en los nombres correctos extraídos del archivo
column_names = ["Fecha", "Total Nacional"] + list(df.iloc[6, 2:])
# Eliminar las primeras filas innecesarias
df_cleaned = df.iloc[7:].reset_index(drop=True)
# Asignar nombres de columna
df_cleaned.columns = column_names
# Filtrar filas que contienen fechas válidas en el formato esperado (YYYYMM)
df_cleaned = df_cleaned[df_cleaned["Fecha"].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)]

# Convertir la columna 'Fecha' en tipo datetime
df_cleaned["Fecha"] = pd.to_datetime(df_cleaned["Fecha"], format="%YM%m")
# Convertir las demás columnas a valores numéricos, reemplazando valores no numéricos con NaN
df_cleaned.iloc[:, 1:] = df_cleaned.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')


# Convertir "Fecha" en índice
df_cleaned.set_index("Fecha", inplace=True)
# Si los datos son mensuales, indicamos la frecuencia de comienzo de mes (MS)
df_cleaned = df_cleaned.asfreq('MS')  # Asegúrate que el periodo sea mensual



# ******************2. Representación gráfica y descomposición estacional
# Representación gráfica simple (ya lo hiciste, pero aquí lo refino con el índice temporal):
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned['Total Nacional'], marker='o', linestyle='-', color='b')
plt.title("Número total de pasajeros transportados por mes")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.grid(True)
plt.show()

# Descomposición estacional
# period=12 asume estacionalidad anual si tus datos son mensuales.
result = sm.tsa.seasonal_decompose(df_cleaned['Total Nacional'], 
                                   model='additive', 
                                   period=12)
result.plot()
plt.suptitle("Descomposición estacional de la serie", y=1.02)
plt.show()

# Mostrar los primeros 12 valores de estacionalidad
print("\nEstacionalidad:")
print(result.seasonal.head(12))



# ******************3. Separación en TRAIN y TEST
# Número de observaciones a reservar
n_test = 12
train = df_cleaned.iloc[:-n_test].copy()
test = df_cleaned.iloc[-n_test:].copy()
train['Total Nacional'] = pd.to_numeric(train['Total Nacional'], errors='coerce')
test['Total Nacional'] = pd.to_numeric(test['Total Nacional'], errors='coerce')
print(f"Tamaño del entrenamiento: {train.shape[0]} observaciones")
print(f"Tamaño del test: {test.shape[0]} observaciones")



# ********4. Ajustar un modelo de suavizado exponencial (Holt-Winters)
# Ajustamos el modelo en TRAIN
hw_model = ExponentialSmoothing(
    train['Total Nacional'],
    trend='add',            # 'add' para tendencia aditiva
    seasonal='add',         # 'add' si la estacionalidad es aditiva (pruébalo o revisa la gráfica)
    seasonal_periods=12     # frecuencia mensual
)
hw_fit = hw_model.fit()

# También puedes ver un resumen más detallado:
# (No todos los atributos están implementados como en ARIMA, pero verás info básica)
print("\n***** RESUMEN DEL MODELO *****")
print(hw_fit.summary())

# Predicciones en TEST
hw_pred = hw_fit.forecast(steps=len(test))

# Comparamos con los valores reales del TEST
comparison_hw = pd.DataFrame({
    'Real': test['Total Nacional'],
    'Predicción': hw_pred
})

print(comparison_hw)

# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, hw_pred, label='Predicción HW', marker='o')
plt.title("Suavizado Exponencial (Holt-Winters)")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()


# BUCLE PARA ENCONTRAR EL MEJOR MODELO DE SUAVIZADO
# Definimos las opciones de tendencia y estacionalidad
trend_options = ['add', 'mul', None]
seasonal_options = ['add', 'mul', None]
seasonal_periods = 12  # Asumiendo periodicidad mensual anual

results = []

# Ajustamos y comparamos modelos
for trend, season in product(trend_options, seasonal_options):
    # Evitamos el caso en el que no haya ni tendencia ni estacionalidad
    if trend is None and season is None:
        continue
    
    try:
        # Ajustamos modelo
        model = ExponentialSmoothing(
            train['Total Nacional'],
            trend=trend,
            seasonal=season,
            seasonal_periods=seasonal_periods
        )
        fitted_model = model.fit(optimized=True)
        
        # Predicciones en el horizonte 'test'
        forecast = fitted_model.forecast(steps=len(test))
        
        # Calculamos métricas de error
        mae = mean_absolute_error(test['Total Nacional'], forecast)
        rmse = mean_squared_error(test['Total Nacional'], forecast, squared=False)
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # Guardamos resultados
        results.append({
            'trend': trend,
            'seasonal': season,
            'AIC': aic,
            'BIC': bic,
            'MAE': mae,
            'RMSE': rmse,
            'model': fitted_model
        })
    
    except Exception as e:
        # Algunos ajustes pueden fallar (por ejemplo si hay ceros en los datos y usas 'mul')
        print(f"Error con trend={trend}, seasonal={season}: {e}")

# Ordenamos la tabla de resultados para ver los mejores
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='AIC', ascending=True)  # o por 'RMSE', etc.

print("Resultados ordenados por AIC (menor es mejor):")
print(results_df_sorted[['trend','seasonal','AIC','BIC','MAE','RMSE']].head())

# Mejor modelo según la métrica elegida (por ejemplo, AIC)
best_row = results_df_sorted.iloc[0]  # la primera fila es la de menor AIC
best_model = best_row['model']
print(f"\nMejor modelo: Trend={best_row['trend']}, Seasonal={best_row['seasonal']}")
print(f"AIC={best_row['AIC']}, RMSE={best_row['RMSE']}")
print("\nResumen del mejor modelo:\n", best_model.summary())

# Si lo deseas, generar la predicción final y graficarla
best_forecast = best_model.forecast(steps=len(test))
# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, best_forecast, label='Predicción HW', marker='o')
plt.title("Suavizado Exponencial (Holt-Winters) Sin tendencia")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()




# *************** 5. Análisis de correlogramas y posible modelo ARIMA manual
# Visualizamos correlogramas en TRAIN
fig, ax = plt.subplots(1,2, figsize=(14,5))
plot_acf(train['Total Nacional'].dropna(), lags=40, ax=ax[0])
plot_pacf(train['Total Nacional'].dropna(), lags=40, ax=ax[1])
plt.show()

# Ejemplo hipotético: SARIMA(1,1,1)(1,1,1,12)
mod_arima = sm.tsa.statespace.SARIMAX(train['Total Nacional'],
                                      order=(1,1,1),              # p,d,q
                                      seasonal_order=(1,1,1,12),  # P,D,Q,m
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
res_arima = mod_arima.fit()

print(res_arima.summary())

# Validación de la incorrelación de los residuos
residuals = res_arima.resid
fig, ax = plt.subplots(1,2, figsize=(14,5))
plot_acf(residuals.dropna(), lags=40, ax=ax[0])
plot_pacf(residuals.dropna(), lags=40, ax=ax[1])
plt.suptitle("Correlogramas de los residuos del modelo ARIMA")
plt.show()

mod_arima_forecast = res_arima.forecast(steps=len(test))
# Comparo con los valores reales del TEST
comparison_sarima = pd.DataFrame({
    'Real': test['Total Nacional'],
    'Predicción': mod_arima_forecast
})
comparison_sarima

# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, mod_arima_forecast, label='Predicción HW', marker='o')
plt.title("Suavizado Exponencial (Holt-Winters)")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()




# ***********   6. Ajustar un modelo ARIMA con ajuste automático
auto_model = pm.auto_arima(
    train['Total Nacional'],
    start_p=0, max_p=3,    # rangos para p
    start_q=0, max_q=3,    # rangos para q
    d=None,                # dejar que lo estime
    seasonal=True, m=12,   # asumes estacionalidad mensual
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=None,
    trace=True,            # para ir viendo qué hace
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(auto_model.summary())

# para comparar con tu conjunto de prueba.
autoarima_forecast = auto_model.predict(n_periods=len(test))

# Comparo con los valores reales del TEST
comparison_auto_arima = pd.DataFrame({
    'Real': test['Total Nacional'],
    'Predicción': autoarima_forecast
})
comparison_auto_arima

# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, autoarima_forecast, label='Predicción HW', marker='o')
plt.title("Suavizado Exponencial (Holt-Winters)")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()


 
# ********  7. Expresión algebraica del modelo ajustado
# extraigo los parámetros estimados del summary()
best_model.summary()

# Formulas en formato celda jupyter
# # Ecuaciones básicas del modelo Holt-Winters multiplicativo (sin tendencia)
# Para cada periodo $(t)$, con estacionalidad de periodo $(m = 12)$:

# 1. **Actualización del nivel $(\ell_t)$**  
# $$
# \ell_t 
# \;=\; 
# 0.9478571 \times \frac{y_t}{s_{t-m}}
# \;+\;
# \bigl(1 - 0.9478571\bigr)\,\ell_{t-1}.
# $$

# 2. **Actualización de la estacionalidad $(s_t)$**  
# $$
# s_t 
# \;=\; 
# 0.0521429 \times \frac{y_t}{\ell_t}
# \;+\;
# \bigl(1 - 0.0521429\bigr)\,s_{t-m}.
# $$

# 3. **Pronóstico a $h$ pasos vista**  
# $$
# \widehat{y}_{t+h}
# \;=\;
# \ell_t
# \times
# s_{\,t - m + (h \bmod m)}.
# $$

# > Dado que `trend=None`, **no hay ecuación de tendencia** (i.e., $\beta=0$).


# *************  8. Predicciones e intervalos de confianza fuera de la muestra
# Predicción para el período TEST
# Índice temporal para los 12 meses futuros
last_date = df_cleaned.index[-1]  # última fecha observada
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                             periods=12, freq='MS')

# Estimación de la desviación típica de los residuos (método aproximado)
residuals = best_model.fittedvalues - train['Total Nacional']
sigma = residuals.std()
z = 1.96  # ~ 95% de confianza

# Intervalos de confianza (IC) aproximados
lower_bound = best_forecast - z * sigma
upper_bound = best_forecast + z * sigma

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned.index, df_cleaned['Total Nacional'], label='Serie Observada', color='blue')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(future_dates, best_forecast, label='Predicción (12 meses)', marker='o', color='green')
plt.fill_between(future_dates, lower_bound, upper_bound,
                 color='gray', alpha=0.3, label='IC 95% (aprox)')
plt.title("Holt-Winters: Predicción y Intervalos de Confianza (aprox)")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()


# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, best_forecast, label='Predicción', marker='o')
plt.fill_between(test.index, lower_bound, upper_bound,
                 color='gray', alpha=0.3, label='IC 95% (aprox)')
plt.title("Holt-Winters: Predicción y Intervalos de Confianza (aprox)")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()




# ***********  9. Comparación de predicciones
# ***********  9. Comparación de predicciones
# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, best_forecast, label='Mejor predicción suavizado', marker='o')
plt.plot(test.index, mod_arima_forecast, label='Predicción Arima Manual', marker='o')
plt.title("Comparativa mejor modelos vs modelo arima manual")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()

# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, best_forecast, label='Mejor predicción suavizado', marker='o')
plt.plot(test.index, hw_pred, label='Predicción HW', marker='o')
plt.title("Comparativa mejor modelo vs prediccion HW")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()


# Representación gráfica
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Total Nacional'], label='Entrenamiento')
plt.plot(test.index, test['Total Nacional'], label='Test Real', marker='o')
plt.plot(test.index, best_forecast, label='Mejor predicción suavizado', marker='o')
plt.plot(test.index, autoarima_forecast, label='Predicción AutoArima', marker='o')
plt.title("Comparativa mejor modelo vs prediccion autoArima")
plt.xlabel("Fecha")
plt.ylabel("Número de pasajeros")
plt.legend()
plt.grid(True)
plt.show()








