# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:01:51 2025

@author: MAICKEL
"""

# Cargo las librerias 
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
os.chdir(r'C:\Users\MAICKEL\OneDrive\Desktop\Entorno')
from FuncionesMineria import (graficoVcramer, impVariablesLog, pseudoR2, summary_glm, 
                           validacion_cruzada_glm, sensEspCorte, crear_data_modelo, curva_roc, 
                           glm_stepwise)


with open('datos_regresion.pickle', 'rb') as f:
    datos = pickle.load(f)
varObjCont = datos['AbstentionPtge']  # Variable continua
varObjBin = datos['AbstencionAlta']  # Variable binaria
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1) 


# Veo el reparto original. Compruebo que la variable objetivo tome valor 1 para el evento y 0 para el no evento
pd.DataFrame({
    'n': varObjBin.value_counts()
    , '%': varObjBin.value_counts(normalize = True)
})


# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos, varObjBin, test_size = 0.2, random_state = 1234567)
y_train, y_test = y_train.astype(int), y_test.astype(int)


var_cont1 = ['Population', 'Age_under19_Ptge', 'Age_19_65_pct', 'Age_over65_pct',
       'WomanPopulationPtge', 'ForeignersPtge', 'SameComAutonPtge',
       'SameComAutonDiffProvPtge', 'DifComAutonPtge', 'UnemployLess25_Ptge',
       'Unemploy25_40_Ptge', 'UnemployMore40_Ptge',
       'AgricultureUnemploymentPtge', 'IndustryUnemploymentPtge',
       'ConstructionUnemploymentPtge', 'ServicesUnemploymentPtge',
       'totalEmpresas', 'Industria', 'Construccion', 'ComercTTEHosteleria',
       'Servicios', 'Pob2010', 'SUPERFICIE', 'PobChange_pct',
       'PersonasInmueble', 'Explotaciones']
var_categ1 = ['CodigoProvincia', 'CCAA', 'Densidad', 'prop_missings']

# Crear los datos procesados para el modelo
x_train_procesado = crear_data_modelo(x_train, var_cont1, var_categ1)
x_test_procesado = crear_data_modelo(x_test, var_cont1, var_categ1)
# Alinear las columnas de x_test_procesado2 con x_train_procesado2
x_test_procesado = x_test_procesado.reindex(columns=x_train_procesado.columns, fill_value=0)

# Modelo de regresión logística usando selección por pasos (stepwise)
modelo_logistico = glm_stepwise(
    y_train, x_train, var_cont1, var_categ1, metodo='BIC')
# Usamos los datos procesados directamente del modelo ajustado
x_train_procesado = modelo_logistico['X']
resumen_modelo_logistico = summary_glm(modelo_logistico['Modelo'], y_train, x_train_procesado)

pseudo_r2 = pseudoR2(modelo_logistico['Modelo'], x_train_procesado, y_train)
print("Pseudo R^2:", pseudo_r2)

x_test_procesado = crear_data_modelo(x_test, var_cont1, var_categ1)
x_test_procesado = x_test_procesado.reindex(columns=x_train_procesado.columns, fill_value=0)
AUC = curva_roc(x_test_procesado, y_test, modelo_logistico)

puntos_corte = [0.3, 0.5, 0.7]
for ptoCorte in puntos_corte:
    resultados_corte = sensEspCorte(
        modelo_logistico['Modelo'],
        x_test_procesado,
        y_test,
        ptoCorte,
        x_train_procesado.columns.tolist(),  # Aseguramos que las columnas coincidan
        []  # Pasamos una lista vacía porque las variables categóricas ya están procesadas
    )
    print(f"Resultados para punto de corte {ptoCorte}:")
    print(resultados_corte)
# Evaluar la importancia de las variables
impVariablesLog(modelo_logistico,y_train,x_train,var_cont1,var_categ1)
graficoVcramer(datos_input, varObjBin)




var_cont2 = ['Population', 'Age_under19_Ptge', 'Age_over65_pct', 'totalEmpresas', 
             'Industria', 'Construccion', 'ComercTTEHosteleria','Servicios', 
             'Pob2010', 'PersonasInmueble']
var_categ2 = ['CodigoProvincia', 'CCAA', 'Densidad']

# Construcción del modelo logístico con selección por pasos (criterio BIC)
modelo_logistico2 = glm_stepwise(
    y_train, x_train, var_cont2, var_categ2, metodo='BIC'
)

# Usamos los datos procesados directamente del modelo ajustado
x_train_procesado2 = modelo_logistico2['X']

# Obtener el resumen del modelo utilizando summary_glm
resumen_modelo2 = summary_glm(modelo_logistico2['Modelo'], y_train, x_train_procesado2)
print(resumen_modelo2)

# Calcular el pseudo R^2
pseudo_r2_2 = pseudoR2(modelo_logistico2['Modelo'], x_train_procesado2, y_train)
print("Pseudo R^2:", pseudo_r2_2)

# Preparamos los datos de prueba en el mismo formato que los datos de entrenamiento


# Validación cruzada
resultados_validacion2 = validacion_cruzada_glm(
    5, x_train, y_train, modelo_logistico2['Variables']['cont'], modelo_logistico2['Variables']['categ']
)

x_test_procesado2 = crear_data_modelo(x_test, var_cont2, var_categ2)
x_test_procesado2 = x_test_procesado2.reindex(columns=x_train_procesado2.columns, fill_value=0)
AUC2 = curva_roc(x_test_procesado2, y_test, modelo_logistico2)


# Obtener el punto de corte óptimo
# Determinamos los puntos de corte para explorar
puntos_corte = [0.3, 0.5, 0.7]
for ptoCorte in puntos_corte:
    resultados_corte2 = sensEspCorte(
        modelo_logistico2['Modelo'],
        x_test_procesado2,
        y_test,
        ptoCorte,
        x_train_procesado2.columns.tolist(),  # Aseguramos que las columnas coincidan
        []  # Pasamos una lista vacía porque las variables categóricas ya están procesadas
    )
    print(f"Resultados para punto de corte {ptoCorte}:")
    print(resultados_corte2)



# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Submuestra aleatoria del 70% de los datos
    x_train_sub, _, y_train_sub, _ = train_test_split(x_train, y_train, test_size=0.3, random_state=1234567 + x)

    # Selección stepwise con criterio BIC
    modelo = glm_stepwise(y_train_sub.astype(int), x_train_sub, var_cont2, var_categ2, [], 'BIC')

    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['X'].columns))
    # variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]


# Validación cruzada final
results = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []})

for rep in range(20):
    modelo1 = validacion_cruzada_glm(5, x_train, y_train, modelo_logistico['Variables']['cont'], modelo_logistico['Variables']['categ'])
    modelo2 = validacion_cruzada_glm(5, x_train, y_train, modelo_logistico2['Variables']['cont'], modelo_logistico2['Variables']['categ'])
    modelo3 = validacion_cruzada_glm(5, x_train, y_train, var_1['cont'], var_1['categ'])
    modelo4 = validacion_cruzada_glm(5, x_train, y_train, var_2['cont'], var_2['categ'])

    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 + modelo4,
        'Resample': ['Rep' + str((rep + 1))] * 5 * 4,
        'Modelo': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
    })
    results = pd.concat([results, results_rep], axis=0)

# Boxplot de validación cruzada
plt.figure(figsize=(10, 6))
plt.grid(True)
grupo_metrica = results.groupby('Modelo')['Rsquared']
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())
plt.xlabel('Modelo')
plt.ylabel('Rsquared')
plt.show()

todos_resultados_corte = []
# Determinar los puntos de corte a explorar
puntos_corte = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for ptoCorte in puntos_corte:
    resultados_corte = sensEspCorte(
        modelo_logistico['Modelo'],
        x_test_procesado,
        y_test,
        ptoCorte,
        x_train_procesado.columns.tolist(),  # Aseguramos que las columnas coincidan
        []  # Pasamos una lista vacía porque las variables categóricas ya están procesadas
    )
    print(f"Resultados para punto de corte {ptoCorte}:")
    print(resultados_corte)
    resultados_corte['PtoCorte'] = ptoCorte
    todos_resultados_corte.append(resultados_corte)

# Concatenar todos los DataFrames en uno solo
df_resultados_corte = pd.concat(todos_resultados_corte, ignore_index=True)
print(df_resultados_corte.head())

# Graficar sensibilidad, especificidad y exactitud frente a los puntos de corte
plt.figure(figsize=(10, 6))
# Graficar todas las métricas con los puntos de corte en el eje X
plt.plot(df_resultados_corte['PtoCorte'], df_resultados_corte['Sensitivity'], label='Sensibilidad', marker='o')
plt.plot(df_resultados_corte['PtoCorte'], df_resultados_corte['Specificity'], label='Especificidad', marker='o')
plt.plot(df_resultados_corte['PtoCorte'], df_resultados_corte['Accuracy'], label='Exactitud (Accuracy)', marker='o')

# Configuración del gráfico
plt.axvline(x=0.5, color='gray', linestyle='--', label='Punto de corte 0.5')
plt.xticks(puntos_corte)  # Asegurarse de que se muestren todos los puntos de corte en el eje X
plt.xlabel('Punto de Corte')
plt.ylabel('Métricas')
plt.title('Sensibilidad, Especificidad y Accuracy vs Punto de Corte')
plt.legend()
plt.grid(True)
plt.show()

# Seleccionar el punto de corte con menor diferencia entre sensibilidad y especificidad
pto_corte_optimo = df_resultados_corte.iloc[
    (df_resultados_corte['Sensitivity'] - df_resultados_corte['Specificity']).abs().idxmin()
]
print("Punto de corte óptimo:", pto_corte_optimo['PtoCorte'])


# Extraer coeficientes del modelo ganador (Modelo 1)
coeficientes = pd.DataFrame({
    'Variable': x_train_procesado.columns,
    'Coeficiente': modelo_logistico['Modelo'].coef_[0]
})
print("Coeficientes del modelo ganador:")
print(coeficientes)

# Seleccionar una variable binaria y una continua para interpretar
variable_binaria = 'CCAA_Aragón'  # Ajusta según las variables presentes en tu modelo
variable_continua = 'Population'

# Extraer los coeficientes correspondientes
coef_binaria = coeficientes[coeficientes['Variable'] == variable_binaria]
coef_continua = coeficientes[coeficientes['Variable'] == variable_continua]

# Interpretar la variable binaria
if not coef_binaria.empty:
    coef_binaria_valor = coef_binaria['Coeficiente'].values[0]
    print(f"\nInterpretación de la variable binaria ({variable_binaria}):")
    print(f"El coeficiente para '{variable_binaria}' es {coef_binaria_valor:.4f}.")
    print("Esto significa que pertenecer a esta categoría aumenta el logit en esa cantidad.")
    print(f"En términos de probabilidad, el efecto multiplicativo es: {np.exp(coef_binaria_valor):.4f}")
else:
    print(f"\nLa variable binaria '{variable_binaria}' no está en el modelo.")

# Interpretar la variable continua
if not coef_continua.empty:
    coef_continua_valor = coef_continua['Coeficiente'].values[0]
    print(f"\nInterpretación de la variable continua ({variable_continua}):")
    print(f"El coeficiente para '{variable_continua}' es {coef_continua_valor:.4f}.")
    print("Esto significa que por cada unidad adicional en la población, el logit aumenta en esa cantidad.")
    print(f"En términos de probabilidad, el efecto multiplicativo por cada unidad adicional es: {np.exp(coef_continua_valor):.4f}")
else:
    print(f"\nLa variable continua '{variable_continua}' no está en el modelo.")
