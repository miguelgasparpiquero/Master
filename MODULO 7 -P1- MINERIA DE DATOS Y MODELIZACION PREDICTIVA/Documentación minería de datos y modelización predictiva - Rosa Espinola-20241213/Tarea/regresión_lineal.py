# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:33:19 2025

@author: MAICKEL
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import mean_squared_error
os.chdir(r'C:\Users\MAICKEL\OneDrive\Desktop\Entorno')

# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import (graficoVcramer, mosaico_targetbinaria, boxplot_targetbinaria, 
                           hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
                           modelEffectSizes, crear_data_modelo, Vcramer, Rsq, lm, lm_forward, lm_backward, lm_stepwise, validacion_cruzada_lm,
                           crear_data_modelo, modelEffectSizes)

# Parto de los datos ya depurados
with open('datos_tarea.pickle', 'rb') as f:
    datos = pickle.load(f)

# Defino las variables objetivo y las elimino del conjunto de datos input
varObjCont = datos['AbstentionPtge']  # Variable continua
varObjBin = datos['AbstencionAlta']  # Variable binaria
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1) 
 
# Genera una lista con los nombres de las variables.
variables = list(datos_input.columns)  

# Obtengo la importancia de las variables
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)

# Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjBin)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer},
                             ignore_index=True)

# Veo graficamente el efecto de dos variables cualitativas sobre la binaria
# Tomo las variables con más y menos relación con la variable objetivo Binaria
#mosaico_targetbinaria(datos_input['DifComAutonPtge'], varObjBin, 'DifComAutonPtge')
#mosaico_targetbinaria(datos_input['Name'], varObjBin, 'Name')

# Veo graficamente el efecto de dos variables cuantitativas sobre la binaria
boxplot_targetbinaria(datos_input['totalEmpresas'], varObjBin,'Objetivo', 'totalEmpresas')
boxplot_targetbinaria(datos_input['TotalCensus'], varObjBin, 'Objetivo', 'TotalCensus')

hist_targetbinaria(datos_input['totalEmpresas'], varObjBin, 'totalEmpresas')
hist_targetbinaria(datos_input['TotalCensus'], varObjBin, 'TotalCensus')

# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_input'
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(8, 6))
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
# Establecer el título del gráfico
plt.title("Matriz de correlación")
# Mostrar el gráfico de la matriz de correlación
plt.show()

# Tras Eliminar variables altamente correlacionadas vuelvo a analizarlas
# Eliminar las nuevas variables redundantes basadas en correlaciones altas
variables_a_eliminar_nuevas = ['TotalCensus', 'Age_0-4_Ptge', 'ActividadPpal', 'inmuebles']
datos_input = datos_input.drop(columns=variables_a_eliminar_nuevas)

# Calcular la matriz de correlación actualizada para las variables numéricas restantes
numericas_finales = datos_input.select_dtypes(include=['int', 'float']).columns
matriz_corr_final = datos_input[numericas_finales].corr(method='pearson').abs()

# Crear una máscara para la mitad superior de la matriz de correlación
mask = np.triu(np.ones_like(matriz_corr_final, dtype=bool))
# Graficar la matriz de correlación actualizada
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(matriz_corr_final, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
plt.title("Matriz de correlación (Variables finales)")
plt.show()
# Identificar nuevamente pares con correlaciones altas
pares_altamente_correlacionados_finales = [
    (var1, var2, matriz_corr_final.loc[var1, var2])
    for var1 in matriz_corr_final.columns
    for var2 in matriz_corr_final.columns
    if var1 != var2 and matriz_corr_final.loc[var1, var2] > 0.8
]
# Convertir en DataFrame para visualizar
pares_correlacionados_finales_df = pd.DataFrame(
    pares_altamente_correlacionados_finales, columns=["Variable 1", "Variable 2", "Correlación"]
)

variables = list(datos_input.columns) 
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
categoricas = [variable for variable in variables if variable not in numericas]

# Guardar datos para la regresion
datos_tarea = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datos_regresion.pickle', 'wb') as archivo:
    pickle.dump(datos_tarea, archivo)




## Comenzamos con la regresion lineal

with open('datos_regresion.pickle', 'rb') as f:
    datos = pickle.load(f)
    
varObjCont = datos['AbstentionPtge']  # Variable continua
varObjBin = datos['AbstencionAlta']  # Variable binaria
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1) 

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_input, np.ravel(varObjCont), test_size = 0.2, random_state = 123456)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
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


# Seleccion de variables Stepwise, métrica AIC
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont1, var_categ1, [], 'AIC')
modeloStepAIC['Modelo'].summary() # Resumen del modelo
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X']) # R-squared del modelo para train
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
x_test_modeloStepAIC = x_test_modeloStepAIC.reindex(columns=modeloStepAIC['X'].columns, fill_value=0)
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC) # R-squared del modelo para test


# Seleccion de variables Backward, métrica AIC
modeloBackAIC = lm_backward(y_train, x_train, var_cont1, var_categ1, [], 'AIC')
modeloBackAIC['Modelo'].summary()
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X']) # R-squared del modelo para train
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])
x_test_modeloBackAIC = x_test_modeloBackAIC.reindex(columns=modeloBackAIC['X'].columns, fill_value=0)
Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)

# Importancia de las variables para el modelo Stepwise AIC
print("Importancia de variables para modelo Stepwise AIC:")
importancia_stepAIC = modelEffectSizes( modeloStepAIC, y_train, x_train, 
                                        modeloStepAIC['Variables']['cont'], 
                                        modeloStepAIC['Variables']['categ'], 
                                        modeloStepAIC['Variables']['inter'])
# Importancia de las variables para el modelo Backward AIC
print("Importancia de variables para modelo Backward AIC:")
importancia_backAIC = modelEffectSizes( modeloBackAIC, y_train, x_train, 
                                        modeloBackAIC['Variables']['cont'], 
                                        modeloBackAIC['Variables']['categ'], 
                                        modeloBackAIC['Variables']['inter'])
# Filtrar variables significativas (R² > 0.0012)
umbral_importancia = 0.0012  # Ajusta este umbral según tu análisis
variables_significativas_stepAIC = importancia_stepAIC[importancia_stepAIC['R2'] > umbral_importancia]
variables_significativas_backAIC = importancia_backAIC[importancia_backAIC['R2'] > umbral_importancia]


# Modelo de selección aleatoria
variables_seleccionadas = {'Formula': [], 'Variables': []}
for i in range(15):  # Iterar (ajustar el número de iteraciones)
    x_train_iter, _, y_train_iter, _ = train_test_split(x_train, y_train, test_size=0.3, random_state=123456 + i)
    modelo = lm_stepwise(y_train_iter, x_train_iter, var_cont1, var_categ1, [], 'BIC')
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))
    
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
    frec_ordenada['Formula'][1])]

# Inicializamos un DataFrame para almacenar los resultados
results = pd.DataFrame({
    'Modelo': [],
    'Fold': [],
    'R²': []
})

# Definir los modelos y variables
modelos = {
    'Stepwise AIC': modeloStepAIC,
    'Backward AIC': modeloBackAIC,
    'Random Model 1': var_1,
    'Random Model 2': var_2
}

# Validación cruzada para cada modelo
for nombre, modelo in modelos.items():
    print(f"Validación cruzada para el modelo: {nombre}")
    if 'Modelo' in modelo:  # Modelos generados por selección clásica
        r_squared = validacion_cruzada_lm(5, x_train, y_train,
                                          modelo['Variables']['cont'], 
                                          modelo['Variables']['categ'], 
                                          modelo['Variables']['inter'])
    else:  # Modelos aleatorios
        r_squared = validacion_cruzada_lm(5, x_train, y_train,
                                          modelo['cont'], 
                                          modelo['categ'], 
                                          modelo['inter'])
    
    # Guardamos los resultados
    temp_df = pd.DataFrame({
        'Modelo': [nombre] * len(r_squared),
        'Fold': [f'Fold {i+1}' for i in range(len(r_squared))],
        'R²': r_squared
    })
    results = pd.concat([results, temp_df], ignore_index=True)

# Calcular promedios por modelo
promedios = results.groupby('Modelo')['R²'].mean().reset_index()
print(promedios)

# Crear el boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=results, x='Modelo', y='R²', palette='pastel')

# Configuración del gráfico
plt.title('Distribución de R² por Modelo', fontsize=16)
plt.ylabel('R²', fontsize=12)
plt.xlabel('Modelo', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# Función para contar el número de parámetros de un modelo
def contar_parametros(modelo):
    if 'Modelo' in modelo:  # Para modelos clásicos (Stepwise o Backward)
        return len(modelo['Variables']['cont']) + len(modelo['Variables']['categ']) + len(modelo['Variables']['inter'])
    else:  # Para modelos aleatorios
        return len(modelo['cont']) + len(modelo['categ']) + len(modelo['inter'])

# Contar parámetros de cada modelo
parametros_stepwise = contar_parametros(modeloStepAIC)
parametros_backward = contar_parametros(modeloBackAIC)
parametros_random1 = contar_parametros(var_1)
parametros_random2 = contar_parametros(var_2)

# Mostrar resultados
print(f"Stepwise AIC: {parametros_stepwise} parámetros")
print(f"Backward AIC: {parametros_backward} parámetros")
print(f"Random Model 1: {parametros_random1} parámetros")
print(f"Random Model 2: {parametros_random2} parámetros")


# Obtener el summary del modelo para identificar los coeficientes
summary_stepwise = modeloStepAIC['Modelo'].summary()
print(summary_stepwise)

# Seleccionar las variables para análisis
variable_binaria = 'WomanPopulationPtge'  # Sustituir con una variable binaria del modelo
variable_categorica = 'Densidad'   # Sustituir con una variable categórica del modelo

# Coeficiente y significancia de la variable binaria
coef_binaria = summary_stepwise.tables[1].data
for row in coef_binaria:
    if variable_binaria in row[0]:
        print(f"Coeficiente de la variable binaria {variable_binaria}: {row[1]}")
        print(f"Valor p: {row[4]}")
        print(f"Intervalo de confianza: {row[5]} a {row[6]}")

# Coeficientes y significancia de las categorías de la variable categórica
print(f"Análisis de la variable categórica {variable_categorica}:")
for row in coef_binaria:
    if variable_categorica in row[0]:
        print(f"Categoría: {row[0]}, Coeficiente: {row[1]}, Valor p: {row[4]}")

