# Cargo las librerias 
import os
import pandas as pd
import numpy as np
import pickle

# Establecemos nuestro escritorio de trabajo
os.chdir(r'C:\Users\MAICKEL\OneDrive\Desktop\Entorno')

from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali, graficoVcramer, mosaico_targetbinaria, boxplot_targetbinaria, 
                           hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
                           modelEffectSizes, crear_data_modelo, Vcramer)

        # Importacion y configuracion de datos
# Leer el archivo Excel
archivo = "DatosEleccionesEspaña.xlsx"
datos = pd.read_excel(archivo)

# Verificar tipos de datos y explorar las columnas
datos.dtypes
cuentaDistintos(datos)
numericasAcategoricas = ['CodigoProvincia', 'Densidad', 'AbstencionAlta', 'Izquierda', 'Derecha']

for var in numericasAcategoricas:  # Las transformo en categóricas
    datos[var] = datos[var].astype(str)
datos.dtypes

# Genera una lista con los nombres de las variables.
variables = list(datos.columns)  

# Seleccionar las columnas numéricas del DataFrame
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleccionar las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]
 
# Comprobamos que todas las variables tienen el formato que queremos  
datos.dtypes

analizar_variables_categoricas(datos)
cuentaDistintos(datos)
descriptivos_num = datos.describe().T
# Añadimos más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)

# Muestra valores perdidos
datos[variables].isna().sum()
# A veces los 'nan' vienen como como una cadena de caracteres, los modificamos a perdidos.
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 
for x in variables:
    datos[x] = datos[x].replace('?', np.nan)
    datos[x] = datos[x].replace(99999, np.nan)


# Variables objetivo seleccionadas
varObjCont = datos['AbstentionPtge']  # Variable continua
varObjBin = datos['AbstencionAlta']  # Variable binaria
variables_objetivo_a_eliminar = ['Name', 'AbstentionPtge', 'AbstencionAlta', 'Izda_Pct', 'Dcha_Pct', 'Otros_Pct', 'Izquierda', 'Derecha']  
datos_input = datos.drop(columns=variables_objetivo_a_eliminar)
# Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  
# Selecionamos las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns
# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]


## ATIPICOS

# Cuento el porcentaje de atipicos de cada variable. 

# Seleccionar las columnas numéricas en el DataFrame
# Calcular la proporción de valores atípicos para cada columna numérica
# utilizando una función llamada 'atipicosAmissing'
# 'x' representa el nombre de cada columna numérica mientras se itera a través de 'numericas'
# 'atipicosAmissing(datos_input[x])' es una llamada a una función que devuelve una dupla
# donde el segundo elemento ([1]) es el númeron de valores atípicos
# 'len(datos_input)' es el número total de filas en el DataFrame de entrada
# La proporción de valores atípicos se calcula dividiendo la cantidad de valores atípicos por el número total de filas
resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

# Modifico los atipicos como missings
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]

# MISSINGS
# Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.
patron_perdidos(datos_input)

# Muestra total de valores perdidos por cada variable
datos_input[variables_input].isna().sum()

# Muestra proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = datos_input.isna().sum()/len(datos_input)

# Creamos la variable prop_missings que recoge el número de valores perdidos por cada observación
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)

# Realizamos un estudio descriptivo básico a la nueva variable
datos_input['prop_missings'].describe()

# Calculamos el número de valores distintos que tiene la nueva variable
len(datos_input['prop_missings'].unique())

# Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes)
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')


## IMPUTACIONES
# Imputo todas las cuantitativas, seleccionar el tipo de imputacion: media, mediana o aleatorio
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'aleatorio')

# Imputo todas las cualitativas, seleccionar el tipo de imputacion: moda o aleatorio
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Reviso que no queden datos missings
datos_input.isna().sum()

# Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardamos
datos_tarea = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datos_tarea.pickle', 'wb') as archivo:
    pickle.dump(datos_tarea, archivo)


