# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:41:32 2024

@author: MIGUEL GASPAR PIQUERO
"""

from scipy.stats import skew, kurtosis, kstest, norm
from scipy.stats import stats
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    path = Path.cwd()
    excel_path = str(path.joinpath("datosejercicioevaluacionanchuras.xlsx"))
    
    datos = pd.read_excel(excel_path)
    columnas_relevantes = ['Época histórica', 'Anchura del cráneo']  
    datos = datos[columnas_relevantes]
    
    # Dividir datos por época histórica (Ejercicio1)
    temprano = datos[datos['Época histórica'] == 1]['Anchura del cráneo']
    tardio = datos[datos['Época histórica'] == 2]['Anchura del cráneo']
    
    print("\n--- Análisis Descriptivo para Predinástico Temprano ---")
    analisis_descriptivo(temprano, 'Predinástico Temprano')
    print("\n--- Análisis Descriptivo para Predinástico Tardío ---")
    analisis_descriptivo(tardio, 'Predinástico Tardío')

    # Aplicar el test a cada grupo
    test_kolmogorov(temprano, "Predinástico Temprano")
    test_kolmogorov(tardio, "Predinástico Tardío")
    
    # Aplicar a los datos (Ejercicio2)
    calcular_intervalos(temprano, tardio)
    # Realizar el Test t
    test_t(temprano, tardio)
    
    
def analisis_descriptivo(datos, nombre):
    print(f"Datos para {nombre}:\n{datos.describe()}")
    
    # Medidas de asimetría y curtosis
    asimetria = skew(datos)
    curtosis_val = kurtosis(datos)
    print(f"Asimetría: {asimetria}")
    print(f"Curtosis: {curtosis_val}")

    # Visualización: Histograma
    plt.hist(datos, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Histograma de la Anchura del Cráneo - {nombre}')
    plt.xlabel('Anchura (mm)')
    plt.ylabel('Frecuencia')
    plt.show()

    # Visualización: Boxplot
    plt.boxplot(datos)
    plt.title(f'Diagrama de Caja y Bigotes - {nombre}')
    plt.ylabel('Anchura (mm)')
    plt.show()


def test_kolmogorov(datos, nombre):
    resultado = kstest(datos, 'norm', args=(datos.mean(), datos.std(ddof=1)))
    print(f"\n *** Resultados del test de Kolmogorov-Smirnov para {nombre}:")
    print(f"Estadístico: {resultado.statistic}")
    print(f"p-valor: {resultado.pvalue}")
    if resultado.pvalue > 0.05:
        print(f"No se rechaza la hipótesis nula: {nombre} sigue una distribución normal.\n")
    else:
        print(f"¡SE RECHAZA! la hipótesis nula: {nombre} no sigue una distribución normal.\n")


def calcular_intervalos(temprano, tardio):
    # Medias y desviaciones estándar
    media1, std1, n1 = np.mean(temprano), np.std(temprano, ddof=1), len(temprano)
    media2, std2, n2 = np.mean(tardio), np.std(tardio, ddof=1), len(tardio)
    
    # Diferencia de medias
    diferencia = media1 - media2
    
    # Error estándar
    error_estandar = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    # Intervalos de confianza
    niveles = [0.90, 0.95, 0.99]
    for nivel in niveles:
        z = norm.ppf(1 - (1 - nivel) / 2)  # Usar scipy.stats.norm correctamente
        intervalo = (diferencia - z * error_estandar, diferencia + z * error_estandar)
        print(f"Intervalo de confianza al {int(nivel * 100)}%: {intervalo}")


def test_t(temprano, tardio):
    # Realizar el test t de Student
    resultado = stats.ttest_ind(temprano, tardio, equal_var=False)  # equal_var=False para varianzas distintas
    print("\n*** Resultados del Test t para la Diferencia de Medias:")
    print(f"Estadístico t: {resultado.statistic}")
    print(f"p-valor: {resultado.pvalue}")
    
    # Decisión
    if resultado.pvalue < 0.05:
        print("Rechazamos la hipótesis nula: Hay una diferencia significativa entre las medias.\n")
    else:
        print("No se rechaza la hipótesis nula: No hay evidencia suficiente para afirmar que las medias son diferentes.\n")


if __name__ == "__main__":
    print("Program Start")
    main()
    print("Program End")
