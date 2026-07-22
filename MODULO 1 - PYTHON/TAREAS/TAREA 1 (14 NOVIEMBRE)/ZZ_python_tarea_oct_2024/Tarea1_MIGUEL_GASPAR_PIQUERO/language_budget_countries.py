# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:17:37 2024

@author: MAICKEL
"""

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRLanguajeBudgetCountries(MRJob):
    """
    Un trabajo de MapReduce que calcula el presupuesto total de películas 
    por combinación de idioma y país.

    Formato de entrada:
        Una línea de texto por película, con los campos delimitados por '|':
        título, idioma, año, país y presupuesto.

    Salida:
        La salida consiste en una clave que es una tupla (idioma, país) y 
        un valor que representa el presupuesto total de las películas en 
        ese idioma y país.
    """

    def mapper(self, _, line):
        """
        Función de mapeo que procesa cada línea de la entrada para extraer 
        el idioma, país y presupuesto.

        Parámetros:
            - _: (str) Clave de entrada no utilizada en MRJob.
            - line: (str) Una línea de texto que representa los datos de una película.

        Emite:
            - key: (tuple) Tupla que contiene el idioma y el país.
            - value: (float) Presupuesto de la película si es positivo.

        Si la línea no tiene al menos cinco campos o el presupuesto no es 
        un número válido, la entrada se ignora.
        """
        # Separa la línea por el delimitador '|'
        fields = line.split('|')

        if len(fields) >= 5:  # Asegúrate de que haya al menos 5 campos
            idioma = fields[1].strip()  # Suponiendo que el idioma está en el segundo campo
            pais = fields[3].strip()     # Suponiendo que el país está en el cuarto campo
            try:
                presupuesto = float(fields[4].strip())  # Suponiendo que el presupuesto está en el quinto campo
                if idioma and pais and presupuesto > 0:  # Solo procesamos si idioma y país no están vacíos y el presupuesto es positivo
                    yield (idioma, pais), presupuesto
            except ValueError:
                pass  # Si el presupuesto no es un número válido, lo ignoramos

    def reducer(self, key, values):
        """
        Función de reducción que agrega el presupuesto total para cada 
        combinación única de idioma y país.

        Parámetros:
            - key: (tuple) Tupla (idioma, país) como clave.
            - values: (iterable de float) Presupuestos de películas con el mismo idioma y país.

        Retorna:
            - key: (tuple) Clave (idioma, país).
            - total_presupuesto: (float) Suma total de presupuestos para esa clave.
        """
        total_presupuesto = sum(values)
        yield key, total_presupuesto

    def steps(self):
        """
       Define los pasos del trabajo de MapReduce.

       Retorna:
           - lista de MRStep: Contiene los pasos de mapeo y reducción.
       """
        return [
            MRStep(mapper=self.mapper,
                    reducer=self.reducer)
        ]

if __name__ == '__main__':
    MRLanguajeBudgetCountries.run()