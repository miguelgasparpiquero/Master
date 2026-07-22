# Librerías:
import csv

# Constantes globales:
MOVIES_DATA = "./data_in/movie_data.csv"

# Parte A. Ejercicios básicos sin usar pandas
# Esta celda debe ser completada por el estudiante
def load_full_data(data_file):
    """
    Carga y retorna los datos de un archivo CSV, incluyendo los encabezados y el contenido completo.

    Parámetros:
        data_file (str): Ruta y nombre del archivo que contiene los datos en formato CSV.
        
    Retorna:
        tuple: Una tupla con dos elementos:
            - full_header (list): Lista con los encabezados de las columnas.
            - full_list_data (list of lists): Lista de listas, donde cada sublista representa 
              una fila de datos (sin encabezados).
    """
    with open(data_file, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
         # Obtener el encabezado (primera fila del archivo)
        full_header = next(lector_csv)
        # Obtener el resto de los datos (todas las filas después del encabezado)
        full_list_data = [fila for fila in lector_csv]

    # Devolver los encabezados y los datos
    return full_header, full_list_data


# Test de funcionamiento
full_header, full_list_data = load_full_data(MOVIES_DATA)
print(full_header)
print()
print(full_list_data[0:5])
# Enumerar los encabezados
list_of_enumerated_headers = list(enumerate(full_header))
# Imprimir los encabezados enumerados
print(list_of_enumerated_headers)


def main_data_from_item(data):
     """
    Extrae y devuelve datos específicos de una lista en posiciones predefinidas.

    Parámetros:
        data (list): Lista que contiene todos los datos de una película en sus distintas posiciones.
        
    Retorna:
        list: Lista de datos seleccionados en el orden especificado, con las siguientes columnas:
            - movie_title
            - title_year
            - director_name
            - actor_1_name
            - language
            - country
            - color
            - budget
            - imdb_score
            - movie_imdb_link
    """
    columnas_interes = [11, 23, 1, 10, 19, 20, 0, 22, 25, 17]
    
    return [data[i] for i in columnas_interes]

# Test de funcionamiento
print(main_data_from_item(full_header))
print()
datos_avatar_2009 = main_data_from_item(full_list_data[0])
print(datos_avatar_2009)
print()
datos_star_wars_7 = main_data_from_item(full_list_data[4])
print(datos_star_wars_7)


def datatypes_arranged(data):
    """
    Limpia y convierte los datos en los tipos de datos adecuados.

    Procesos de limpieza y conversión:
        - Limpia el título de la película eliminando caracteres especiales como '\xa0'.
        - Convierte el año de estreno en un entero; si está vacío o no es un número, se asigna -1.
        - Convierte el presupuesto en un entero; si está vacío o no es válido, se asigna -1.
        - Convierte la puntuación de IMDb en un número flotante; si está vacío o no es válido, se asigna 0.0.
        - Limpia la URL de IMDb eliminando el fragmento final "?ref_".

    Parámetros:
        data (list): Lista de datos de una película que contiene el título, año, presupuesto, IMDb score y URL, entre otros.
        
    Retorna:
        list: La lista de datos con las conversiones y limpiezas realizadas.
    """
    
    # Limpiar el título de la película eliminando caracteres especiales
    data[0] = data[0].strip().replace('\xa0', '')
    
    # Convertir el año en entero; si está vacío, imputar con -1
    data[1] = int(data[1]) if data[1].isdigit() else -1
    
    # Convertir el presupuesto en entero; si está vacío, imputar con -1
    try:
        data[7] = int(data[7]) if data[7] else -1
    except ValueError:
        data[7] = -1
    
    # Convertir la puntuación IMDb en un número real
    try:
        data[8] = float(data[8]) if data[8] else 0.0
    except ValueError:
        data[8] = 0.0
    
    # Limpiar la URL eliminando el fragmento final "?ref_"
    if "?ref_" in data[9]:
        data[9] = data[9].split("?ref_")[0]
    
    return data

print(datatypes_arranged(datos_avatar_2009))
print(datatypes_arranged(datos_star_wars_7))


# Esta celda debe ser completada por el estudiante
def get_unique_colors(list_data):
    """
   Obtiene un conjunto de colores únicos a partir de una lista de datos de películas.

   Parámetros:
       list_data (list of lists): Lista de listas, donde cada sublista representa los datos de una película.
       
   Retorna:
       set: Conjunto de colores únicos encontrados en los datos, excluyendo valores vacíos y el encabezado "Color".
   """
   
    colors = set()
    
    for data in list_data:
        color = data[0].strip()
        if color and color != "Color":  # Me aseguro de que no esté vacío y no sea el encabezado
            colors.add(color)
    
    return colors

def get_movies_with_victor(list_data, search):
    """
   Recupera títulos de películas que contienen una cadena específica en su título y cuenta el número de calificadores.

   Parámetros:
       list_data (list of lists): Lista de listas donde cada sublista representa los datos de una película.
       search (str): Cadena de texto a buscar en los títulos de las películas.
       
   Retorna:
       list of tuples: Lista de tuplas donde cada tupla contiene:
           - title (str): Título de la película que contiene la cadena de búsqueda.
           - count_qualifiers (int): Número de calificadores en el campo 'actor_1_name'.
           
   Notas:
       - La función limpia los títulos de espacios y caracteres especiales antes de buscar.
       - El campo 'actor_1_name' (columna 3) se usa para contar calificadores, separados por comas.
   """
   
    movies_with_victor = []

    for movie_data in list_data:
        title = movie_data[11].strip().replace('\xa0', '') 
        if search in title:
            # el campo de calificadores es 'actor_1_name' (columna 3)
            qualifiers = movie_data[3].strip().split(",")  
            count_qualifiers = len(qualifiers) if qualifiers[0] else 0  # Contar solo si hay actores
            movies_with_victor.append((title, count_qualifiers))
    
    return movies_with_victor

colores = get_unique_colors(full_list_data)
pelis_victor = get_movies_with_victor(full_list_data, "Victor")

# Test de funcionamiento
print(colores)
print(pelis_victor)


# B Datos en un diccionario [2 puntos]
def load_main_data(data_file):
    """
    Carga y extrae datos específicos de un archivo CSV, retornando los encabezados y un diccionario con el título y año de cada película.

    Parámetros:
        data_file (str): Ruta y nombre del archivo que contiene los datos en formato CSV.
        
    Retorna:
        tuple: Una tupla con dos elementos:
            - main_header (list): Lista con los encabezados de las columnas seleccionadas (título y año).
            - dict_data (dict): Diccionario donde cada clave es el título de una película (str) y cada valor es el año (str) 
              (si no hay año, se asigna '-1').
    """
    
    with open(data_file, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        
        full_header = next(lector_csv)
        main_header = [full_header[11], full_header[23]]
    
        dict_data = {}
        for fila in lector_csv:
            titulo = fila[11].strip()
            año = fila[23].strip() if fila[23].strip() else '-1'  # Limpiar el año, y si está vacío poner -1
            dict_data[titulo] = año  # Guardar en el diccionario
            
    return main_header, dict_data

main_header, main_dict_data = load_main_data(MOVIES_DATA)

print(main_header)
print()
for title_year, pieces in list(main_dict_data.items())[:5]:
    print(title_year, " -> ", pieces)
    

del full_list_data
try:
    print(full_list_data)
except:
    print('La variable full_list_data está suprimida correctamente')
    
  
def movies_anno_for_director(dict_data: dict, busca: str):
    """
    Devuelve los títulos y años de las películas dirigidas por un director específico.

    Parámetros:
        dict_data (dict): Diccionario con los títulos de las películas como claves y los años como valores.
        busca (str): Nombre del director que queremos buscar.
    
    Retorna:
        list of tuples: Lista de tuplas, donde cada tupla contiene:
            - título (str): Título de una película dirigida por el director buscado.
            - año (str): Año de la película, obtenido de `dict_data`.
    
    Notas:
        - La función abre el archivo CSV para buscar coincidencias en el nombre del director (columna 1).
        - Si el director coincide con el nombre buscado y el título existe en `dict_data`, se agrega una tupla (título, año) a la lista de resultados.
    """
    
    data_file = "./data_in/movie_data.csv"
    peliculas = []
    
    with open(data_file, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  # Saltar la línea de encabezado
        
        for fila in lector_csv:
            director = fila[1].strip()  # Limpiar el nombre del director (columna 1)
            titulo = fila[11].strip().replace('\xa0', '')  # Limpiar el título (columna 11)
            
            # Si el director coincide y el título está en dict_data
            if director.lower() == busca.lower() and titulo in dict_data:
                año = dict_data[titulo]  # Obtener el año desde dict_data
                peliculas.append((titulo, año))  # Guardar la tupla (título, año)
    
    return peliculas

movies_anno_for_director(main_dict_data, "James Cameron")
    
  
def directors_max_movies(dict_data):
    """
   Devuelve el director o directores que han dirigido la mayor cantidad de películas y el número total de películas dirigidas.

   Parámetros:
       dict_data (dict): Diccionario con los títulos de las películas como claves y los años como valores.
   
   Retorna:
       tuple: Una tupla que contiene:
           - max_director (list): Lista de los nombres de los directores que han dirigido el máximo número de películas.
           - max_peliculas (int): Número máximo de películas dirigidas por esos directores.
   
   Notas:
       - La función lee un archivo CSV para contar cuántas películas ha dirigido cada director.
       - Si hay múltiples directores con el mismo número máximo de películas, todos serán incluidos en la lista de `max_director`.
   """
   
    data_file = "./data_in/movie_data.csv"
    directores = []
    
    with open(data_file, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  # Saltar la línea de encabezado

        directores = {}
        for fila in lector_csv:
            director = fila[1].strip()  # Limpiar el nombre del director (columna 1)            
            # Si el director coincide y el título está en dict_data
            if director:
                if director not in directores:
                    directores[director] = 1 
                else:
                    directores[director] += 1
                    
        # Encontrar el número máximo de películas dirigidas        
        max_peliculas = max(directores.values())
        # Filtrar los directores que tienen el número máximo de películas
        max_director = [director for director, count in directores.items() if count == max_peliculas]
    
    return max_director, max_peliculas

print(directors_max_movies(main_dict_data))


def years_num_movies(dict_data: dict, start_year: int, end_year: int):
    """
   Cuenta cuántas películas se han realizado en cada año dentro de un intervalo dado.

   Parámetros:
       dict_data (dict): Diccionario con los títulos de las películas como claves y los años como valores.
       start_year (int): Año inicial del intervalo (inclusive).
       end_year (int): Año final del intervalo (inclusive).
   
   Retorna:
       dict: Un diccionario donde las claves son los años (int) y los valores son el número de películas (int)
             que se han realizado en esos años dentro del intervalo dado.
   
   Notas:
       - Si un año no es válido (no puede ser convertido a entero), se omite en el conteo.
   """
    
    peliculas_por_año = {}

    for titulo, año in dict_data.items():
        try:
            año = int(año)  # Convertir el año a entero
        except ValueError:
            continue  # Saltar si el año no es válido

        # Contar solo las películas dentro del intervalo dado
        if start_year <= año <= end_year:
            if año in peliculas_por_año:
                peliculas_por_año[año] += 1
            else:
                peliculas_por_año[año] = 1
    
    return dict(peliculas_por_año.items())

num_movies = years_num_movies(main_dict_data, 2000, 2015)
print(num_movies)

num_movies_sorted = sorted(num_movies.items())
print(num_movies_sorted)


FEW_FIELDS = "algunos_campos.txt"
def store_file(dict_data, data_file):
    """
    Función que almacena en un archivo los datos seleccionados de cada película:
    título, idioma, año, país y presupuesto. Los datos se separan con el carácter '|'.
    
    Parámetros:
        dict_data (dict): Diccionario con los títulos como claves y los años como valores. 
                          Este parámetro no se utiliza directamente en la función, pero se menciona para mantener 
                          la consistencia con el encabezado del parámetro.
        data_file (str): Ruta del archivo donde se almacenarán los datos.
    
    Notas:
        - La función abre un archivo en modo escritura y utiliza el módulo csv para escribir los datos.
        - Se espera que los datos provengan de una fuente con un formato específico, donde los índices de columna
          para el título, idioma, año, país y presupuesto son 11, 19, 23, 20 y 22 respectivamente.
        - Si el año o el presupuesto están vacíos, se imputará un valor de '-1' en su lugar.
        - La primera fila del archivo CSV de entrada se omite, ya que se considera que contiene los encabezados.
    """
    with open(data_file, mode='w', newline='', encoding='utf-8') as archivo_salida:
        escritor = csv.writer(archivo_salida, delimiter='|')

        # Iterar sobre los datos en el diccionario original y escribir los campos seleccionados
        with open(MOVIES_DATA, mode='r', newline='', encoding='utf-8') as archivo_csv:
            lector_csv = csv.reader(archivo_csv)
            next(lector_csv)  # Saltar la cabecera
            
            for fila in lector_csv:
                # Extraer los campos de interés (título, idioma, año, país, presupuesto)
                titulo = fila[11].strip()  # Columna del título
                idioma = fila[19].strip()  # Columna del idioma
                año = fila[23].strip() if fila[23].strip() else '-1'  # Columna del año, imputar -1 si está vacío
                pais = fila[20].strip()  # Columna del país
                presupuesto = fila[22].strip() if fila[22].strip() else '-1'  # Columna del presupuesto, imputar -1 si está vacío
                
                # Escribir los campos en el archivo con el separador '|'
                escritor.writerow([titulo, idioma, año, pais, presupuesto])

store_file(main_dict_data, FEW_FIELDS)
dir algunos*.*

print()

with open(FEW_FIELDS) as f:
    for i in range(5):
        print(f.readline())


def actor_directors(dict_data):
    """
   Cuenta cuántas veces ha actuado cada actor principal bajo la dirección de un director específico.

   Parámetros:
       dict_data (dict): Diccionario con los títulos de las películas como claves y los años como valores.
   
   Retorna:
       dict: Un diccionario cuyas claves son los nombres de los actores (str) y cuyos valores son
             otros diccionarios, donde las claves son los nombres de los directores (str) y los valores son
             el número de veces que el actor ha trabajado bajo cada director (int).

   Notas:
       - La función no utiliza `defaultdict`, sino que inicializa manualmente los diccionarios anidados.
       - Solo se cuentan las películas en las que hay tanto un director como un actor principal válidos.
   """
   
    actor_director_count = {}

    with open(MOVIES_DATA, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  # Saltar la cabecera

        for fila in lector_csv:
            director = fila[1].strip()  # Columna del director (2da columna, índice 1)
            actor_principal = fila[10].strip()  # Columna del actor principal (11va columna, índice 10)

            if director and actor_principal:
                if actor_principal not in actor_director_count:  # Si no existe en el diccionario (Actor y direcor), los inicializo.
                    actor_director_count[actor_principal] = {}

                if director not in actor_director_count[actor_principal]:
                    actor_director_count[actor_principal][director] = 0

                actor_director_count[actor_principal][director] += 1

    return actor_director_count

# Test de funcionamiento
num_collaborations = actor_directors(main_dict_data)

print(type(num_collaborations))
key_a, value_a = list(num_collaborations.items())[0]
print(type(key_a), type(value_a))
key_b, value_b = list(value_a.items())[0]
print(type(key_b), type(value_b))

print()

print(num_collaborations)


def print_collaborations_min(dict_data, min_colaboraciones):
    """
    Función que cuenta cuántas veces ha actuado cada actor como actor principal bajo la dirección
    de un director específico, mostrando solo aquellos que superan un número mínimo de colaboraciones.

    Parámetros:
        dict_data (dict): Diccionario donde las claves son actores y los valores son diccionarios
                          con directores y el número de colaboraciones.
        min_colaboraciones (int): Número mínimo de colaboraciones para que se muestren.

    Retorna:
        Un diccionario cuyas claves serán los nombres de los actores y cuyos valores serán diccionarios
        con los directores y el número de veces que han trabajado juntos, pero solo si superan el mínimo.
    """
    # Filtrar los actores con colaboraciones que superen el mínimo
    resultado_filtrado = {}
    for actor, directores in dict_data.items():
        # Crear un diccionario solo con directores que superen el mínimo
        directores_filtrados = {director: veces for director, veces in directores.items() if veces >= min_colaboraciones}
        
        # Solo incluir el actor si tiene colaboraciones que superan el mínimo
        if directores_filtrados:
            resultado_filtrado[actor] = directores_filtrados

    # Imprimir el resultado filtrado
    for actor, directores in resultado_filtrado.items():
        print(f"Actor: {actor}")
        for director, veces in directores.items():
            print(f"  Director: {director} - Colaboraciones: {veces}")
    

# Test de funcionamiento
print_collaborations_min(num_collaborations, 5)



# C. Algunos gráficos sencillos
import matplotlib.pyplot as plt

def representar_xxx_yyy(pares, labels=None):
    """
    Representa una gráfica a partir de una lista de pares (x, y).

    Parámetros:
        pares (list of tuples): Lista de pares de valores (x, y) donde cada par representa un punto en la gráfica.
        labels (list, optional): Lista de tres etiquetas [título, etiqueta de ordenadas (Y), etiqueta de abcisas (X)] 
                                  para personalizar la gráfica. Si no se proporciona, se utilizan etiquetas predeterminadas.
    
    Notas:
        - Se utiliza `matplotlib` para generar la gráfica.
        - La gráfica se ajusta automáticamente para mejorar la presentación.
        - Las etiquetas del eje X se rotan 45 grados para una mejor legibilidad.
    """
    
    # Desempaquetar los pares en dos listas: una para las x y otra para las y
    x_vals, y_vals = zip(*pares)
    
    # Crear la gráfica
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
    
    if labels:   # Configurar los rótulos si los hay
        plt.title(labels[0])            # Titulo
        plt.ylabel(labels[1])           # Etiqueta del eje Y
        plt.xlabel(labels[2])           # Etiqueta del eje X
    else:
        plt.title("Gráfica de pares")    # Título predeterminado
        plt.ylabel("Y")                 # Etiqueta predeterminada del eje Y
        plt.xlabel("X")                 # Etiqueta predeterminada del eje X    
    
    plt.xticks(rotation=45)  # Rotar las etiquetas de las abcisas
    
    # Mostrar la gráfica
    plt.grid(True) # Cuadrícula
    plt.tight_layout() # Ajustar márgenes y espaciado de los elementos
    plt.show()

representar_xxx_yyy([(1, 8), (2, 4), (3, 2), (4, 1), (5, 0.5), (6, 0.25)], ["Serie descendente", "Ordenadas", "Abcisas"])
representar_xxx_yyy([(1, 1), (2, 2), (3, 4), (4, 8), (5, 16), (6, 32)])


def years_num_movies(dict_data: dict, start_year: int, end_year: int):
    """
   Cuenta cuántas películas se han realizado en cada año dentro de un intervalo dado.

   Parámetros:
       dict_data (dict): Diccionario con los títulos de las películas como claves y los años como valores.
       start_year (int): Año inicial del intervalo (inclusive).
       end_year (int): Año final del intervalo (inclusive).
   
   Retorna:
       dict: Un diccionario donde las claves son los años (int) y los valores son el número de películas (int)
             que se han realizado en esos años dentro del intervalo dado.
   
   Notas:
       - Si un año no es válido (no puede ser convertido a entero), se omite en el conteo.
   """
   
    peliculas_por_año = {}

    for titulo, año in dict_data.items():
        try:
            año = int(año)
        except ValueError:
            continue

        if start_year <= año <= end_year:
            if año in peliculas_por_año:
                peliculas_por_año[año] += 1
            else:
                peliculas_por_año[año] = 1

    return peliculas_por_año





def repr_movies_years(dict_data, start_year, end_year):
    """
    Representa el número de películas por año en un intervalo de años dado.

    Parámetros:
        dict_data (dict): Diccionario con los títulos de las películas como claves y los años como valores.
        start_year (int): Año inicial del intervalo (inclusive).
        end_year (int): Año final del intervalo (inclusive).
    
    Notas:
        - La función utiliza `years_num_movies` para obtener el conteo de películas en el intervalo especificado.
        - Se genera un gráfico de barras con `matplotlib`, donde el eje X representa los años y el eje Y representa 
          el número de películas.
        - Las etiquetas del eje X se rotan 45 grados para mejorar la legibilidad.
    """
    
    # Obtener los datos de películas por año en el intervalo dado
    peliculas_por_año = years_num_movies(dict_data, start_year, end_year)
    
    years = sorted(peliculas_por_año.keys())
    num_movies = [peliculas_por_año[year] for year in years]
    
    # Crear la gráfica
    plt.bar(years, num_movies, color='skyblue')
    
    # Títulos y etiquetas
    plt.title(f'Número de películas entre {start_year} y {end_year}')
    plt.xlabel('Año')
    plt.ylabel('Número de películas')
    
    # Inclinación de las etiquetas en el eje X
    plt.xticks(rotation=45)
    
    # Activar el grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar el diseño para evitar solapamientos
    plt.tight_layout()
    
    # Mostrar la gráfica
    plt.show()
    
repr_movies_years(main_dict_data, 2000, 2010)
repr_movies_years(main_dict_data, 2005, 2015)


# D. Acceso a las urls de imdb y webscraping
# Función para cargar todas las URLs de las películas desde el archivo CSV
def extract_movie_urls(data_file):
    """
    Extrae las URLs de las películas de un archivo CSV.

    Parámetros:
        data_file (str): Ruta y nombre del archivo CSV que contiene los datos de las películas.
    
    Retorna:
        list: Lista de URLs de las películas extraídas del archivo. Las URLs se limpian para eliminar cualquier
              parte que comience con "?ref".
    
    Notas:
        - La función asume que las URLs están en la columna 17 del archivo CSV (índice 16).
        - Se omite la primera fila del archivo, que se asume es la línea de encabezado.
        - Si una fila no contiene una URL válida, no se añadirá nada a la lista de URLs.
    """
    
    urls = []  # Lista donde se almacenarán las URLs
    
    with open(data_file, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  # Saltar la línea de encabezado
        
        for fila in lector_csv:
            if "?ref_" in fila[17]:
                url = fila[17].split("?ref")[0]  # Queda solo la parte antes de ?ref
            urls.append(url)
    
    # Devolver la lista completa de URLs
    return urls
    
urls = extract_movie_urls(MOVIES_DATA)
first_url_movie = urls[0] if urls else "No hay URLs disponibles"  # La primera URL
first_ten_urls = urls[:10]  # Las primeras 10 URLs (si existen)

print(len(urls))
print()
print(first_url_movie)
print()
print(first_ten_urls)


import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

def soup_movie(url):
    """
    Extrae el código HTML de una película dada su URL.

    Parámetros:
        url (str): La URL de la película en IMDb.

    Retorna:
        BeautifulSoup: Objeto BeautifulSoup que representa el código HTML de la página.
                       Devuelve None si hay un error en la solicitud.

    Notas:
        - Se utiliza la biblioteca `requests` para realizar la solicitud HTTP.
        - Se incluyen encabezados personalizados en la solicitud para evitar bloqueos.
        - Si la solicitud falla (por ejemplo, por problemas de conexión o un código de estado HTTP no exitoso),
          se captura la excepción y se imprime un mensaje de error.
    """
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Lanza un error si la solicitud no es exitosa

        # Analizar el contenido HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup

    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")
        return None

# Test de funcionamiento
soup = soup_movie(first_url_movie)

print(str(soup)[:1000])
print()
print("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...")
print()
print(str(soup)[-1000:])


def extract_movie_info(url):
    """
    Extrae información de la película desde la URL dada.

    Parámetros:
        url (str): La URL de la película en IMDb.

    Retorna:
        dict: Un diccionario que contiene la información de la película, incluyendo:
              - 'title': Título de la película (str).
              - 'description': Descripción de la película (str).
              - 'actors': Lista de actores del reparto principal (list).
              - 'budget': Presupuesto de la película (str), si está disponible.
              Retorna None si no se puede obtener información.

    Notas:
        - Se utiliza la función `soup_movie` para obtener el objeto BeautifulSoup del código HTML de la página.
        - Se manejan casos donde los elementos pueden no estar presentes (por ejemplo, si el presupuesto no se especifica).
    """
    
    soup = soup_movie(url)
    if not soup:
        return None

    movie_info = {}

    # Extraer el título de la película
    title_tag = soup.find('h1')
    if title_tag:
        movie_info['title'] = title_tag.text.strip()

    # Extraer la descripción de la película
    description_tag = soup.find('div', class_='summary_text')
    if description_tag:
        movie_info['description'] = description_tag.text.strip()

    # Extraer la lista de actores del reparto principal
    actors_tags = soup.find_all('a', class_='sc-16z54d8-1 fZyZxu')
    actors = [actor.text.strip() for actor in actors_tags]
    movie_info['actors'] = actors

    # Extraer el presupuesto
    budget_tag = soup.find('div', string='Budget:')
    if budget_tag:
        # El siguiente elemento después de "Budget:" contiene el valor
        budget_value = budget_tag.find_next_sibling().text.strip()
        movie_info['budget'] = budget_value

    return movie_info

# Test de funcionamiento
movie_info = extract_movie_info(first_url_movie)

if movie_info:
    print("Título:", movie_info.get('title'))
    print("Descripción:", movie_info.get('description'))
    print("Reparto Principal:", movie_info.get('actors'))
    print("Presupuesto:", movie_info.get('budget'))


import re  # Importa la biblioteca de expresiones regulares

def gather_actors(filename, urls):
   """
   Extrae actores del reparto principal de las películas a partir de una lista de URLs y los guarda en un archivo.

   Parámetros:
       filename (str): Nombre del archivo donde se guardarán los actores.
       urls (list): Lista de URLs de las películas.

   Notas:
       - Se utiliza la función `soup_movie` para obtener el contenido HTML de cada URL.
       - La función busca la etiqueta `<meta>` con el atributo `name="description"` para extraer la descripción de la película.
       - Se utiliza una expresión regular para encontrar y extraer la lista de actores a partir de la descripción.
       - Si no se encuentran actores, se registra un mensaje en el archivo indicando que no se encontraron actores.
       - Los actores se guardan en el archivo en el formato "URL: [url]\nActores:\n[lista de actores]\n".
   """
   
    with open(filename, 'w', encoding='utf-8') as file:
        for url in urls:
            soup = soup_movie(url)
            if not soup:
                continue
            
            # Encontrar la etiqueta <meta> con la descripción
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if not meta_tag:
                file.write(f"URL: {url}\n")
                file.write("No se encontraron actores.\n\n")
                continue
            
            # Obtener el contenido de la descripción
            description = meta_tag['content']
            # print(f"URL: {url}\nDescripción encontrada: {description}")
        
            # Utilizar expresión regular para extraer los actores
            # Utilizar expresión regular para extraer los actores
            match = re.search(r'With (.+?)\.', description)  # Ajustada
            if match:
                # Obtener los actores, separando por comas y limpiando espacios
                actors = match.group(1).split(', ')
                actors = [actor.strip() for actor in actors]
            
                # Guardar los actores en el archivo
                file.write(f"URL: {url}\n")
                file.write("Actores:\n")
                file.write(", ".join(actors) + "\n\n")

            else:
                file.write(f"URL: {url}\n")
                file.write("No se encontraron actores en la descripción.\n\n")
                
# Test de funcionamiento
# OJO: esta operación puede llevar bastante tiempo.
# Para esta prueba, usamos un número limitado de películas.
gather_actors("actors_3_first_movies.txt", urls[:3])
#! type actors_3_first_movies.txt

# La siguente llamada llevaría un tiempo realmente largo:
# import time # para cronometrar esta función, que tarda mucho
# reloj_inicio = time.time()
# gather_actors("actors_all_movies.txt", urls)
# reloj_fin = time.time()

# print("Tiempo invertido: %s segundos." % (reloj_fin - reloj_inicio))


# E. Pandas [2 puntos]
# Esta celda debe ser completada por el estudiante
import pandas as pd

def load_dataframe(filepath):
    """
    Carga un archivo CSV en un DataFrame de pandas.

    Parámetros:
        filepath (str): Ruta al archivo CSV que se va a cargar.

    Retorna:
        pandas.DataFrame: Un DataFrame que contiene los datos del archivo CSV.
    """
    
    # Carga el archivo en un DataFrame, ajustando el delimitador si es necesario
    return pd.read_csv(filepath, delimiter=',', encoding='utf-8')
    
tabla_completa = load_dataframe(MOVIES_DATA)
tabla_completa


def fields_selected_dataframe(dataframe):
    """
    Filtra las columnas seleccionadas del DataFrame original.

    Parámetros:
        dataframe (pandas.DataFrame): El DataFrame original del cual se filtrarán las columnas.

    Retorna:
        pandas.DataFrame: Un nuevo DataFrame que contiene solo las columnas seleccionadas.

    Notas:
        - Se seleccionan las siguientes columnas: 
            'movie_title', 'title_year', 'director_name', 'actor_1_name',
            'language', 'country', 'color', 'budget', 'imdb_score', 
            'movie_imdb_link'.
        - Si alguna de las columnas seleccionadas no existe en el DataFrame original, se generará un error.
    """
    
    # Columnas que deseamos conservar
    selected_columns = [
        'movie_title', 'title_year', 'director_name', 'actor_1_name',  
        'language', 'country', 'color', 'budget', 'imdb_score', 
        'movie_imdb_link']
    
    return dataframe[selected_columns]

tabla_breve = fields_selected_dataframe(tabla_completa)
tabla_breve

print("Las columnas de nuestra tabla breve son: \n", tabla_breve.columns)

tabla_breve = tabla_breve.fillna("Desc")
tabla_breve


def titulos_de_director_df(dataframe, director_name):
    """
    Filtra las películas de un director específico y devuelve un DataFrame con esos títulos, limpiando caracteres especiales.

    Parámetros:
        dataframe (pandas.DataFrame): DataFrame que contiene los datos de las películas.
        director_name (str): Nombre del director para filtrar.

    Retorna:
        pandas.DataFrame: DataFrame que contiene únicamente los títulos de las películas del director especificado.

    Notas:
        - Se realiza una copia del DataFrame original para evitar modificarlo directamente.
        - Se normalizan y limpian los títulos de las películas para eliminar caracteres especiales y espacios en blanco al inicio y al final.
        - Si no hay películas del director en el DataFrame, se devolverá un DataFrame vacío.
    """
    
    # Filtrar las filas donde el nombre del director coincida
    director_movies = dataframe[dataframe["director_name"] == director_name].copy()
    
    # Limpiar el texto en la columna de títulos
    director_movies.loc[:, "movie_title"] = director_movies["movie_title"].str.normalize('NFKC').str.strip()
    
    return director_movies[["movie_title"]]

tabla_tits = titulos_de_director_df(tabla_breve, "James Cameron")
tabla_tits
list_tits = tabla_tits["movie_title"].to_list()
print(list_tits)


def directors_max_movies_df(dataframe):
    """
    Identifica los directores que han dirigido el mayor número de películas y devuelve
    sus nombres junto con el número de películas.

    Parámetros:
        dataframe (pandas.DataFrame): DataFrame que contiene los datos de las películas.

    Retorna:
        pandas.DataFrame: Un DataFrame con los directores que han dirigido el mayor número de películas
                          y el conteo de esas películas.

    Notas:
        - La función filtra los directores que tienen el nombre 'Desc', ya que estos no representan directores reales.
        - Se utiliza la función groupby de pandas para contar cuántas películas ha dirigido cada director.
        - En caso de que haya múltiples directores con el mismo número máximo de películas, todos se incluirán en el DataFrame resultante.
    """
    # Filtrar los directores que no son 'Desc'
    filtered_dataframe = dataframe[dataframe['director_name'] != 'Desc']
    # Contar el número de películas por director
    director_counts = filtered_dataframe.groupby("director_name").size().reset_index(name='movie_count')
    
    # Obtener el número máximo de películas
    max_movies = director_counts["movie_count"].max()
    
    # Filtrar los directores que tienen el máximo número de películas
    max_directors = director_counts[director_counts["movie_count"] == max_movies]
    
    return max_directors

directors_max_movies_df(tabla_breve)


# Parte F. Un cálculo masivo con map-reduce en language_budgets_country.py

# Parte G. Un apartado libre [0.5 puntos]
# Este apartado debe ser completado por el estudiante
"""
El objetivo de este proyecto es analizar los presupuestos de películas a partir de un archivo de datos en formato 
de texto delimitado por '|'. Se busca responder a las siguientes preguntas:
Para ello se parte del archivo generado con mapreduce del apartado anterior "language_countries_budgets.txt"

¿Cuál es el idioma con el presupuesto total más alto y en qué país se produce la mayoría de las películas en ese idioma? 
¿Cómo se distribuyen los presupuestos de las películas por idioma en un gráfico? ¿Existen diferencias significativas en 
los presupuestos entre los países que producen películas en un mismo idioma? Para llevar a cabo este análisis, utilizaremos 
la librería Pandas para el manejo de datos y Seaborn para la visualización gráfica. Se presentarán los resultados en un 
informe que incluye gráficos que ilustren los hallazgos de manera clara y atractiva.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def analizar_presupuestos(file_path):
    """
    Analiza los presupuestos de películas por idioma y país desde un archivo de texto.

    Esta función lee un archivo que contiene datos de presupuesto organizados por idioma y país,
    calcula el presupuesto total por idioma, identifica el idioma con el presupuesto más alto,
    y visualiza los resultados a través de gráficos de barras.

    Parámetros:
    -----------
    file_path : str
        La ruta del archivo de texto que contiene los datos de idioma, país y presupuesto.
        El archivo debe tener un formato donde cada línea contiene una lista de idioma y país
        seguido del presupuesto, separado por tabulaciones.

    Salida:
    -------
    - Imprime un resumen con el idioma que tiene el mayor presupuesto.
    - Muestra un gráfico de barras que representa la distribución de presupuestos por idioma.
    - Muestra un gráfico de barras que representa la distribución de presupuestos por país
      para el idioma con el mayor presupuesto.

    Ejemplo de uso:
    ---------------
    >>> analizar_presupuestos('ruta/al/archivo/language_countries_budgets.txt')
    """
    # Creamos listas para almacenar los datos
    idiomas = []
    paises = []
    presupuestos = []

    # Leemos el archivo línea por línea
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')# Separar los datos en idioma, país y presupuesto
            
            if len(parts) == 2:  # Asegúrate de que hay dos partes (idioma y presupuesto)
                idioma_pais = ast.literal_eval(parts[0])  # Convierte el string a tupla
                presupuesto = float(parts[1])  # Presupuesto como número
                
                # Añadimos los datos a las listas
                idiomas.append(idioma_pais[0])  # Idioma
                paises.append(idioma_pais[1])   # País
                presupuestos.append(presupuesto) # Presupuesto
    
    df = pd.DataFrame({   # Creamos el DataFrame
        'Idioma': idiomas,
        'País': paises,
        'Presupuesto': presupuestos})

    
    print(df.head())  # Mostramos las primeras filas del DataFrame

    # Idioma con el Presupuesto Total Más Alto
    presupuesto_por_idioma = df.groupby('Idioma')['Presupuesto'].sum().reset_index()
    idioma_max = presupuesto_por_idioma.loc[presupuesto_por_idioma['Presupuesto'].idxmax()]

    print(f"Idioma con mayor presupuesto: {idioma_max['Idioma']} con un total de {idioma_max['Presupuesto']}")

    # Distribución de Presupuestos por Idioma
    plt.figure(figsize=(10, 6))
    sns.barplot(data=presupuesto_por_idioma, x='Idioma', y='Presupuesto', color='blue')  # Cambiamos palette a color
    plt.title('Distribución de Presupuestos por Idioma')
    plt.xlabel('Idioma')
    plt.ylabel('Presupuesto Total')
    plt.xticks(rotation=60, ha='right', fontsize=10)  # Rotar las etiquetas del eje x más y ajustarlas a la derecha plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Presupuestos por País en el Idioma con Mayor Presupuesto
    df_idioma_max = df[df['Idioma'] == idioma_max['Idioma']]
    presupuesto_por_pais = df_idioma_max.groupby('País')['Presupuesto'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=presupuesto_por_pais, x='País', y='Presupuesto', color='orange')  # Cambiamos palette a color
    plt.title(f'Distribución de Presupuestos por País')
    plt.xlabel('País')
    plt.ylabel('Presupuesto Total')
    plt.xticks(rotation=60, ha='right', fontsize=10)  # Rotar las etiquetas del eje x más y ajustarlas a la derecha plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Pruebas de funcionamiento, también tarea del estudiante:
analizar_presupuestos("language_countries_budgets.txt")
