# Importamos las bibliotecas necesarias
import os
import pandas as pd    
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import warnings

# Establecer el nivel de advertencias a "ignore" para ignorar todas las advertencias
warnings.filterwarnings("ignore")

os.chdir(r'C:\Users\MAICKEL\OneDrive\Documents\MASTER_UCM\MODULO 7 -P2- MINERIA DE DATOS (CLUSTERS)\Documentación minería de datos y modelización predictiva  - Daniel Martín-20250121\Datos') 
datos = pd.read_excel('penguins.xlsx') 
df = datos.select_dtypes(include=[np.number]) # Seleccionar solo las columnas numéricas

# Visualizamos las primeras filas del DataFrame para verificar los datos
df.head()


# Calculamos distancias sin estandarizar
# Calcula la matriz de distancias Euclidianas entre las observaciones
distance_matrix = distance.cdist(df, df, 'euclidean')
# Crea un DataFrame a partir de la matriz de distancias con los índices de df
distance_df = pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
# La distance_matrix es una matriz 2D que contiene las distancias Euclidianas 
# entre todas las parejas de observaciones.
# Seleccionamos las primeras 5 filas y columnas de la matriz de distancias
distance_small = distance_matrix[:5, :5]
# Agregamos los nombres de índice a la matriz de distancias
distance_small = pd.DataFrame(distance_small, index=df.index[:5], columns=df.index[:5])
# Redondeamos los valores en la matriz de distancias
distance_small_rounded = distance_small.round(2)
distance_small_rounded

# Representamos gráficamente la matriz de distancias
# Crea una nueva figura para el gráfico con un tamaño específico
plt.figure(figsize=(10, 8))
# Genera un mapa de calor usando Seaborn
# - `distance_df`: DataFrame de pandas que contiene los datos para el mapa de calor.
# - `annot=False`: No muestra las anotaciones (valores de los datos) en las celdas del mapa.
# - `cmap="YlGnBu"`: Utiliza la paleta de colores "Yellow-Green-Blue" para el mapa de calor.
# - `fmt=".1f"`: Formato de los números en las anotaciones, en este caso no se usan.
sns.heatmap(distance_df, annot=False, cmap="YlGnBu", fmt=".1f")
# Muestra el gráfico
plt.show()



# Realizamos clustering jerárquico para obtener la matriz de enlace (linkage matrix). 
# Clustermap es una función compleja que combina un mapa de calor con dendrogramas para mostrar la agrupación de datos.
# Aquí, estamos usando el dataframe 'distance_df' que contiene las distancias euclidianas entre las observaciones.
# La opción 'cmap' establece la paleta de colores a 'YlGnBu', que es un gradiente de azules y verdes.
# La opción 'fmt' se usa para formatear las anotaciones numéricas, en este caso a un decimal.
# La opción 'annot=False' indica que no queremos anotaciones numéricas en las celdas del mapa de calor.
# La opción 'method' especifica el método de agrupamiento a usar, en este caso 'average' que calcula la media de las distancias.
linkage = sns.clustermap(distance_df, cmap="YlGnBu", fmt=".1f", annot=False, method='ward').dendrogram_row


# Estandarizamos los datos
# Inicializamos el escalador de estandarizacion
scaler = StandardScaler()
# Ajustamos y transformamos el DataFrame para estandarizar las columnas
# 'fit_transform' primero calcula la media y la desviacion estandar de cada columna para luego realizar la estandarizacion.
df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# Asignamos el indice del DataFrame original 'df' al nuevo DataFrame 'df_std'
# Esto es importante para mantener la correspondencia de los indices de las filas despues de la estandarizacion.
df_std.index = df.index
df_std


# Calculamos las distancias euclidianas por pares entre las filas del DataFrame estandarizado
# 'cdist' calcula la distancia euclidiana entre cada par de filas en 'df_std'.
# Esto resulta en una matriz de distancias donde cada elemento [i, j] es la distancia entre la fila i y la fila j.
distance_std = distance.cdist(df_std, df_std, 'euclidean') 
# Imprimimos los primeros 5x5 elementos de la matriz de distancias para tener una vista previa
print(distance_std[:5,:5].round(2))


# Esto determina las dimensiones del grafico
plt.figure(figsize=(10, 8))
# Creamos un nuevo DataFrame para la matriz de distancias
# 'distance_std' se convierte en un DataFrame con indices y columnas correspondientes a 'df_std'
# Esto facilita la interpretacion del mapa de calor, ya que las filas y columnas estaran etiquetadas con los indices de 'df_std'
df_std_distance = pd.DataFrame(distance_std, index=df_std.index, columns=df_std.index)
# Generamos un mapa de calor utilizando Seaborn
# - 'df_std_distance': DataFrame que contiene los datos de distancia para visualizar.
# - 'annot=False': No muestra anotaciones numericas en cada celda del mapa de calor.
# - 'cmap="YlGnBu"': Define una paleta de colores en tonos de azul y verde para el mapa de calor.
# - 'fmt=".1f"': Formato de los numeros en las anotaciones, en este caso, un decimal.
sns.heatmap(df_std_distance, annot=False, cmap="YlGnBu", fmt=".1f")
# Mostramos el grafico resultante
plt.show()


# Realizamos clustering jerárquico para obtener la matriz de enlace (linkage matrix) sobre las distancias estandarizadas. 
linkage = sns.clustermap(df_std_distance, cmap="YlGnBu", fmt=".1f", annot=False, method='ward').dendrogram_row


# Establecemos un umbral de color para el dendrograma
color_threshold = 10
linkage_matrix = sch.linkage(df_std, method='ward')  # Puedes elegir un metodo de enlace diferente si es necesario
# Creamos el dendrograma con el umbral de color especificado
dendrogram = sch.dendrogram(linkage_matrix, labels=df_std_distance.index.tolist(), leaf_rotation=90, color_threshold=color_threshold)
# Mostramos el dendrograma
plt.show()


# Asignamos las observaciones de datos a 4 clusters
# Especificamos el numero de clusters a formar
num_clusters = 4
# Asignamos los datos a los clusters
# 'fcluster' forma clusters planos a partir de la matriz de enlace 'linkage_matrix'
# 'criterion='maxclust'' significa que formaremos un numero maximo de 'num_clusters' clusters
cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
# Mostramos las asignaciones de clusters
print("Cluster Assignments:", cluster_assignments) 
# Creamos una nueva columna 'Cluster4' y asignamos los valores de 'cluster_assignments' a ella
# Ahora 'df' contiene una nueva columna 'Cluster4' con las asignaciones de cluster
df['Cluster4'] = cluster_assignments


# Visualización de la distribución espacial de los clusters
# Paso 1: Realizar PCA
pca = PCA(n_components=2)  # Inicializamos PCA para 2 componentes principales
eliminar = ['Cluster4']
principal_components = pca.fit_transform(df.drop(eliminar, axis=1))  # Transformamos los datos a 2 componentes
fit = pca.fit(df_std)
autovalores = fit.explained_variance_
autovalores[0]/sum(autovalores)
autovalores[1]/sum(autovalores)
# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(df.drop(eliminar, axis=1)), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=df.index)
# Añadimos las componentes principales a la base de datos estandarizada.
df_z_cp = pd.concat([df_std, resultados_pca], axis=1)
# Calculo la matriz de correlaciones entre veriables y componentes
Correlaciones_var_comp = df_z_cp.corr()
Correlaciones_var_comp = Correlaciones_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]
Correlaciones_var_comp


# Creamos un nuevo DataFrame para los componentes principales 2D
# Nos aseguramos de que df_pca tenga el mismo índice que df
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)

# Paso 2: Crear un gráfico de dispersión con colores para los clusters
plt.figure(figsize=(10, 6))  # Establecemos el tamaño del gráfico

# Recorremos las asignaciones únicas de clusters y trazamos puntos de datos con el mismo color
for cluster in np.unique(cluster_assignments):
    cluster_indices = df_pca.loc[cluster_assignments == cluster].index
    plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                df_pca.loc[cluster_indices, 'PC2'],
                label=f'Cluster {cluster}')  # Etiqueta para cada cluster
    # Anotamos cada punto con el nombre del país
    for i in cluster_indices:
        plt.annotate(i,
                     (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']), fontsize=10,
                     textcoords="offset points",  # cómo posicionar el texto
                     xytext=(0,10),  # distancia del texto a los puntos (x,y)
                     ha='center')  # alineación horizontal puede ser izquierda, derecha o centro

# Líneas de referencia para los ejes x e y
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

plt.title("Gráfico de PCA 2D con Asignaciones de Cluster")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid()
plt.show()


# Metodo del codo
# Creamos un array para almacenar los valores de WCSS para diferentes valores de K
wcss = []
    
for k in range(1, 11):  # Puedes elegir un rango diferente de valores de K
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)  # Inserta es el valor de WCSS

# Graficamos los valores de WCSS frente al numero de grupos (K) y buscamos el punto "codo"
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Metodo del Codo')
plt.xlabel('Numero de Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Metodo de la silueta  
# Creamos un array para almacenar los puntajes de silueta para diferentes valores de K
silhouette_scores = []
    
# Ejecutamos el clustering K-means para un rango de valores de K y calculamos el puntaje de silueta para cada K
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_std, labels)
    silhouette_scores.append(silhouette_avg)
    
# Graficamos los puntajes de silueta frente al numero de clusters (K)
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Metodo de la Silueta')
plt.xlabel('Numero de Clusters (K)')
plt.ylabel('Puntaje de Silueta')
plt.grid(True)
plt.show()





# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Analisis no jerarquico
# Configurar el número de clusters (k=4)
k = 4
# Inicializar el modelo KMeans
# 'n_clusters=k' indica el número de clusters a formar
# 'random_state=0' asegura la reproducibilidad de los resultados
# 'n_init=10' indica el número de veces que el algoritmo se ejecutará con diferentes centroides iniciales
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
# Ajustar el modelo KMeans a los datos estandarizados
# 'df_std' es el DataFrame que contiene los datos previamente estandarizados
kmeans.fit(df_std)
# Obtener las etiquetas de los clusters para los datos
# 'kmeans.labels_' contiene la asignación de cada punto a un cluster
kmeans_cluster_labels = kmeans.labels_
# Creamos una nueva columna 'Cluster' y asignamos los valores de 'kmeans_cluster_labels' a ella
# 'Cluster4_v2' sera el nombre de la nueva columna en el DataFrame 'df'
df['Cluster4_v2'] = kmeans_cluster_labels
# Ahora 'df' contiene una nueva columna 'Cluster4_v2' con las asignaciones de cluster
# Imprimimos los valores de la columna 'Cluster4_v2' para verificar las asignaciones de cluster
print(df["Cluster4_v2"])



# Calculamos los valores de silueta para cada observación
silhouette_values = silhouette_samples(df_std, kmeans.labels_)
    
# Configuramos el tamaño de la figura para el gráfico
plt.figure(figsize=(8, 6))
y_lower = 10  # Inicio del margen inferior en el gráfico

# Iteramos sobre los 4 clusters para calcular los valores de silueta y dibujar el gráfico
for i in range(4):
    # Extraemos los valores de silueta para las observaciones en el cluster i
    ith_cluster_silhouette_values = silhouette_values[kmeans.labels_ == i]
    # Ordenamos los valores para que el gráfico sea más claro
    ith_cluster_silhouette_values.sort()
    
    # Calculamos donde terminarán las barras de silueta en el eje y
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    # Elegimos un color para el cluster
    color = plt.cm.get_cmap("Spectral")(float(i) / 4)
    # Rellenamos el gráfico entre un rango en el eje y con los valores de silueta
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    # Etiquetamos las barras de silueta con el número de cluster en el eje y
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Actualizamos el margen inferior para el siguiente cluster
    y_lower = y_upper + 10  # 10 para el espacio entre clusters

# Títulos y etiquetas para el gráfico
plt.title("Gráfico de Silueta para los Clusters")
plt.xlabel("Valores del Coeficiente de Silueta")
plt.ylabel("Etiqueta del Cluster")
plt.grid(True)  # Añadimos una cuadrícula para mejor legibilidad
plt.show()  # Mostramos el gráfico resultante



# Visualizacion de la distribucion espacial de los clusters
plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura

# Iterar a traves de las etiquetas unicas de clusters y graficar puntos de datos con el mismo color
for cluster in np.unique(kmeans_cluster_labels):
    cluster_indices = df_pca.loc[kmeans_cluster_labels == cluster].index
    plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                df_pca.loc[cluster_indices, 'PC2'],
                label=f'Cluster {cluster}')  # Poner una etiqueta para cada cluster
    
    # Anotar cada punto con su respectivo indice
    for i in cluster_indices:
        plt.annotate(i,
                     (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']),fontsize=10,
                     textcoords="offset points",  # Define como se posicionara el texto
                     xytext=(0,10),  # Define la distancia del texto a los puntos (x,y)
                     ha='center')  # Define la alineacion horizontal del texto

# Líneas de referencia para los ejes x e y
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

# Configurar el titulo y las etiquetas de los ejes del grafico
plt.title("Grafico 2D de PCA con Asignaciones de Cluster KMeans")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()  # Mostrar la leyenda
plt.grid()  # Mostrar la cuadricula
plt.show()  # Mostrar el grafico


# Caracterizamos cada cluster
# Añadimos las etiquetas como una nueva columna al DataFrame original
df['Cluster'] = kmeans.labels_
# Ordenamos el DataFrame por la columna "Cluster"
df_sort = df.sort_values(by="Cluster")

# Agrupamos los datos por la columna 'Cluster' y calculamos la media de cada grupo
# Esto proporcionará las coordenadas de los centroides de los clusters en el espacio de los datos originales
cluster_centroids_orig = df_sort.groupby('Cluster').mean()
# Redondeamos los valores para facilitar la visualización
cluster_centroids_orig.round(2)
# 'cluster_centroids_orig' ahora contiene los centroides de cada cluster en el espacio de los datos originales












