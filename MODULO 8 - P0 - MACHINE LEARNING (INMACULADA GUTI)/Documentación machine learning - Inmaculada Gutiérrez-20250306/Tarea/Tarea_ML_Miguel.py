# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:11:32 2025

@author: MAICKEL
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc, roc_auc_score

# Cargar el archivo Excel
archivo_tarea = r'C:\Users\MAICKEL\OneDrive\Documents\MASTER_UCM\MODULO 8 - P0 - MACHINE LEARNING (INMACULADA GUTI)\Documentación machine learning - Inmaculada Gutiérrez-20250306\Tarea\datos_tarea25.xlsx'
df = pd.read_excel(archivo_tarea)
print(df.head())

# Limpieza de la columna Levy:
df['Levy'] = df['Levy'].replace('-', np.nan)
# Crear un imputador para reemplazar los NaN por la mediana
imputer = SimpleImputer(strategy='median')
df['Levy'] = imputer.fit_transform(df[['Levy']]) # Imputar la mediana en la columna 'Levy'
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')

# Limpieza de la columna Mileage:
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

# Limpiar la columna 'Engine volume', eliminando ' Turbo'
df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False)  # Eliminar ' Turbo'
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')  # Convertir a numérico

# Codificación One-Hot para la columna 'Wheel' eliminando la primera columna, con 1 es suficiente
df = pd.get_dummies(df, columns=['Wheel'], drop_first=True)

# Listar todas las columnas numéricas y categoricas que necesitan ser escaladas y tratadas respectivamente
numeric_columns = ['Price', 'Levy', 'Mileage', 'Engine volume', 'Airbags']
categorical_columns = ['Manufacturer', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels']

encoder = OneHotEncoder(sparse_output=False) # Inicializar el codificador OneHotEncoder
encoded_features = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
df = df.join(encoded_df).drop(columns=categorical_columns)

# Codificar 'Leather interior' con 1 para 'Yes' y 0 para 'No'
df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0})


# Revisar duplicados:
print("Duplicados:", df.duplicated().sum())
# Eliminar duplicados exactos
df = df.drop_duplicates()
# Verificar si todavía quedan duplicados
print("Duplicados después de eliminar:", df.duplicated().sum())

# Revisar la variable Color para homogeneizar valores:
df['Color'] = df['Color'].str.strip().str.capitalize()
print(df['Color'].value_counts())
# Codificar la columna 'Color' en 0 (Black) y 1 (White)
df['Color'] = df['Color'].apply(lambda x: 1 if x == 'White' else 0)
print(df['Color'].value_counts())
# **************   FIN DEPURACIÓN DATOS    ****************


seed = 123     # Creo la semilla de aleatoriedad

X = df.drop(columns=['Color'])  # Eliminar la columna 'Color' de las características
y = df['Color']  # La variable dependiente es 'Color'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

scaler = StandardScaler()
X_train_tr = scaler.fit_transform(X_train)
X_test_tr = scaler.transform(X_test)


# --------- CREACION MEJOR MODELO SVM (2 KERNELS) ---------
# Crear el modelo SVM
# Definir los parámetros para la búsqueda de GridSearch
param_grid = {
    'C': [0.1, 1],  # Menos valores de C
    'kernel': ['linear', 'rbf'],  # Mantén los dos kernels
    'gamma': ['scale', 'auto']  # Solo probamos 'scale' para gamma
}

# Crear el clasificador SVM
svm = SVC(probability=True, random_state=seed)
# Configurar GridSearchCV con validación cruzada
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=3, verbose=3)
grid_search.fit(X_train_tr, y_train) # Ajustar el modelo SVM con el conjunto de entrenamiento

# Obtener los mejores parámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# Mostrar los mejores parámetros encontrados
print(f"Mejores parámetros encontrados: {best_params}")
print(f"Mejor estimador encontrado: {best_model}")


# Ver los resultados completos de la búsqueda de hiperparámetros
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)

# Predecir sobre el conjunto de prueba
y_pred = best_model.predict(X_test_tr)
y_prob = best_model.predict_proba(X_test_tr)[:, 1]  # Probabilidades para la AUC

# Calcular la precisión (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del mejor modelo: {accuracy * 100:.2f}%")

# Calcular la AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"AUC del mejor modelo: {roc_auc:.2f}")

# Graficar la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el mejor modelo SVM')
plt.legend(loc='lower right')
plt.show()

# Predicciones con el mejor modelo encontrado
grid_predictions = grid_search.predict(X_test_tr)

# Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, grid_predictions))
# Matriz de confusión
cm = confusion_matrix(y_test, grid_predictions)
print("Matriz de Confusión:")
print(cm)
# Calcular accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / (cm.sum())
print(f'Accuracy con el mejor modelo: {accuracy * 100:.2f}%')



# ----------   CREAR MODELO BAGGING A PARTIR DEL MEJOR MODELO ---------------
# Crear el clasificador de Bagging con el mejor modelo SVM
bagging_model = BaggingClassifier(estimator=best_model,  # Cambiar 'base_estimator' por 'estimator'
                                 n_estimators=50,  # Número de modelos SVM
                                 random_state=seed)

bagging_model.fit(X_train_tr, y_train)

# Predicciones del modelo de Bagging
y_pred_bagging = bagging_model.predict(X_test_tr)
y_prob_bagging = bagging_model.predict_proba(X_test_tr)[:, 1]  # Probabilidades para AUC

# Calcular la precisión (Accuracy)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"Precisión del modelo Bagging: {accuracy_bagging * 100:.2f}%")

# Calcular AUC
roc_auc_bagging = roc_auc_score(y_test, y_prob_bagging)
print(f"AUC del modelo Bagging: {roc_auc_bagging:.2f}")

print(f"Precisión del SVM original: {accuracy * 100:.2f}%")
print(f"AUC del SVM original: {roc_auc:.2f}")
print(f"Precisión del Bagging: {accuracy_bagging * 100:.2f}%")
print(f"AUC del Bagging: {roc_auc_bagging:.2f}")

fpr_bagging, tpr_bagging, thresholds_bagging = roc_curve(y_test, y_prob_bagging)
roc_auc_val_bagging = auc(fpr_bagging, tpr_bagging)

plt.figure(figsize=(8, 6))
plt.plot(fpr_bagging, tpr_bagging, color='darkorange', lw=2, label=f'Bagging ROC curve (AUC = {roc_auc_val_bagging:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el modelo de Bagging')
plt.legend(loc='lower right')
plt.show()


# PUNTO 3
# MODELO STACKING 1
base_learners = [
    ('svm', SVC(probability=True, kernel='linear', random_state=seed)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=seed)),
    ('knn', KNeighborsClassifier())
]

meta_model = LogisticRegression(random_state=seed)

stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
stacking_model.fit(X_train_tr, y_train)

# Predicciones del modelo de Stacking
y_pred_stacking = stacking_model.predict(X_test_tr)
y_prob_stacking = stacking_model.predict_proba(X_test_tr)[:, 1]  # Probabilidades para AUC

# Calcular la precisión (Accuracy)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Precisión del modelo de Stacking: {accuracy_stacking * 100:.2f}%")

# Calcular AUC
roc_auc_stacking = roc_auc_score(y_test, y_prob_stacking)
print(f"AUC del modelo de Stacking: {roc_auc_stacking:.2f}")

# GRAFICAR CURVA ROC
fpr_stacking, tpr_stacking, thresholds_stacking = roc_curve(y_test, y_prob_stacking)
roc_auc_val_stacking = auc(fpr_stacking, tpr_stacking)

plt.figure(figsize=(8, 6))
plt.plot(fpr_stacking, tpr_stacking, color='darkorange', lw=2, label=f'Stacking ROC curve (AUC = {roc_auc_val_stacking:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el modelo de Stacking')
plt.legend(loc='lower right')
plt.show()


# MODELO STACKING 2
# Definir los clasificadores base
base_learners = [
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=seed)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=seed)),
    ('knn', KNeighborsClassifier())
]

# Definir el clasificador meta
meta_model = RandomForestClassifier(n_estimators=100, random_state=seed)

# Crear el modelo de Stacking
stacking_model2 = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
stacking_model2.fit(X_train_tr, y_train)

# Predicciones del modelo de Stacking
y_pred_stacking = stacking_model2.predict(X_test_tr)
y_prob_stacking = stacking_model2.predict_proba(X_test_tr)[:, 1]  # Probabilidades para AUC

# Calcular la precisión (Accuracy)
accuracy_stacking2 = accuracy_score(y_test, y_pred_stacking)
print(f"Precisión del modelo de Stacking: {accuracy_stacking * 100:.2f}%")

# Calcular AUC
roc_auc_stacking2 = roc_auc_score(y_test, y_prob_stacking)
print(f"AUC del modelo de Stacking: {roc_auc_stacking:.2f}")

# GRAFICAR CURVA ROC
fpr_stacking, tpr_stacking, thresholds_stacking = roc_curve(y_test, y_prob_stacking)
roc_auc_val_stacking = auc(fpr_stacking, tpr_stacking)

plt.figure(figsize=(8, 6))
plt.plot(fpr_stacking, tpr_stacking, color='darkorange', lw=2, label=f'Stacking ROC curve (AUC = {roc_auc_val_stacking:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el modelo de Stacking 2')
plt.legend(loc='lower right')
plt.show()




# COMPARAR MODELOS
print("\n------- COMPARACION DE TODOS LOS MODELOS --------")
print(f"Precisión del modelo SVM: {accuracy * 100:.2f}%")
print(f"AUC del modelo SVM: {roc_auc:.2f}")
print(f"Precisión del Bagging: {accuracy_bagging * 100:.2f}%")
print(f"AUC del Bagging: {roc_auc_bagging:.2f}")
print(f"Precisión del modelo de Stacking: {accuracy_stacking * 100:.2f}%")
print(f"AUC del modelo de Stacking: {roc_auc_stacking:.2f}")
print(f"Precisión del modelo de Stacking2: {accuracy_stacking2 * 100:.2f}%")
print(f"AUC del modelo de Stacking2: {roc_auc_stacking2:.2f}")



# Simulamos varios resultados de precisión para cada modelo
np.random.seed(123)

# Crear 10 simulaciones de precisión para cada modelo
precisiones_svm = np.random.normal(accuracy * 100, 0.5, 10)  # SVM con desviación de 0.5
precisiones_bagging = np.random.normal(accuracy_bagging * 100, 0.5, 10)  # Bagging
precisiones_stacking = np.random.normal(accuracy_stacking * 100, 0.5, 10)  # Stacking
precisiones_stacking2 = np.random.normal(accuracy_stacking2 * 100, 0.5, 10)  # Stacking 2

# Combinar todas las precisiones en una lista
precisiones = [
    precisiones_svm,
    precisiones_bagging,
    precisiones_stacking,
    precisiones_stacking2
]

# Etiquetas para los modelos
modelos = ['SVM', 'Bagging', 'Stacking', 'Stacking 2']

# Crear el boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=precisiones)

# Personalizar gráfico
plt.xticks(ticks=range(len(modelos)), labels=modelos)
plt.title('Comparación de Modelos en términos de Precisión')
plt.xlabel('Modelos')
plt.ylabel('Precisión (%)')
plt.show()



# Lista de modelos ya definidos
models = [
    ('SVM', best_model), 
    ('Bagging', bagging_model), 
    ('Stacking', stacking_model),
    ('Stacking 2', stacking_model2)
]

# Listas para almacenar las precisiones de cada modelo
precisiones = []

# Aplicar validación cruzada a cada modelo
for model_name, model in models:
    cv_scores = cross_val_score(model, X_train_tr, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
    precisiones.append(cv_scores)

# Crear el boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=precisiones)

# Etiquetas para los modelos
modelos = [model[0] for model in models]

# Personalizar gráfico
plt.xticks(ticks=range(len(modelos)), labels=modelos)
plt.title('Comparación de Modelos en términos de Precisión (Validación Cruzada)')
plt.xlabel('Modelos')
plt.ylabel('Precisión (%)')
plt.show()

for model_name, model in models:
    model.fit(X_train_tr, y_train)
    y_pred = model.predict(X_train_tr)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Modelo: {model_name}")
    print(f"Precisión en Train: {train_accuracy:.4f}")
    print(f"Precisión en Test: {test_accuracy:.4f}")
    print("-" * 50)
    
    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(2, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.show()
    


# SVM opción 2
param_grid = {
    'C': [0.8, 0.9, 1, 1.1, 1.2, 1.3], 
    'kernel': ['linear', 'rbf', 'poly'],  
    'gamma': ['scale', 'auto']  
}

svm = SVC(probability=True, random_state=seed)
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
}

grid_model = GridSearchCV(estimator=svm, param_grid=param_grid, scoring=scoring_metrics, refit='accuracy', cv=5, n_jobs=-1, verbose=3)
grid_model.fit(X_train_tr, y_train) # Ajustar el modelo SVM con el conjunto de entrenamiento
results = pd.DataFrame(grid_model.cv_results_)

print(f"Mejores parámetros encontrados: {grid_model.best_params_}")
print(f"Mejor estimador encontrado: {grid_model.best_estimator_}")

print("Resultados de Grid Search:")
print(results[['params', 'mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro', 'mean_test_f1_macro']])

best_model = grid_model.best_estimator_

y_pred = best_model.predict(X_train_tr)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Precisión en Train: {train_accuracy:.4f}")
print(f"Precisión en Test: {test_accuracy:.4f}")
print("-" * 50, "\n")