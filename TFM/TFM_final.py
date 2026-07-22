#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================
# 0) IMPORTS Y AJUSTES DE ENTORNO
# ============================================
import warnings
warnings.filterwarnings("ignore")

import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline


pd.set_option("display.max_columns", 200) # formato de salida
pd.set_option("display.width", 140)

seed = 42


# In[2]:


# ============================================
# 1) CARGA Y FILTRO INICIAL (Zaragoza)
# ============================================
# Ruta local del CSV
DATA_PATH = "pisos.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

# Eliminar columnas irrelevantes
cols_to_drop = ["Unnamed: 0", "photo", "description", "recomendado", "Num Photos"]
df_cleaned = df.drop(columns=cols_to_drop, errors="ignore")

# Filtrar Zaragoza
df_zaragoza = df_cleaned[df_cleaned["region"].astype(str).str.lower() == "zaragoza"].copy()

print("Filas Zaragoza:", df_zaragoza.shape[0])


# In[3]:


# ============================================
# 2) LIMPIEZA DE FORMATOS NUMÉRICOS / TEXTO
# ============================================
def clean_numeric(series: pd.Series) -> pd.Series:
    """Convierte strings con separadores y símbolos a float."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r"[^\d.,]", "", regex=True)
              .str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False),
        errors="coerce"
    )

df_zaragoza["price"]    = clean_numeric(df_zaragoza["price"])
df_zaragoza["size"]     = clean_numeric(df_zaragoza["size"])
df_zaragoza["price/m2"] = clean_numeric(df_zaragoza["price/m2"])
df_zaragoza["rooms"]    = pd.to_numeric(df_zaragoza["rooms"], errors="coerce")
df_zaragoza["bathrooms"]= pd.to_numeric(df_zaragoza["bathrooms"], errors="coerce")
df_zaragoza["location"] = df_zaragoza["location"].astype(str).str.partition(" (")[0]
df_zaragoza = df_zaragoza.dropna(subset=["price", "size", "rooms", "bathrooms", "price/m2"])

loc_mean = (df_zaragoza.groupby("location")["price/m2"]
                      .mean()
                      .sort_values(ascending=False))


# Nulos 
null_summary = (df_zaragoza.isnull().mean()*100).round(2).sort_values(ascending=False)
print("\n% Nulos (Top 10):\n", null_summary.head(10))

# eliminar filas con nulos (la pérdida es pequeña, mantiene veracidad)
df_zaragoza = df_zaragoza.dropna()
print("\nFilas tras dropna():", df_zaragoza.shape[0])

# Simplificar 'location' → corresponde a la parte antes del paréntesis
df_zaragoza["location"] = df_zaragoza["location"].astype(str).str.partition(" (")[0]


# In[4]:


# Eliminar nulos críticos
df_zaragoza = df_zaragoza.dropna(subset=["price", "size", "rooms", "bathrooms", "price/m2"])
# Estadísticos descriptivos
desc_price = df_zaragoza["price"].describe()
desc_size = df_zaragoza["size"].describe()
desc_rooms = df_zaragoza["rooms"].describe()
desc_bathrooms = df_zaragoza["bathrooms"].describe()


# In[5]:


# Top y bottom locations por precio/m2
loc_mean = df_zaragoza.groupby("location")["price/m2"].mean().sort_values(ascending=False)
top_loc = loc_mean.head(5)
bottom_loc = loc_mean.tail(5)


# In[6]:


# Histograma de precios
plt.figure(figsize=(7,5))
df_zaragoza["price"].plot.hist(bins=50, edgecolor="black")
plt.title("Distribución de precios de vivienda en Zaragoza")
plt.xlabel("Precio (€)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("hist_precios.png")

# Histograma de tamaños
plt.figure(figsize=(7,5))
df_zaragoza["size"].plot.hist(bins=50, edgecolor="black", color="orange")
plt.title("Distribución de tamaños (m²) en Zaragoza")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("hist_tamano.png")

# Boxplot price/m2 por top 10 locations
top10_locs = df_zaragoza["location"].value_counts().head(10).index
plt.figure(figsize=(10,6))
df_zaragoza[df_zaragoza["location"].isin(top10_locs)].boxplot(column="price/m2", by="location", rot=45)
plt.title("Distribución de precio/m² en las 10 localizaciones con más anuncios")
plt.suptitle("")
plt.ylabel("€/m²")
plt.tight_layout()
plt.savefig("boxplot_loc.png")


# In[7]:


# ============================================
# 3) CODIFICACIÓN ONE-HOT (type, region, location)
# ============================================
categorical_cols = ["type", "region", "location"]
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded_array = ohe.fit_transform(df_zaragoza[categorical_cols])

encoded_df = pd.DataFrame(
    encoded_array,
    columns=ohe.get_feature_names_out(categorical_cols),
    index=df_zaragoza.index
)

df_encoded = pd.concat([df_zaragoza.drop(columns=categorical_cols), encoded_df], axis=1)
print("\nShape tras OHE:", df_encoded.shape)


# In[8]:


# ============================================
# 4) BASELINES (Regresión Lineal vs RandomForest)
#    sin fuga: quitamos 'summary' y 'price/m2' antes
# ============================================
df_model = df_encoded.copy().drop(columns=["summary", "price/m2"], errors="ignore")
y = df_model["price"]
X = df_model.drop(columns=["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Baseline 1: Linear Regression (escalado)
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lin = LinearRegression()
lin.fit(X_train_scaled, y_train)
pred_lin = lin.predict(X_test_scaled)
rmse_lin = mean_squared_error(y_test, pred_lin, squared=False)
r2_lin   = r2_score(y_test, pred_lin)
print(f"\nLinear RMSE: {rmse_lin:,.0f}  |  R²: {r2_lin:.3f}")

# Baseline 2: RandomForest
rf = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_rf = mean_squared_error(y_test, y_pred, squared=False)
r2_rf   = r2_score(y_test, y_pred)
print(f"RandomForest RMSE: {rmse_rf:,.0f} € | R²: {r2_rf:.3f}")

# Métricas extra
mae  = mean_absolute_error(y_test, y_pred)
mape = (np.abs((y_test - y_pred)/y_test).replace([np.inf,-np.inf], np.nan).dropna()).mean()*100
print(f"MAE: {mae:,.0f} €  |  MAPE: {mape:.2f}%")

# Importancias (top 15)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8,5))
imp[::-1].plot(kind="barh"); plt.title("Importancia de variables (RandomForest)")
plt.tight_layout(); plt.show()


# In[9]:


# ============================================
# 5) MARCO TEMPORAL SIMULADO (2018–2023) E INTEGRACIÓN IPV (INE)
# ============================================
# Fechas simuladas
df_encoded = df_encoded.copy()
df_encoded["fecha"]      = pd.date_range(start="2018-01-01", end="2023-12-31", periods=len(df_encoded))
df_encoded["año"]        = df_encoded["fecha"].dt.year
df_encoded["trimestre"]  = df_encoded["fecha"].dt.to_period("Q").astype(str)

# Cargar IPV Aragón desde el Excel del INE
xlsx_path = Path(r"C:\Users\MAICKEL\OneDrive\Documents\MASTER_UCM\TFM\TFM MGP\25171.xlsx")
raw = pd.read_excel(xlsx_path, header=None, engine="openpyxl")

def is_trim(x):
    return isinstance(x, str) and re.fullmatch(r"\d{4}T[1-4]", x.strip()) is not None

# localizar cabeceras y fila 'General'
header_row = next((i for i in range(len(raw)) if sum(is_trim(v) for v in raw.iloc[i,:].tolist()) >= 4), None)
assert header_row is not None, "No se encontró fila de cabeceras 2018T1…"
general_row = next((i for i in range(header_row+1, len(raw))
                    if isinstance(raw.iloc[i,0], str) and raw.iloc[i,0].strip().lower()=="general"), None)
assert general_row is not None, "No se encontró fila 'General'."

quarters, values = [], []
for j, cell in enumerate(raw.iloc[header_row, :]):
    if is_trim(cell):
        quarters.append(cell.strip())
        values.append(raw.iloc[general_row, j])

ipv_df = pd.DataFrame({"trimestre": quarters, "ipv": pd.to_numeric(values, errors="coerce")})
ipv_df["trimestre"] = ipv_df["trimestre"].str.replace("T", "Q", regex=False)

# unir IPV
df_encoded["trimestre"] = df_encoded["trimestre"].astype(str)
df_encoded = df_encoded.merge(ipv_df, on="trimestre", how="left")
print("\nFilas sin IPV:", int(df_encoded["ipv"].isna().sum()))

# Vista de trimestres únicos
print("\nTrimestres-IPv:")
print(df_encoded[["trimestre","ipv"]].drop_duplicates().sort_values("trimestre").to_string(index=False))


# In[ ]:


# ============================================
# 6) CORRELACIONES Y DECISIONES (año vs ipv)
# ============================================
df_num = df_encoded.select_dtypes(include=["number"])
corr = df_num.corr(method="pearson")
plt.figure(figsize=(12,9))
sns.heatmap(corr, cmap="coolwarm", center=0); plt.title("Matriz de Correlación (Pearson)")
plt.show()

# Mostrar pares con alta correlación
threshold = 0.80
high_corr = (corr.where(lambda x: abs(x)>threshold).stack().reset_index())
high_corr = high_corr[high_corr["level_0"] != high_corr["level_1"]]
high_corr.columns = ["Var1","Var2","Correlación"]
print("\nPares con |corr|>0.8:\n", high_corr.sort_values("Correlación", ascending=False).head(10))


# In[ ]:


# Decisiones: quitar 'año' y 'price/m2' para modelar
df_model_final = df_encoded.copy().drop(columns=["summary", "año", "price/m2"], errors="ignore")
print("\nShape df_model_final:", df_model_final.shape)


# In[11]:


# ============================================
# 7) PROYECCIÓN DEFLACTADA
#     price_real_2015 = price / (ipv/100)
# ============================================
# Serie histórica IPV para crecimiento medio
df_ipv_hist = (df_encoded[["trimestre","ipv"]]
               .drop_duplicates()
               .dropna()
               .sort_values("trimestre")
               .reset_index(drop=True))
df_ipv_hist["ipv_shift"] = df_ipv_hist["ipv"].shift(1)
df_ipv_hist["crec_trimestral"] = (df_ipv_hist["ipv"]/df_ipv_hist["ipv_shift"]) - 1
crecimiento_medio = df_ipv_hist["crec_trimestral"].dropna().mean()
print(f"\nCrecimiento trimestral medio histórico ~ {crecimiento_medio*100:.2f}%")

# 7.1 Entrenar modelo para precio en € de 2015
df_def = df_model_final.copy()
df_def["price_real_2015"] = df_def["price"] / (df_def["ipv"]/100.0)

X_def = df_def.drop(columns=["price", "price_real_2015", "trimestre"], errors="ignore")
y_def = df_def["price_real_2015"]

# eliminar posibles datetime
dt_cols = X_def.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns
X_def = X_def.drop(columns=list(dt_cols), errors="ignore")

X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y_def, test_size=0.2, random_state=seed)

rf_def = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
rf_def.fit(X_train_def, y_train_def)

pred_def = rf_def.predict(X_test_def)
rmse_def = mean_squared_error(y_test_def, pred_def, squared=False)
r2_def   = r2_score(y_test_def, pred_def)
print(f"[Deflactado] RMSE: {rmse_def:,.0f}  |  R²: {r2_def:.3f}")

cols_def = X_train_def.columns  # columnas usadas

# 7.2 Construir 8 trimestres futuros (2024Q1–2025Q4) para IPV
ult_trim = df_ipv_hist["trimestre"].iloc[-1]     # p.ej. '2023Q4'
anio, trim = int(ult_trim[:4]), int(ult_trim[-1])
ipv_actual = df_ipv_hist["ipv"].iloc[-1]

futuros = []
for _ in range(8):
    trim = 1 if trim == 4 else (trim + 1)
    if trim == 1:
        anio += 1
    ipv_actual *= (1 + crecimiento_medio)
    futuros.append([f"{anio}Q{trim}", ipv_actual])

df_ipv_fut = pd.DataFrame(futuros, columns=["trimestre","ipv"])

# 7.3 Base real: último trimestre histórico
df_base = df_model_final[df_model_final["trimestre"] == ult_trim].drop(columns=["price"], errors="ignore")
dt_cols = df_base.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns
df_base = df_base.drop(columns=list(dt_cols), errors="ignore")

# replicar mix con ipv actualizado por trimestre
bloques = []
for _, row in df_ipv_fut.iterrows():
    tmp = df_base.copy()
    tmp["trimestre"] = row["trimestre"]
    tmp["ipv"] = row["ipv"]
    bloques.append(tmp)

df_fut = pd.concat(bloques, ignore_index=True)

# Alinear EXACTAMENTE a columnas del modelo deflactado
faltantes = [c for c in cols_def if c not in df_fut.columns]
for c in faltantes:
    df_fut[c] = 0
extras = [c for c in df_fut.columns if c not in cols_def and c not in ["trimestre","ipv"]]
df_fut.drop(columns=extras, inplace=True, errors="ignore")
X_fut_def = df_fut[cols_def]

# 7.4 Predecir en €2015 y re-inflar con IPV futuro
pred_2015 = rf_def.predict(X_fut_def)
df_fut["pred_price"] = pred_2015 * (df_fut["ipv"]/100.0)

proyeccion = (df_fut.groupby("trimestre", as_index=False)["pred_price"]
              .mean()
              .sort_values("trimestre"))

print("\nProyección (precio medio) por trimestre:")
print(proyeccion.to_string(index=False))

# Gráfico rápido
plt.figure(figsize=(8,4))
plt.plot(proyeccion["trimestre"], proyeccion["pred_price"], marker="o")
plt.xticks(rotation=45); plt.ylabel("€"); plt.title("Proyección de precio medio (deflactado → re‑inflado)")
plt.tight_layout(); plt.show()


# In[12]:


# ========== LINEAR REGRESSION (deflactado) ==========
# Pipeline: escalado + regresión lineal
lin_def = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False)),
    ("lin", LinearRegression())
])

lin_def.fit(X_train_def, y_train_def)

pred_lin_def = lin_def.predict(X_test_def)
rmse_lin_def = mean_squared_error(y_test_def, pred_lin_def, squared=False)
r2_lin_def   = r2_score(y_test_def, pred_lin_def)
print(f"[Deflactado-Linear] RMSE: {rmse_lin_def:,.0f}  |  R²: {r2_lin_def:.3f}")

# guardamos las columnas usadas (mismo set que X_train_def)
cols_def_lin = X_train_def.columns


# In[13]:


# ========== GRADIENT BOOSTING (deflactado) ==========
gbr_def = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=seed
)
gbr_def.fit(X_train_def, y_train_def)

pred_gbr_def = gbr_def.predict(X_test_def)
rmse_gbr_def = mean_squared_error(y_test_def, pred_gbr_def, squared=False)
r2_gbr_def   = r2_score(y_test_def, pred_gbr_def)
print(f"[Deflactado-GBR]   RMSE: {rmse_gbr_def:,.0f}  |  R²: {r2_gbr_def:.3f}")


# In[14]:


# ========== TABLA COMPARATIVA DE MÉTRICAS (deflactado) ==========
resumen_modelos = pd.DataFrame([
    {"Modelo": "RandomForest (def)", "RMSE_€2015": rmse_def,      "R2": r2_def},
    {"Modelo": "LinearReg (def)",    "RMSE_€2015": rmse_lin_def,  "R2": r2_lin_def},
    {"Modelo": "GBR (def)",          "RMSE_€2015": rmse_gbr_def,  "R2": r2_gbr_def},
]).sort_values("RMSE_€2015")
print(resumen_modelos.to_string(index=False))


# In[15]:


# ========== FUNCIÓN GENÉRICA DE PROYECCIÓN (deflactado) ==========
def proyectar_con_modelo_deflactado(model, feature_cols, n_trimestres=8, factor_crec=1.0):
    """
    model: modelo entrenado sobre y = price_real_2015 (por ej., rf_def, lin_def, gbr_def)
    feature_cols: columnas usadas para el fit (p.ej., X_train_def.columns)
    n_trimestres: nº de trimestres a proyectar
    factor_crec: multiplicador sobre crecimiento_medio (0.75=pes, 1.0=cen, 1.25=opt)
    """
    crec = crecimiento_medio * factor_crec

    # --- construir serie IPV futura ---
    ult = df_ipv_hist["trimestre"].iloc[-1]
    an, tr = int(ult[:4]), int(ult[-1])
    ipv_act = df_ipv_hist["ipv"].iloc[-1]
    fut = []
    for _ in range(n_trimestres):
        tr = 1 if tr == 4 else tr + 1
        if tr == 1: an += 1
        ipv_act *= (1 + crec)
        fut.append([f"{an}Q{tr}", ipv_act])
    df_ipv_fut_local = pd.DataFrame(fut, columns=["trimestre","ipv"])

    # --- base real último trimestre ---
    df_base = df_model_final[df_model_final["trimestre"] == ult].drop(columns=["price"], errors="ignore")
    dt_cols = df_base.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns
    df_base = df_base.drop(columns=list(dt_cols), errors="ignore")

    # --- replicar mix por trimestre futuro ---
    blocks = []
    for _, r in df_ipv_fut_local.iterrows():
        tmp = df_base.copy()
        tmp["trimestre"] = r["trimestre"]
        tmp["ipv"] = r["ipv"]
        blocks.append(tmp)
    df_fut_local = pd.concat(blocks, ignore_index=True)

    # --- alinear columnas EXACTAMENTE a feature_cols ---
    falt = [c for c in feature_cols if c not in df_fut_local.columns]
    for c in falt: df_fut_local[c] = 0
    extra = [c for c in df_fut_local.columns if c not in feature_cols and c not in ["trimestre","ipv"]]
    df_fut_local.drop(columns=extra, inplace=True, errors="ignore")
    X_fut_local = df_fut_local[feature_cols]

    # --- predecir € 2015 y re‑inflar ---
    pred_2015 = model.predict(X_fut_local)
    df_fut_local["pred_price"] = pred_2015 * (df_fut_local["ipv"]/100.0)

    return df_fut_local.groupby("trimestre", as_index=False)["pred_price"].mean()


# In[16]:


# -- Proyectar € --
proj_rf_cen  = proyectar_con_modelo_deflactado(rf_def,  cols_def,      n_trimestres=8, factor_crec=1.0)
proj_lin_cen = proyectar_con_modelo_deflactado(lin_def, cols_def_lin,  n_trimestres=8, factor_crec=1.0)
proj_gbr_cen = proyectar_con_modelo_deflactado(gbr_def, X_train_def.columns, n_trimestres=8, factor_crec=1.0)

# Comparativa en una tabla
proy_comp = proj_rf_cen.merge(proj_lin_cen, on="trimestre", suffixes=("_RF","_LIN")).merge(
    proj_gbr_cen, on="trimestre")
proy_comp = proy_comp.rename(columns={"pred_price":"pred_price_GBR"})
print("\nProyección central comparada (precio medio por trimestre):")
print(proy_comp.to_string(index=False))


# In[17]:


# ========== GRÁFICO SOLO GBR ==========
plt.figure(figsize=(8,4))
plt.plot(proj_gbr_cen["trimestre"], proj_gbr_cen["pred_price"], marker="o", color="green")
plt.xticks(rotation=45)
plt.ylabel("€")
plt.title("Proyección de precio medio por trimestre – GBR (escenario central)")
plt.tight_layout()
plt.savefig("proyeccion_trimestral.png")  # gráfico citado en la memoria
plt.show()


# In[18]:


# ========== GRÁFICO COMPARATIVO ==========
plt.figure(figsize=(9,4))

# Dibuja primero GBR para que RF quede por encima y se vea
plt.plot(proj_gbr_cen["trimestre"], proj_gbr_cen["pred_price"],
         marker="s", linestyle="-.", linewidth=2, label="GBR (def)", zorder=1)

# RF por encima, con línea discontinua y marcadores huecos
plt.plot(proj_rf_cen["trimestre"], proj_rf_cen["pred_price"],
         marker="o", linestyle="--", linewidth=2.5, label="RF (def)", zorder=3)

# Lineal diferente para no confundir
plt.plot(proj_lin_cen["trimestre"], proj_lin_cen["pred_price"],
         marker="^", linestyle="-", linewidth=2, label="Linear (def)", zorder=2)

plt.xticks(rotation=45)
plt.ylabel("€")
plt.title("Proyección de precio medio por trimestre (escenario central)")
plt.legend()
plt.tight_layout()
plt.show()


# In[19]:


# Usamos K-Fold con 5 particiones
cv = KFold(n_splits=5, shuffle=True, random_state=seed)

# ---- RandomForest ----
rf_scores = cross_val_score(rf_def, X_def, y_def, cv=cv,
                            scoring="neg_root_mean_squared_error")
print("RandomForest CV RMSE:", -rf_scores.mean(), "±", rf_scores.std())

# ---- LinearRegression ----
lin_cv = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("lin", LinearRegression())
])
lin_scores = cross_val_score(lin_cv, X_def, y_def, cv=cv,
                             scoring="neg_root_mean_squared_error")
print("Linear CV RMSE:", -lin_scores.mean(), "±", lin_scores.std())

# ---- Gradient Boosting ----
gbr_cv = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=seed
)
gbr_scores = cross_val_score(gbr_cv, X_def, y_def, cv=cv,
                             scoring="neg_root_mean_squared_error")
print("GradientBoosting CV RMSE:", -gbr_scores.mean(), "±", gbr_scores.std())

# ---- Resumen comparativo ----
resumen_cv = pd.DataFrame({
    "Modelo": ["RandomForest", "LinearReg", "GBR"],
    "CV_RMSE_mean": [-rf_scores.mean(), -lin_scores.mean(), -gbr_scores.mean()],
    "CV_RMSE_std": [rf_scores.std(), lin_scores.std(), gbr_scores.std()]
})

print("\nResumen validación cruzada:")
print(resumen_cv.to_string(index=False))


# In[20]:


# Datos de CV
modelos = ["RandomForest", "GradientBoosting"]
rmse_mean = [47818.61, 50806.26]
rmse_std = [5956.35, 5028.09]

plt.figure(figsize=(7,5))
plt.bar(modelos, rmse_mean, yerr=rmse_std, capsize=7, color=["skyblue","lightgreen"])
plt.ylabel("RMSE (Cross-Validation)")
plt.title("Comparación de modelos (validación cruzada)")
plt.tight_layout()
plt.show()


# In[21]:


# nos aseguramos de usar el array completo, no solo la media
rf_cv_scores = -cross_val_score(rf_def, X_def, y_def, cv=5, scoring="neg_root_mean_squared_error")
gbr_cv_scores = -cross_val_score(gbr_def, X_def, y_def, cv=5, scoring="neg_root_mean_squared_error")

# Crear dataframe para seaborn
df_cv = pd.DataFrame({
    "RandomForest": rf_cv_scores,
    "GradientBoosting": gbr_cv_scores,
})

# Boxplot
plt.figure(figsize=(8,5))
df_cv.boxplot()
plt.ylabel("RMSE (validación cruzada)")
plt.title("Distribución de errores por modelo (CV)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# In[22]:


# Excluyendo la regresion lineal:
resumen_cv_plot = resumen_cv.query("Modelo != 'LinearReg'").copy()

labels = resumen_cv_plot["Modelo"].to_list()
means  = resumen_cv_plot["CV_RMSE_mean"].to_numpy()
errs   = resumen_cv_plot["CV_RMSE_std"].to_numpy()

x = np.arange(len(labels))

plt.figure(figsize=(7,5))
bars = plt.bar(x, means, yerr=errs, capsize=8)
plt.xticks(x, labels, rotation=0)
plt.ylabel("RMSE medio (validación cruzada)")
plt.title("Comparación de modelos con CV (media ± std)")
plt.tight_layout()
plt.show()


# In[23]:


# ========= Fig. 7 — NOMINAL vs DEFLACTADO→RE-INFLADO (GBR) =========
# Reconstruir la serie de IPV futura para los mismos trimestres que proj_gbr_cen
ult = df_ipv_hist["trimestre"].iloc[-1]
an, tr = int(ult[:4]), int(ult[-1])
ipv = float(df_ipv_hist["ipv"].iloc[-1])

fut = []
for _ in range(len(proj_gbr_cen)):
    tr = 1 if tr == 4 else tr + 1
    if tr == 1:
        an += 1
    ipv *= (1.0 + crecimiento_medio)
    fut.append((f"{an}Q{tr}", ipv))

df_ipv_fut_cmp = pd.DataFrame(fut, columns=["trimestre", "ipv"])

# Unir IPV futuro a la proyección nominal del GBR y calcular deflactado/re-inflado
cmp = proj_gbr_cen.merge(df_ipv_fut_cmp, on="trimestre", how="left")
cmp["pred_€2015"] = cmp["pred_price"] / (cmp["ipv"] / 100.0)   # deflactado (precio real 2015)
cmp["reinflado"]  = cmp["pred_€2015"] * (cmp["ipv"] / 100.0)   # vuelve al nominal (≈ pred_price)

# verificacion de que coinciden numéricamente
print("\nDiferencia nominal vs re-inflado:")
print((cmp["pred_price"] - cmp["reinflado"]).describe())

# Graficar
import matplotlib.ticker as mticker
plt.figure(figsize=(9,4))
plt.plot(cmp["trimestre"], cmp["pred_price"], marker="o", label="GBR nominal")
plt.plot(cmp["trimestre"], cmp["reinflado"], marker="s", linestyle="--", label="GBR deflactado→re-inflado")
plt.xticks(rotation=45)
plt.ylabel("€")
plt.title("Proyección: nominal vs deflactado (GBR)")
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))
plt.legend()
plt.tight_layout()
plt.savefig("proyeccion_nominal_vs_real.png")
plt.show()


# In[24]:


# ============================================
# 10) EXPORTS (modelo y resultados)
# ============================================
joblib.dump(rf, "modelo_rf_nominal.pkl")
joblib.dump(rf_def, "modelo_rf_deflactado.pkl")
proyeccion.to_csv("proyeccion_precio_trimestral_central.csv", index=False)

