# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:37:57 2025

@author: jvera
"""

import pandas as pd

# === 1. Cargar archivos ===
df_analisis = pd.read_csv("analisis_vecinos_embedding.csv")
df_familias = pd.read_csv("comunidades_familias_limpio.csv")

# === 2. Normalización para merge ===
df_familias["nodo_norm"] = df_familias["nodo"].str.replace("http://example.org/lang/", "", regex=False).str.strip()
df_analisis["entidad_norm"] = df_analisis["entidad"].str.strip()

# === 3. Unir info de familias ===
df_merged = pd.merge(
    df_analisis,
    df_familias[["nodo_norm", "familia"]],
    left_on="entidad_norm",
    right_on="nodo_norm",
    how="left"
)

# === 4. Filtrar lenguas con familia definida ===
df_con_familia = df_merged[df_merged["familia"].notna()].copy()

# === 5. Agrupar por comunidad y calcular métricas ===
df_grouped = df_con_familia.groupby("comunidad").agg({
    "ratio_intra/global": "mean",
    "overlap@k": "mean",
    "familia": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    "entidad": "count"
}).reset_index()

# === 6. Renombrar columnas para claridad ===
df_grouped = df_grouped.rename(columns={
    "entidad": "n_miembros",
    "familia": "familia_mas_frecuente",
    "ratio_intra/global": "prom_ratio",
    "overlap@k": "prom_overlap"
}).sort_values(by="n_miembros", ascending=False)

# === 7. Guardar CSV ===
df_grouped.to_csv("analisis_comunidades_embeddings_filtrado.csv", index=False)

# === 8. Mostrar resumen (opcional)
print(df_grouped.head())
