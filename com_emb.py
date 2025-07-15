# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:11:28 2025

@author: jvera
"""

import pandas as pd

# === Cargar archivos ===
df_comm_fam = pd.read_csv("comunidades_familias_limpio.csv")
df_vecinos = pd.read_csv("analisis_vecinos_embedding.csv")

# === Normalizar URIs ===
df_comm_fam["nodo_norm"] = df_comm_fam["nodo"].str.replace("http://example.org/lang/", "").str.strip()
df_vecinos["entidad_norm"] = df_vecinos["entidad"].str.strip()

# === Merge ===
df_merged = pd.merge(
    df_vecinos,
    df_comm_fam[["nodo_norm", "familia"]],
    left_on="entidad_norm",
    right_on="nodo_norm",
    how="left"
)

# === Agrupación por comunidad ===
grouped = df_merged.groupby("comunidad").agg({
    "ratio_intra/global": "mean",
    "overlap@k": "mean",
    "familia": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    "entidad": "count"
}).reset_index().rename(columns={
    "entidad": "n_miembros",
    "familia": "familia_mas_frecuente",
    "ratio_intra/global": "prom_ratio",
    "overlap@k": "prom_overlap"
})

# === Ordenar por tamaño de comunidad ===
grouped = grouped.sort_values(by="n_miembros", ascending=False)

# === Guardar CSV ===
grouped.to_csv("analisis_comunidades_embeddings.csv", index=False)
print("✅ Resultados guardados en 'analisis_comunidades_embeddings.csv'")
