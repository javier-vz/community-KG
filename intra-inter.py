# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 06:49:08 2025

@author: jvera
"""

import faiss
import numpy as np
import pickle
import pandas as pd

# === 1. Cargar FAISS index e ID map ===
index = faiss.read_index("index_faiss_hybrid.faiss")
with open("id_map_hybrid.pkl", "rb") as f:
    id_map = pickle.load(f)
uri_to_index = {v: i for i, v in enumerate(id_map)}

# === 2. Cargar CSV y limpiar URIs ===
df = pd.read_csv("comunidades_familias_limpio.csv")
df["lang_id"] = df["nodo"].str.extract(r"lang/([^/]+)$")
df_valid = df[df["lang_id"].isin(uri_to_index.keys()) & df["comunidad"].notnull()].copy()
df_valid["embedding_idx"] = df_valid["lang_id"].map(uri_to_index)

# === 3. Agrupar por comunidad ===
comunidad_grupos = df_valid.groupby("comunidad")["embedding_idx"].apply(list)

# === 4. Calcular distancias por comunidad ===
resultados = []
all_embeddings = index.reconstruct_n(0, index.ntotal)

for comunidad, indices in comunidad_grupos.items():
    if len(indices) < 2:
        continue

    # Intra-comunidad
    intra_dists = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            vi, vj = all_embeddings[indices[i]], all_embeddings[indices[j]]
            intra_dists.append(np.linalg.norm(vi - vj))

    # Inter-comunidad (muestra aleatoria de otras)
    otros = df_valid[df_valid["comunidad"] != comunidad]["embedding_idx"].tolist()
    muestra_otros = np.random.choice(otros, size=min(30, len(otros)), replace=False)
    inter_dists = []
    for i in indices:
        vi = all_embeddings[i]
        for j in muestra_otros:
            vj = all_embeddings[j]
            inter_dists.append(np.linalg.norm(vi - vj))

    resultados.append({
        "comunidad": comunidad,
        "n_miembros": len(indices),
        "intra_prom": np.mean(intra_dists),
        "inter_prom": np.mean(inter_dists),
        "relacion_inter/intra": np.mean(inter_dists) / np.mean(intra_dists) if np.mean(intra_dists) > 0 else np.nan
    })

# === 5. Tabla ordenada y exportada ===
df_resultados = pd.DataFrame(resultados).sort_values(by="n_miembros", ascending=False)
df_resultados.to_csv("distancias_intra_inter_por_comunidad.csv", index=False)
print(df_resultados.head(10))
