import faiss
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

K = 10

# === Cargar FAISS index e id_map ===
index = faiss.read_index("index_faiss_hybrid.faiss")
with open("id_map_hybrid.pkl", "rb") as f:
    id_map = pickle.load(f)

# === Cargar comunidades ===
df_comm = pd.read_csv("comunidades_familias_limpio.csv")

# === Normalizar URIs ===
def normalize_uri(uri):
    if isinstance(uri, str):
        return uri.replace("http://example.org/lang/", "").strip()
    return uri

df_comm["nodo_norm"] = df_comm["nodo"].apply(normalize_uri)
id_map_norm = [normalize_uri(uri) for uri in id_map]

uri_to_comm = dict(zip(df_comm["nodo_norm"], df_comm["comunidad"]))
uri_to_idx = {uri: idx for idx, uri in enumerate(id_map_norm)}
idx_to_uri = {idx: uri for uri, idx in uri_to_idx.items()}

# === Reconstruir vectores ===
vectors = index.reconstruct_n(0, index.ntotal)
vectors = np.array(vectors)

# === URIs válidas ===
valid_uris = [uri for uri in id_map_norm if uri in uri_to_comm and uri in uri_to_idx]

# === Métricas por entidad ===
results = []

for uri in tqdm(valid_uris, desc="Procesando entidades"):
    idx = uri_to_idx[uri]
    vec = vectors[idx].reshape(1, -1)
    comm_id = uri_to_comm[uri]

    # Vecinos globales
    D, I = index.search(vec, K + 1)
    I = I[0][1:]
    global_uris = [id_map_norm[i] for i in I]

    # Vecinos intra-comunidad
    same_comm_uris = [u for u in valid_uris if uri_to_comm[u] == comm_id and u != uri]
    same_comm_idxs = [uri_to_idx[u] for u in same_comm_uris]
    if len(same_comm_idxs) == 0:
        continue
    comm_vecs = vectors[same_comm_idxs]
    dists_comm = cosine_distances(vec, comm_vecs)[0]
    sorted_comm = np.argsort(dists_comm)[:K]
    intra_uris = [same_comm_uris[i] for i in sorted_comm]
    intra_dists = [dists_comm[i] for i in sorted_comm]

    global_vecs = vectors[[uri_to_idx[u] for u in global_uris]]
    global_dists = cosine_distances(vec, global_vecs)[0]

    # Métricas finales
    avg_intra = np.mean(intra_dists)
    avg_global = np.mean(global_dists)
    ratio = avg_intra / avg_global if avg_global > 0 else np.nan
    overlap = len(set(global_uris) & set(intra_uris)) / K

    results.append({
        "entidad": uri,
        "comunidad": comm_id,
        "avg_intra": avg_intra,
        "avg_global": avg_global,
        "ratio_intra/global": ratio,
        "overlap@k": overlap
    })

# === Exportar CSV ===
df_results = pd.DataFrame(results)
df_results.to_csv("analisis_vecinos_embedding.csv", index=False)
print("✅ Resultados guardados en analisis_vecinos_embedding.csv")
