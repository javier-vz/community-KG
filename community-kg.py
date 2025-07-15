# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 23:36:51 2025

@author: jvera
"""

from rdflib import Graph, URIRef
from rdflib.namespace import RDF
import networkx as nx
import pandas as pd
import community as community_louvain  # pip install python-louvain

# === 1. Cargar TTL ===
ttl_path = "grafo_ttl_hibrido.ttl"
g = Graph()
g.parse(ttl_path, format="turtle")

# === 2. Identificar lenguas ===
LANGUAGE_CLASS = URIRef("http://example.org/lang/Language")
lenguas_validas = {
    str(s) for s, p, o in g.triples((None, RDF.type, LANGUAGE_CLASS))
}

# === 3. Crear grafo limpio solo entre lenguas ===
G = nx.Graph()
for s, p, o in g:
    if isinstance(s, URIRef) and isinstance(o, URIRef):
        if str(s) in lenguas_validas and str(o) in lenguas_validas:
            G.add_edge(str(s), str(o), label=str(p))

print(f"Grafo limpio con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

# === 4. Comunidades Louvain ===
partition = community_louvain.best_partition(G)

# === 5. Calcular modularidad ===
modularity = nx.algorithms.community.modularity(
    G,
    [set(n for n, c in partition.items() if c == comm_id) for comm_id in set(partition.values())]
)
print(f"Modularidad del grafo: {modularity:.3f}")

# === 6. Extraer solo la primera familia lingüística ===
BELONGS_TO_FAMILY = URIRef("http://example.org/lang/belongsToFamily")
node_families = {}

for s, p, o in g.triples((None, BELONGS_TO_FAMILY, None)):
    s_str = str(s)
    if s_str not in node_families:
        node_families[s_str] = str(o)  # solo la primera familia registrada

# === 7. Crear DataFrame comparativo ===
rows = []
for node in G.nodes:
    comunidad = partition.get(node)
    familia = node_families.get(node)
    rows.append({"nodo": node, "comunidad": comunidad, "familia": familia})

df = pd.DataFrame(rows)
print("Muestra del DataFrame resultante:")
print(df.sample(10))

# === 8. Exportar CSV ===
df.to_csv("comunidades_familias_limpio.csv", index=False)
