# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:29:24 2025

@author: jvera
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual global
sns.set(style="whitegrid", font_scale=1.25)
plt.rcParams.update({'figure.autolayout': True})

# Cargar datos
df = pd.read_csv("analisis_comunidades_linguisticas.csv")

# ============================
# Plot 1: Inter/Intra vs. Num. families
# ============================
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="n_familias",
    y="relacion_inter/intra",
    size="n_lenguas",
    hue="n_lenguas",
    sizes=(40, 400),
    palette="viridis",
    legend="brief"
)
plt.title("Semantic Separation vs. Linguistic Diversity", fontsize=16)
plt.xlabel("Number of Language Families per Community")
plt.ylabel("Inter/Intra Embedding Distance Ratio")
plt.legend(title="Community Size (Languages)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("inter_intra_vs_families.png", dpi=300)
plt.show()

# ============================
# Plot 2: Intra vs. Geographic Spread (no legend)
# ============================
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="lat_std",
    y="intra_prom",
    size="n_lenguas",
    hue="macroareas",
    sizes=(40, 400),
    palette="tab10",
    legend=False  # Sin leyenda
)
plt.title("Geographic Spread vs. Intra-Community Coherence", fontsize=16)
plt.xlabel("Standard Deviation of Latitude")
plt.ylabel("Intra-Community Embedding Distance")
plt.tight_layout()
plt.savefig("intra_vs_latstd_nolegend.png", dpi=300)
plt.show()
