"""Empirical backbone for a UNIFIED interpretation of the corpora that does NOT
use the train/eval (role) axis. Question: once role is dropped, what organizes
the metric space? Outputs:
  1. eta^2 of each metric explained by role vs domain vs source (which label
     actually structures the data?)
  2. metric correlation structure (which axes are redundant vs independent)
  3. PCA of the standardized metric space (the real latent axes + loadings)
  4. silhouette of role/domain/source partitions (does the data cluster by role?)
  5. k-means clusters x role cross-tab (do unsupervised groups track role?)
All on data/merged_analysis.csv. Hurst (n=25) reported separately, not in the
full-corpus PCA.
"""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/merged_analysis.csv")
df = df[df["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])].reset_index(drop=True)
print(f"active corpora (EXCLUDED dropped): n={len(df)}\n")
df["log_turns"] = np.log10(df["mean_turns"].clip(lower=1))
FEATS = ["alpha", "h_inf", "bpc_32768", "log_turns"]


def eta2(values, labels):
    """one-way eta^2 = between-group SS / total SS."""
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_bet = sum(len(g) * (g.mean() - grand) ** 2
                 for _, g in values.groupby(labels))
    return ss_bet / ss_tot

print("=== (1) eta^2: which label structures each metric? ===")
print(f"{'metric':12s} {'role':>7s} {'domain':>7s} {'source':>7s}")
for m in ["alpha", "h_inf", "bpc_32768", "log_turns"]:
    v = df[m]
    print(f"{m:12s} " + " ".join(
        f"{eta2(v, df[c]):7.3f}" for c in ["role", "domain", "source"]))

print("\n=== (2) metric correlations (Spearman) ===")
print(df[FEATS].corr(method="spearman").round(2).to_string())

print("\n=== (3) PCA of standardized metric space ===")
X = StandardScaler().fit_transform(df[FEATS])
pca = PCA().fit(X)
print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print(f"{'feat':12s}" + "".join(f" {'PC'+str(i+1):>7s}" for i in range(len(FEATS))))
for f, row in zip(FEATS, pca.components_.T):
    print(f"{f:12s}" + "".join(f" {v:7.2f}" for v in row))

print("\n=== (4) silhouette: does data cluster by each label? ===")
for c in ["role", "domain", "source"]:
    print(f"  {c:8s}: {silhouette_score(X, df[c]):+.3f}")
print("  (higher = label partitions the metric space; ~0 = it doesn't)")

print("\n=== (5) unsupervised k=4 clusters x role ===")
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
df["cluster"] = km.labels_
print(pd.crosstab(df["cluster"], df["role"]))
print("\ncluster medians (the real groups):")
print(df.groupby("cluster")[["alpha", "h_inf", "bpc_32768", "mean_turns"]]
      .median().round(2).to_string())
print("\ncluster dominant domain:")
for k, g in df.groupby("cluster"):
    print(f"  cluster {k}: " + ", ".join(
        f"{d}({n})" for d, n in g["domain"].value_counts().head(3).items()))

print("\n=== Hurst (n=25 subset) independence check ===")
h = df[df["hurst"].notna()]
print("Spearman(hurst, h_inf):", round(h["hurst"].corr(h["h_inf"], "spearman"), 2))
print("Spearman(hurst, alpha):", round(h["hurst"].corr(h["alpha"], "spearman"), 2))
