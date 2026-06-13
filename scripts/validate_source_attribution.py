"""Leave-one-corpus-out validation of the claim that H_inf attributes generator
SOURCE. Associational eta^2 (paper Sec 5.1) is NOT predictive validation; this
runs source as a held-out classification problem and reports BALANCED accuracy
(classes are imbalanced) + confusion matrix, against two baselines that the
H_inf signal must beat to count as real attribution:
  - majority-class prior
  - domain-only (does H_inf add anything beyond the domain confound?)
Feature sets: H_inf alone; H_inf + BPC@32K (the disambiguation companion).
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import sys

df = pd.read_csv("data/merged_analysis.csv")

# Coarse grouping: the medians say H_inf only supports human / frontier /
# degenerate, not 6 fine classes. Run with arg "coarse" to test that claim.
COARSE = {
    "human_task": "human", "human_demo": "human",
    "frontier": "frontier", "synth_task": "frontier",
    "distill": "degenerate", "mid": "degenerate",
}
if len(sys.argv) > 1 and sys.argv[1] == "coarse":
    df = df.assign(source=df["source"].map(COARSE))
    print("=== COARSE 3-class grouping (human / frontier / degenerate) ===")

y = df["source"].to_numpy()
classes = sorted(set(y))
n = len(df)


def loo_predict(feature_fn):
    """feature_fn(train_df, test_row) -> predicted source label."""
    preds = []
    for i in range(n):
        tr = df.drop(index=df.index[i])
        preds.append(feature_fn(tr, df.iloc[i]))
    return np.array(preds)


def gnb_features(cols):
    # uniform priors: test the FEATURE signal, not the class-frequency prior
    def fn(tr, row):
        cl = sorted(set(tr["source"]))
        clf = GaussianNB(priors=np.ones(len(cl)) / len(cl))
        clf.fit(tr[cols].to_numpy(), tr["source"].to_numpy())
        return clf.predict(row[cols].to_numpy().reshape(1, -1))[0]
    return fn


def majority(tr, row):
    return tr["source"].mode().iloc[0]


def domain_only(tr, row):
    sub = tr[tr["domain"] == row["domain"]]
    return (sub if len(sub) else tr)["source"].mode().iloc[0]


configs = {
    "majority prior":   majority,
    "domain-only":      domain_only,
    "H_inf alone":      gnb_features(["h_inf"]),
    "H_inf + BPC@32K":  gnb_features(["h_inf", "bpc_32768"]),
}

print(f"n={n} corpora, {len(classes)} sources: {classes}\n")
results = {}
for name, fn in configs.items():
    p = loo_predict(fn)
    bal = balanced_accuracy_score(y, p)
    acc = (p == y).mean()
    results[name] = (bal, acc, p)
    print(f"{name:18s}  balanced_acc={bal:.3f}  raw_acc={acc:.3f}")

print("\n--- Confusion matrix: H_inf + BPC@32K (rows=true, cols=pred) ---")
p = results["H_inf + BPC@32K"][2]
cm = confusion_matrix(y, p, labels=classes)
hdr = "true\\pred    " + " ".join(f"{c[:6]:>7s}" for c in classes)
print(hdr)
for c, row in zip(classes, cm):
    print(f"{c:12s}" + " ".join(f"{v:7d}" for v in row))

print("\n--- Per-class recall: H_inf + BPC@32K ---")
for c in classes:
    mask = y == c
    rec = (p[mask] == c).mean() if mask.sum() else float("nan")
    print(f"{c:12s} n={mask.sum():2d}  recall={rec:.2f}")
