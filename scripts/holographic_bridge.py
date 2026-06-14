"""Holographic bridge (steps 1-2): place real corpora on the holographic-data
(beta, gamma) phase diagram and import a predicted learnability axis.

KEY FINDING: the naive identity bridge gamma_hat = 2*beta*alpha FAILS (backwards):
our LZ-alpha tracks CONTENT (alpha<->H_inf = +0.78), so 2*beta*alpha makes
content-rich corpora look high-noise. The shared formula alpha = gamma/(2 beta) is
a coincidence of FORM: holographic gamma = noise-token rate, but our paper's "gamma"
in alpha_D=gamma/2beta is the LZ entropy-decay exponent. They are different gammas.

PRINCIPLED bridge: holographic gamma (noise rate) <-> our measured noise/boilerplate
rate scaffold_frac. Both are "fraction of non-informative tokens".

Empirical emergence boundary (holographic main_findings, natural log):
  gamma*(beta) = 0.274 + 0.265*ln(beta)        [emergent/learnable when gamma < gamma*]
Predicted learnability (their 2D fit, R^2=0.9999):
  train_acc(beta, gamma) = 0.254 + 0.0523*ln(beta) - 0.197*gamma
"""
import numpy as np, pandas as pd

b = pd.read_csv("data/gamma_beta.csv"); b = b.rename(columns={b.columns[0]: "slug"})
m = pd.read_csv("data/merged_analysis.csv")[["slug", "alpha", "h_inf", "bpc_32768", "role", "domain", "source"]]
sf = pd.read_csv("data/scaffold_frac.csv")
df = m.merge(b[["slug", "beta"]], on="slug").merge(sf, on="slug").dropna(subset=["alpha", "beta", "scaffold_frac"])
df = df[df["beta"] > 0].copy()
print(f"=== {len(df)} corpora with beta + scaffold + alpha ===\n")

# --- diagnostic: naive gamma = 2*beta*alpha (expected to fail) ---
df["gamma_naive"] = 2 * df["beta"] * df["alpha"]
print("[diagnostic] naive gamma=2*beta*alpha (conflates noise-gamma with LZ-alpha):")
print(f"  Spearman(gamma_naive, H_inf)    = {df.gamma_naive.corr(df.h_inf,'spearman'):+.2f}  (wrong sign: +)")
naive_learn = 0.254 + 0.0523*np.log(df.beta) - 0.197*df.gamma_naive
print(f"  Spearman(naive learnability, H_inf) = {naive_learn.corr(df.h_inf,'spearman'):+.2f}  (backwards)\n")

# --- principled: gamma = scaffold_frac (measured noise/boilerplate rate) ---
df["gamma"] = df["scaffold_frac"]
df["gamma_star"] = 0.274 + 0.265 * np.log(df["beta"])
df["margin"] = df["gamma_star"] - df["gamma"]          # >0 => emergent/learnable side
df["learnability"] = 0.254 + 0.0523 * np.log(df["beta"]) - 0.197 * df["gamma"]
df["region"] = np.where(df["margin"] > 0, "emergent", "chaos")

print("[principled] gamma = scaffold_frac (noise/boilerplate rate):")
print(f"  Spearman(gamma, H_inf)         = {df.gamma.corr(df.h_inf,'spearman'):+.2f}  (expect <0: noise vs content)")
print(f"  Spearman(learnability, H_inf)  = {df.learnability.corr(df.h_inf,'spearman'):+.2f}  (expect >0: healthy=learnable)")
print(f"  Spearman(margin, H_inf)        = {df.margin.corr(df.h_inf,'spearman'):+.2f}\n")

print(df[["slug", "beta", "gamma", "h_inf", "gamma_star", "margin", "learnability", "region"]]
      .round(3).sort_values("learnability", ascending=False).to_string(index=False))

df["health"] = np.where(df.h_inf == 0, "pooled", np.where(df.h_inf >= 0.6, "healthy", "mid"))
print("\n=== region x health ===")
print(pd.crosstab(df["region"], df["health"]))
df.to_csv("data/holographic_placement.csv", index=False)
print("\nwrote data/holographic_placement.csv")

# ---- figure (principled placement) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9.5, 6.8))
bgrid = np.linspace(max(df.beta.min(), 0.05), df.beta.max()*1.1, 200)
ax.plot(bgrid, 0.274 + 0.265*np.log(bgrid), "k--", lw=1.6,
        label=r"holographic boundary $\gamma^*(\beta)=0.274+0.265\ln\beta$")
ax.fill_between(bgrid, 0.274 + 0.265*np.log(bgrid), -1, alpha=0.06, color="tab:green")
sc = ax.scatter(df.beta, df.gamma, c=df.h_inf, cmap="viridis", vmin=0, vmax=2.5,
                s=95, edgecolors="k", linewidths=0.5, zorder=3)
for _, r in df.iterrows():
    ax.annotate(r.slug[:15], (r.beta, r.gamma), fontsize=6, xytext=(3, 3),
                textcoords="offset points", alpha=0.8)
fig.colorbar(sc, ax=ax, label=r"$H_\infty$ (measured content floor)")
ax.set_xlabel(r"$\beta$ (correlation-decay sharpness)")
ax.set_ylabel(r"$\gamma$ = scaffold_frac (measured noise/boilerplate rate)")
ax.set_ylim(-0.02, max(0.6, df.gamma.max()*1.1))
ax.set_title("Real agentic corpora on the holographic phase diagram (principled bridge)\n"
             "green = emergent/learnable ($\\gamma<\\gamma^*$); color = measured content $H_\\infty$")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig_holographic_bridge.png", dpi=160)
print("wrote figures/fig_holographic_bridge.png")
