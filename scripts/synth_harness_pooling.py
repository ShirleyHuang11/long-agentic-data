"""Synthetic control for the harness-pooling claim (paper section 5.3/8).

Builds episodes = [shared boilerplate prefix] + [unique content], sweeps the
boilerplate fraction, and measures reference-exact H_inf vs directly-measured
BPC@32K. Prediction: as the shared prefix grows, cross-episode dedup steepens
the BPC(n) curve so the 3-point H_inf collapses toward (and clamps at) 0, while
BPC@32K stays positive and ordered -- i.e. heavy shared scaffold confounds H_inf
but not the directly measured companion. Uses only lz_oracle (zstd)."""
import random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle

rng = random.Random(0)
WORDS = [f"w{i:04d}" for i in range(2000)]  # vocabulary for "content"
N_EP = 400
EP_LEN = 6000  # chars per episode (boilerplate + content)
# one fixed boilerplate string shared by every episode (a fake system prompt)
BOILER = (" ".join(rng.choice(WORDS) for _ in range(4000)))

def content(n_chars):
    s = []
    tot = 0
    while tot < n_chars:
        w = rng.choice(WORDS) + " "
        s.append(w); tot += len(w)
    return "".join(s)[:n_chars]

print(f"{'boiler_frac':>11} {'alpha':>7} {'H_inf':>7} {'H_raw':>8} {'BPC@32K':>8}")
for frac in [0.0, 0.5, 0.8, 0.9, 0.95]:
    lb = int(EP_LEN * frac)
    lc = EP_LEN - lb
    prefix = BOILER[:lb]
    docs = [prefix + content(lc) for _ in range(N_EP)]
    s = lz_oracle.score(docs)
    print(f"{frac:>11.2f} {s['alpha']:>7.3f} {s['h_inf']:>7.3f} {s['h_inf_raw']:>8.3f} {s['bpc_32768']:>8.3f}")
