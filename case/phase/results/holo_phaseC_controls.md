# Phase C — 对照(H2 截断 / H3 Mamba)

## H2 — holographic-short vs truncated-long(matched token budget, N=3）
holo = Phase A（自包含短序列）; trunc = 长度-2048 程序的尾部窗口（含悬挂召回）。

| anchor | holo train/L2048/ret | trunc train/L2048/ret |
|---|---|---|
| Natural | 0.946/0.236/0.250 | 0.270/0.165/0.599 |
| CoT | 0.967/0.181/0.187 | 0.699/0.195/0.280 |
| Edge | 1.000/0.200/0.200 | 0.210/0.167/0.799 |
| Abyss | 1.000/0.247/0.247 | 0.175/0.140/0.799 |

**takeaway（H2 ✅ 部分支持）**：truncated 的高 retention（0.6–0.8）是**低分母伪信号**——
truncated 模型 train_acc 仅 0.18–0.70（学不会，因为切片有悬挂引用→ill-posed）。
按**绝对 L2048 acc**：holographic 3/4 锚点更高，且训练 acc 普遍 ~1.0。
→ holographic-short 比 truncated-long 更可学、长程绝对更好，但部分原因是 truncated 切片本身 ill-posed（结论偏弱）。

## H3 — Transformer vs Mamba（部分；Mamba scan 太慢，已改快版重投）
| anchor | Tx train/L2048/ret | Mamba train/L2048/ret |
|---|---|---|
| Natural | 0.946/0.236/0.250 | 0.870/0.267/0.307 |
（Mamba 在 Natural 略优于 Transformer，与本项目此前 N=3 发现一致；完整 4 锚点见快版重投结果。）
