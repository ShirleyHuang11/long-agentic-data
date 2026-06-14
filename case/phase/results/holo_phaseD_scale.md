# Phase D — 100M scale 确认(H1 在 scale 下)

holo_100m(RoPE, 101M), NTP, train L=256 → eval L=2048, N=3, 3000 steps, op_kind=perm。

| anchor | (β,γ) | train | L2048 | retention |
|---|---|---|---|---|
| Natural | (2.0,0.8) | 0.994 | 0.324 | **0.326 ±.065** |
| Abyss | (0.0,0.0) | 1.000 | 0.294 | 0.294 ±.013 |
| Edge | (0.05,0.05) | 1.000 | 0.267 | 0.267 ±.049 |
| CoT | (0.5,0.4) | 1.000 | 0.256 | 0.256 ±.017 |

## 🔑 takeaway
- 排序与 19M 一致:**Natural > Abyss > Edge > CoT**。
- 100M 整体 retention 更高(0.26–0.33 vs 19M 的 0.19–0.25)——**更大模型一致地提升长度泛化,但不产生混沌边缘优势**。
- **H1 全息预测在 100M scale 下同样被证伪**;局部/高γ的 Natural 区反而最好。
