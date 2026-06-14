# Phase A — 锚点 retention 排序(H1 方向性判定)

标准 RoPE decoder-only Transformer，holo_small(19M），NTP，train L=256，eval L∈{256,512,1024,2048}，N=3，2000 steps，op_kind=perm。retention = acc(2048)/acc(256)（answer-masked）。

| anchor | (β,γ) | train acc | L2048 acc | retention |
|---|---|---|---|---|
| Natural | (2.0, 0.8) | 0.946 | 0.236 | **0.250 ±.006** |
| Abyss | (0.0, 0.0) | 1.000 | 0.247 | **0.247 ±.009** |
| Edge | (0.05, 0.05) | 1.000 | 0.200 | **0.200 ±.003** |
| CoT | (0.5, 0.4) | 0.967 | 0.181 | **0.187 ±.007** |

每长度 acc 曲线：
- Natural: 0.946 / 0.762 / 0.424 / 0.236
- Abyss:   1.000 / 0.789 / 0.456 / 0.247
- Edge:    1.000 / 0.714 / 0.380 / 0.200
- CoT:     0.967 / 0.627 / 0.331 / 0.181

## 🔑 takeaway
⚠️ **不支持 holo.pdf 的"混沌边缘=最佳长度泛化"预测**：Edge 仅中游，Natural/Abyss 最好，CoT 最差。
retention 整体较平（0.19–0.25），但 std 很紧、差异真实（Natural 0.250 vs CoT 0.187 ≈ 9σ）。
机制：vanilla Transformer 在局部依赖（大β）下长度外推更稳；小β 长程召回在测试长度上更难。
⏳ 完整 6×6 网格热力图将定论是否存在 ridge / 平坦相图。
