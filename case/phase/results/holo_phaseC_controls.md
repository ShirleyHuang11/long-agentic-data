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

## H3 — Transformer vs Mamba：**INCONCLUSIVE（实现受限）** 🟡
纯-pytorch Mamba 序列扫描在 L≥1024 极慢:eval≤2048 跑 5.5h 仅 3/12;eval≤1024 跑 5h 仅 3/12。
完整 4-anchor×3 需 ~20h A100,且 Mamba 在 2000 steps 下**欠拟合**(Natural acc@256=0.621<1.0),
两次运行结论矛盾——故 H3 在本任务上不可靠,已取消(省 GPU)。

仅 Natural 锚点(两套指标互相矛盾):
| 指标 | Transformer | Mamba |
|---|---|---|
| eval@2048: train/L2048/ret | 0.946/0.236/0.250 | 0.870/0.267/0.307（Mamba 略优）|
| eval@1024: acc256/acc1024/ret | 0.946/0.424/0.448 | 0.621/0.245/0.395（Tx 优,但 Mamba 欠拟合）|

**verdict**:本任务 Tx-vs-Mamba 因 (a) 纯-pytorch scan 太慢、(b) Mamba 欠拟合 而**不可靠**。
架构对长度泛化的影响以本项目此前**干净的 N=3 KV 结果**(commit d371e5a:Mamba 2.0–2.6× retention)为准。
nested_monoid 上重做 H3 需 mamba_ssm CUDA kernel + 更多 steps。
