# Holographic nested_monoid — knob verification (§3.4 gate)

任务：affine/perm 寄存器机 + 命名算子召回（`DEF name idx` / `USE name → result`）。
β = 召回距离（Zipf recency），γ = filler 比例。

## Gate 结果 ✅ PASS

**β-axis (γ=0.2 固定)** — 实测中位召回 token 距离随 β 单调下降：

| β | median recall lag |
|---|---|
| 0.05 | 24 |
| 0.50 | 12 |
| 2.00 | 7 |

**γ-axis (β=0.5 固定)** — filler 比例随 γ 单调上升：

| γ | filler fraction |
|---|---|
| 0.05 | 0.016 |
| 0.40 | 0.163 |
| 0.80 | 0.527 |

> ⚠️ 关键修复：原始设计未在序列中编码"应用了哪个算子"，导致 next-token 目标欠定 + 无可测长程相关（`‖C(n)‖≈0`）。改为 KV 式 **召回 cue**（`USE name` 重复 name token）后 task well-posed，且 β 真正控制长程召回距离。详见 BitLesson BL-20260613-recall-cue / BL-20260613-gate-instrument。

> 📌 算子选择：默认 `op_kind=perm`（固定置换池查表，可学习），而非 affine（模运算 grokking 困难）。
