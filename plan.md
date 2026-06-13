 下一步策略 — 三个 priority

  排序基于 (paper impact) × (effort)。

  🥇 Priority 1: 等 anchor pilots 完成 + 启动 Mamba 实现

  为什么: 当前数据是 Transformer-only,但 paper 的 central claim 是架构相关的 boundary。Mamba 实验是把"有趣的 Transformer 相图"升级为"架构相关相图定律"的关键。

  动作:
  - T-1h: 等 seed 2 of Natural + Edge 落地 (~16:08 EDT) → N=2 验证
  - T-2h: CoT seed 1 (~16:24) + seed 2-3 of all 3 anchors (~17:00)
  - 趁等待写 case/phase/model_mamba.py:
    - clone model.py, 实现 MambaCausal 匹配 d_model=1024, layers=8 budget
    - 修改 phase_core.train_one_model 接 architecture 选项
    - 加 model_preset=mamba_100m
  - Effort: ~3-4h code + multi-hour Mamba training

  🥈 Priority 2: 新 task — Logical Folding (per CLAUDE.md "设计其他任务")

  为什么: CLAUDE.md 明确说 "等 KV retrieval 任务完成后,设计其他任务并训练更多 >100M transformers"。KV retrieval 数据已饱和(91 cells),now is the right time。

  动作:
  - 实现 data/logical_folding.py: 嵌套函数组合 f(g(h(i(j(x))))),带 (β, γ) 等价参数
  - Run 4-anchor 实验(同 proposal 推荐 β=2/γ=0.8, β=0.05/γ=0.05 等)
  - 测试: γ*(β) boundary 是否跨任务相同? — universality test
  - Effort: ~3h code + 3 jobs × 2.5h training

  🥉 Priority 3: 边界细化 — 低 β 区段 probe

  为什么: Result 9c flagged 在 β < 1 区域 fit 是 concave (linear-in-log-β breakdown)。提交几个 cells 在 β ∈ {0.1, 0.2, 0.3, 0.4} × γ ∈ {0.05, 0.10, 0.15} 收紧 fit 系数。

  动作:
  - 12 single cells × 1 seed × ~51min ≈ 10h compute
  - Wait time depends on queue
  - Effort: 5 min submit

  何时做: 等 anchor pilots 完成后再做(避免堆叠 PENDING)。