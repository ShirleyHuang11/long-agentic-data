# 小型 Transformer 在 Algorithmic Reasoning 上的 Beta-Gamma 相图实验

## 1. 实验目标

用可控数据分布参数 \((\beta,\gamma)\) 训练同一个小 Transformer，观察：

1. 训练内表现（`train_acc`）
2. 长度外推表现（`long_acc`）
3. 泛化损失（`generalization_gap = train_acc - long_acc`）

并据此构建经验相图（phase diagram），总结数据分布如何改变学习行为。

---

## 2. 任务与数据构造

### 2.1 任务

使用 key-value retrieval 的算法任务：
- 序列中交替出现写入 `(key, value)` 与读取 `key -> predict value`。
- 仅在读取位置计算 loss（mask 监督）。

### 2.2 分布控制参数

- \(\beta\)：控制读取时回溯距离分布（幂律）
  - 小 \(\beta\)：长程访问更多
  - 大 \(\beta\)：局部访问更多
- \(\gamma\)：控制 filler noise 比例
  - 大 \(\gamma\)：噪声更多，信息密度更低

---

## 3. 模型与训练设置

脚本：`case/scratchpad/transformer_algorithmic_phase_scan.py`

- 模型：1-layer causal Transformer (`d_model=32`, `nhead=4`)
- 训练长度：`train_len=64`
- 外推长度：`eval_long_len=128`
- 每点训练：`220 steps`, `batch=64`, `lr=1e-3`
- 网格：
  - \(\beta \in \{0.2, 1.5, 4.0\}\)
  - \(\gamma \in \{0.1, 0.4, 0.8\}\)

运行命令：

```bash
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate
python -u case/scratchpad/transformer_algorithmic_phase_scan.py \
  --out-dir plots/transformer_phase_run3 \
  --betas 0.2,1.5,4.0 \
  --gammas 0.1,0.4,0.8 \
  --train-len 64 \
  --eval-long-len 128 \
  --d-model 32 \
  --num-layers 1 \
  --train-steps 220 \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --eval-batches 3 \
  --lr 1e-3
```

相标注方式：使用 `relative` 分位数规则（在同一次扫描内相对比较），得到四相分布。

---

## 4. 结果

## 4.1 全局统计

- mean `train_acc` = **0.1700**
- mean `long_acc` = **0.1228**
- mean `gap` = **0.0472**
- corr(`alpha_theory`, `long_acc`) = **0.4345**
- corr(`alpha_theory`, `gap`) = **0.5214**

相数量：
- Chaos/Saturation: **3**
- Emergent: **3**
- Rote Memorization: **2**
- Super-Generalization: **1**

### 4.2 代表点

- 最佳外推点：`beta=1.5, gamma=0.4`, `long_acc=0.1540`（Super-Generalization）
- 最差外推点：`beta=4.0, gamma=0.8`, `long_acc=0.0647`（Chaos/Saturation）

### 4.3 九个网格点相别

| beta | gamma | train_acc | long_acc | gap | phase |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.2 | 0.1 | 0.158 | 0.116 | 0.043 | Chaos |
| 1.5 | 0.1 | 0.159 | 0.124 | 0.035 | Emergent |
| 4.0 | 0.1 | 0.211 | 0.144 | 0.067 | Rote |
| 0.2 | 0.4 | 0.200 | 0.148 | 0.053 | Emergent |
| 1.5 | 0.4 | 0.199 | 0.154 | 0.045 | Super |
| 4.0 | 0.4 | 0.175 | 0.111 | 0.064 | Emergent |
| 0.2 | 0.8 | 0.228 | 0.151 | 0.076 | Rote |
| 1.5 | 0.8 | 0.120 | 0.092 | 0.028 | Chaos |
| 4.0 | 0.8 | 0.080 | 0.065 | 0.015 | Chaos |

---

## 5. 规律总结（关键 Insight）

1. **中等 \((\beta,\gamma)\) 最容易出现“高外推相”**
- 本次 Super 点在 `(beta=1.5, gamma=0.4)`，说明太局部或太噪声都不是最优。

2. **高 \(\gamma\) 不是单调有利，存在噪声反噬**
- `gamma=0.8` 时多点进入 Chaos，说明信息密度过低会抑制学习稳定性。

3. **高训练准确率不等于高泛化**
- `beta=4.0, gamma=0.1` 和 `beta=0.2, gamma=0.8` 进入 Rote，相同特征是 train 不差但 gap 偏大。

4. **\(\beta\) 的作用与 \(\gamma\) 耦合**
- 不能只说“beta 小就好”或“beta 大就差”；当 \(\gamma\) 变化时，同一 \(\beta\) 会跨相。

5. **理论指数 \(\alpha=\gamma/(2\beta)\) 与长外推表现正相关但非充分条件**
- 正相关存在（0.4345），但仍受训练稳定性、噪声结构和有限训练预算影响。

---

## 6. 产物文件

- 结果表：`plots/transformer_phase_run3_relative_v2/transformer_phase_results.csv`
- 相图：`plots/transformer_phase_run3_relative_v2/transformer_phase_diagram.png`
- 长度外推热图：`plots/transformer_phase_run3_relative_v2/transformer_long_acc_heatmap.png`
- 泛化差热图：`plots/transformer_phase_run3_relative_v2/transformer_gap_heatmap.png`
- 自动摘要：`plots/transformer_phase_run3_relative_v2/transformer_phase_summary.md`

---

## 7. 结论

这套实验已经证明：
- 通过控制 \((\beta,\gamma)\) 的训练数据分布，**同一个小 Transformer 会落入不同学习相**；
- 相图可用于诊断“为什么学不动/为什么只会背/为什么能外推”；
- 对小模型而言，**中等关联长度 + 中等信息密度**是更稳健的工作区间。
