# 长程智能体 (Long-Horizon Agentic) 代表性数据集汇总 — 试点样本 (Pilot, iteration 1)

> **状态:试点 (pilot)。** 本文件是按 `data_format.md`(形式化数学数据集汇总,88 轮迭代)同款格式启动的**长程智能体数据**收集的第一批样本条目:6 个最常用的 long-horizon agentic benchmark / 轨迹语料,单 seed LZ oracle 评分。**请先审阅格式与口径,确认后再推全量收集**(候选清单见文末)。每一步收集与数据来源记录于 [`data/COLLECTION_LOG.md`](data/COLLECTION_LOG.md),逐数据集 provenance(HF revision SHA、采样参数、UTC 时间戳)见 `data/provenance/*.json`。

## TL;DR (试点版)

经 1 轮试点,覆盖 HuggingFace 上 **6 个公开 long-horizon agentic 数据集/配置**(SWE-agent 轨迹、AgentTuning/AgentGym 多环境轨迹、SWE-bench Verified、Mind2Web),并用 `scripts/lz_oracle.py` 的 LZ data oracle(3-point analytical estimation, n₁=128, n₂=2048, n₃=32768, zstd-19)给出每个的 `(α, H∞)` 评分。语料口径:每数据集 ≤1500 episodes 或 8 MB,一个 document = 一条完整轨迹(turn 渲染为 `[role]\ntext` 块)。

**试点期 3 个初步观察**(单 seed,待多 seed 验证,不作为结论):

1. **轨迹语料集体落在"低 α + H∞≈0"象限**(nebius SWE-agent / AgentInstruct / AgentTraj-L 的 H∞ 全部触底 0)—— 即 `data_format.md` cheat-sheet 里的 **Q4「模板退化/该去重」签名**(同款签名在形式化数学目录里解释了 33 个 prompt-spam 数据集)。机制显然:每条轨迹重复同一 system prompt + 脚手架样板(SWE-agent 的 file-viewer 每步重渲染 100 行窗口),32 KB chunk 内跨 episode 冗余被 LZ 全部吃掉。**含义:拿原始轨迹直接当训练语料,信息密度极低,curation(去重/剥脚手架)是第一优先级。**
2. **环境侧/任务侧文本远比智能体轨迹多样**:SWE-bench Verified(人写 GitHub issue + patch,H∞=1.57)和 Mind2Web 紧凑动作视图(人类标注网页任务,H∞=1.70)都在"多样核心"范围 —— 与发现 1 形成 **task-side vs agent-side 的干净对照**。
3. **同为 SWE 轨迹,格式选择影响一个量级**:SWE-smith(XML tool-call 格式)H∞=0.555,nebius SWE-agent(经典 ACI file-viewer 格式)H∞=0.00 —— 脚手架渲染方式本身就是数据质量变量。

**关键产物**:
- `SAMPLES.md` — 此文件(试点条目,待审)
- `data/agentic_alpha_hinf.csv` — 6 行评分表(由 provenance sidecar 重建,字段含 HF revision SHA)
- `data/COLLECTION_LOG.md` — 全部收集步骤 + 数据来源 + 事故记录(逐条)
- `data/provenance/<slug>.json` — 逐数据集机器可读 provenance
- `samples_cache/<slug>.txt` — 每数据集前 3 条序列化轨迹原文(oracle 实际输入)
- `scripts/lz_oracle.py` + `score_agentic_datasets.py` + `rebuild_csv.py`

---

## I. 试点代表数据集 (iteration 1, 单 seed)

| 数据集名称 | 环境/任务 | 数据类别 | 规模与 horizon | 核心特点与用途 | α (LZ-oracle) | H∞ (BPC) | HF 镜像 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **SWE-agent trajectories (Nebius)** | 真实 GitHub 仓库 issue 修复 | 轨迹 (SFT/拒采样) | 80,036 条;采样均值 **56.2 turns / 58 KB**·条 | **目前最大的公开 SWE-agent 轨迹释放之一**:含 model patch、exit status、eval logs,可按 `target` 字段筛成功轨迹。ACI file-viewer 脚手架,每步重渲染 → 极高样板冗余。 | 0.153 | **0.00** | `nebius/SWE-agent-trajectories` |
| **SWE-smith trajectories (xml)** | 合成 SWE 任务 (SWE-smith 工厂) | 轨迹 (SFT) | 26,076 条 (xml split;另有 tool 24k / ticks 26k);采样均值 **64.2 turns / 96 KB**·条 | 训练 **SWE-agent-LM-32B** 的轨迹语料;XML tool-call 格式。同类任务下 H∞ 显著高于 ACI 格式 → 格式即变量。 | 0.285 | 0.555 | `SWE-bench/SWE-smith-trajectories:xml` |
| **AgentInstruct (AgentTuning)** | OS / DB / ALFWorld / WebShop / Mind2Web / KG 六环境 | 轨迹 (SFT) | 1,866 条 (6 split 合并);采样均值 12.8 turns / 3.3 KB·条 | **多环境智能体 SFT 的开山数据**(AgentTuning, THUDM):GPT-4 轨迹 + ReAct 风格 CoT(Think/Act)。episode 短、六环境模板强 → H∞ 触底。 | 0.105 | **0.00** | `THUDM/AgentInstruct` |
| **AgentTraj-L (AgentGym)** | 14 环境 (ALFWorld/TextCraft/BabyAI/Sci-World/工具…) | 轨迹 (SFT/进化起点) | 14,485 条;采样均值 26.5 turns / 3.2 KB·条 | AgentGym 平台的扩展轨迹集,AgentEvol 自进化的 base 数据。**注意:首-N 采样在本试点全部落在 ALFWorld 段**(数据按环境排序),全量收集需按环境分片评分。 | 0.100 | **0.00** | `AgentGym/AgentTraj-L` |
| **SWE-bench Verified** | 真实 GitHub 仓库 (12 repo) | **评测基准** (任务侧文本) | 500 题 (人工核验);1 doc = issue+hints+patch+test_patch | **SWE 智能体行业金标准**(OpenAI 核验子集)。任务侧文本 = 人写 issue + 真实 patch → H∞=1.57,落"多样核心",与轨迹侧形成对照锚点。 | 0.334 | 1.567 | `princeton-nlp/SWE-bench_Verified` |
| **Mind2Web (action view)** | 137 真实网站 / 31 域 | **评测基准/训练** (动作轨迹紧凑视图) | train 1,009 任务 (全量 2,350);均值 7.7 步/任务 | 通用网页智能体基准。本行评分用 `confirmed_task + action_reprs` **紧凑动作视图**(原始 `cleaned_html` 观测 MB 级/步,2–3 条即占满 8 MB 语料,留待全量阶段单独评分)。 | 0.420 | 1.696 | `osunlp/Mind2Web` |

**口径备注**:α 越高 = BPC 随上下文长度下降越快(更强模板性/长程可压结构);H∞ = 外推到无穷上下文的不可压熵(多样性下限)。试点为**单 seed、首-N 流式采样**;发布质量需按模板协议补 5-seed σ。

### 样本轨迹摘录(oracle 实际输入,完整版见 `samples_cache/`)

```text
# nebius/SWE-agent-trajectories — 56 turns,角色交替 [system]/[user]/[assistant]
[system]
SETTING: You are an autonomous programmer, and you're working directly in the
command line with a special interface. The special interface consists of a file
editor that shows you 100 lines of a file at a time. ...
[user]
We're currently solving the following issue within our repository. ...
ISSUE: Memset provider: TypeError: string indices must be integers ...

# THUDM/AgentInstruct (os split) — ReAct 风格 Think/Act 轨迹
[human]
You are an assistant that will act like a person, I'will play the role of
linux(ubuntu) operating system. ... take exact one of the three actions:
"bash", "finish" or "answer". Think: put your thought here. Act: bash ...

# osunlp/Mind2Web — 紧凑动作视图
[task]
Find one-way flights from New York to Toronto.
[step 0] [combobox]  Flight type -> SELECT: One way
[step 1] [textbox]  Flying from -> TYPE: New York
...
```

## 试点诚实 caveats

1. **单 seed、无 shuffle 的首-N 采样** —— AgentTraj-L 已知受排序偏置(全落 ALFWorld);全量收集改为按 split/环境分片 + 5-seed。
2. **H∞=0 部分是协议产物**:3 KB 短 episode 下一个 32 KB chunk 跨 ~10 条共享 system prompt 的轨迹,跨 episode 冗余主导(对应模板 experiment-12 的边界密度效应)。全量阶段加两个受控变体:**剥 system-prompt 评分** 与 **per-episode chunk 评分**,把"脚手架冗余"与"内容冗余"分开。
3. **Mind2Web 评的是紧凑动作视图**,不含原始 HTML 观测,数值不可与全观测切片直接比较。
4. Oracle 为按 `data_format.md` 协议描述的重实现(原 `compute-free/hurst/lempel-ziv.py` 不在本仓库),绝对值可能有偏移,**表内相对排序自洽**;sanity check:随机可打印 ASCII → H∞=6.64≈log₂94 ✓,模板文本 → H∞→0 ✓。
5. 本表 6 条全部为**真实流式下载的公开数据**(HF Hub,采集时 revision SHA 已 pin 在 provenance sidecar),无任何合成/估算数值。

## 下一步:全量收集候选(待确认后执行)

| 候选 | 类型 | 备注 |
| :--- | :--- | :--- |
| `SWE-bench/SWE-smith-trajectories` 的 tool/ticks split、`SWE-Gym/*`、OpenHands 轨迹释放 | SWE 轨迹 | 同管线直接跑 |
| `gaia-benchmark/GAIA` | 通用助手基准 | **gated**,需 HF token + 接受条款 |
| WebArena / VisualWebArena、OSWorld、AgentNet | 环境/GUI | 无标准 HF 轨迹释放或多模态,需单独 schema 工作 |
| ToolBench、AgentBank、AgentOhana、tau-bench、AppWorld、TheAgentCompany | 工具调用/办公 | 逐个探 schema |
| AgentInstruct / AgentTraj-L **按环境分片** + 全部数据集 5-seed σ | 协议升级 | 对齐模板 iter-22 发布口径 |
| 剥 system-prompt / per-episode-chunk 受控变体 | 协议升级 | 分离脚手架冗余 vs 内容冗余 |
| 轨迹特有维度:tool-call 分布熵(对标模板的 tactic mode collapse)、成功 vs 失败轨迹 Δ(α,H∞)、benchmark↔训练轨迹污染探针 | 新实验线 | 模板第 64–199 行同款方法直接移植 |
