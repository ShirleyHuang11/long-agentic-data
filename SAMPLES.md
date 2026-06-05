# 长程智能体 (Long-Horizon Agentic) 代表性数据集汇总

逐轮迭代调研 HuggingFace 上的长程智能体轨迹语料 / 基准，模板沿用形式化数学综述 `data_format.md` 的 registry 格式，并用 `scripts/lz_oracle.py` 的 LZ data oracle 给出每个数据集的 `(α, H∞)` 评分。

**选集偏好：长 horizon、长 context 优先 —— 单 episode 越长（turns 多、bytes 大）越优先收录。** 每轮迭代只动一个 section，向其中追加 markdown 行即可。

α / H∞ 协议（3-point analytical estimation, n₁=128 / n₂=2048 / n₃=32768，zstd-19）：α 是 BPC ~ N^(−α) 关于上下文长度的标度指数（高 α = 模板性强 / 长程可压），H∞ 是外推到无穷上下文的不可压熵（BPC；低 H∞ = 模板退化信号）。每数据集采样 ≤1500 episodes 或 8 MB，一个 *document* = 一条完整 serialized 轨迹。脚本：`scripts/score_agentic_datasets.py`；逐数据集 provenance（pinned HF revision SHA、采样参数、原始分数）：`data/provenance/<slug>.json`；前 3 条 episode 原文：`samples_cache/<slug>.txt`。协议细节与诚实 caveats 见 `data/COLLECTION_LOG.md`。

---

## I. SWE / 代码智能体轨迹 (repo-scale, 最长 horizon)

| 数据集 | 类别 | 规模 | 轨迹长度 (turns / bytes·ep⁻¹) | 特点 | α | H∞ | HF 镜像 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SWE-rebench OpenHands 轨迹 | SWE 轨迹 | 数千条 | 127.9 / 195,763 | OpenHands 在 SWE-rebench 上的完整轨迹（含 tools schema、resolved 标注）；**turns 最多**（128/ep） | 0.294 | 0.67 | [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) |
| OpenHands feedback 会话 | 真实用户会话轨迹 | 数千条 | 61.8 / 205,677 | OpenHands 官方真实用户会话 dump（event-stream：action/content/extras，含正负反馈标注）；**bytes/ep 最长**（206 KB）；唯一的"野生"非基准轨迹源 | 0.361 | 0.75 | [`OpenHands/openhands-feedback`](https://huggingface.co/datasets/OpenHands/openhands-feedback) |
| SWE-Hero OpenHands 轨迹 | SWE 轨迹 | 数万条 | 125.0 / 141,052 | SWE-Zero 的姊妹 dump，horizon 第二长（125 turns / 141 KB·ep⁻¹） | 0.318 | 0.81 | [`nvidia/SWE-Hero-openhands-trajectories`](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) |
| SWE-smith-trajectories | SWE 轨迹 | 5k 条 | 64.2 / 96,001 | 训练 SWE-agent-LM-32B 用的轨迹（`xml` split）；H∞=0.56 介于模板退化与 NL 之间 | 0.285 | 0.56 | [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) |
| SWE-Zero OpenHands 轨迹 | SWE 轨迹 | 数万条 | 63.7 / 79,325 | NVIDIA 在 SWE-Fixer-Train-110K 任务上跑 OpenHands 的轨迹（带 license 字段）；H∞=1.21 全轨迹类最高 —— 模板退化最少 | 0.315 | 1.21 | [`nvidia/SWE-Zero-openhands-trajectories`](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) |
| OpenHands-SFT-Trajectories | SWE 轨迹 | 数百条（success.oss） | 39.0 / 62,786 | SWE-Gym 官方 SFT 轨迹（成功 episode 过滤）；H∞=1.11 健康 | 0.349 | 1.11 | [`SWE-Gym/OpenHands-SFT-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) |
| OpenHands-Sampled-Trajectories | SWE 轨迹（未过滤） | 数千条（train.raw） | 28.3 / 32,300 | SFT 版的未过滤原始采样（含失败 episode）；**与 SFT 版对照：成功过滤对 α/H∞ 几乎无影响**（Δα=0.015, ΔH∞=0.01），但成功 episode 更长（39.0 vs 28.3 turns） | 0.334 | 1.10 | [`SWE-Gym/OpenHands-Sampled-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) |
| OpenHands-Verifier-Trajectories | 验证器 (judge) 轨迹 | 数千条（train.mixture） | 3 / 63,167 | judge 对轨迹的评估对话（验证器训练数据），非 agent rollout 本体 | 0.339 | 1.07 | [`SWE-Gym/OpenHands-Verifier-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-Verifier-Trajectories) |
| SWE-agent-trajectories | SWE 轨迹 | 80k 条 | 56.2 / 58,315 | Nebius 用 SWE-agent 修真实 GitHub issue 的完整轨迹（含失败）；规模 × 长度乘积最大 | 0.153 | 0.00 | [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) |
| Nemotron-RL-SWE-Pivot | SWE RL（per-step） | 大规模（采样 534） | 90.0 / 15,749 | OpenHands 风格 per-*step* RL 数据（context messages + expected_action + pass_rate）；⚠️ 同 episode 多 step 行 → 前缀重叠 caveat | 0.207 | 0.59 | [`nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1) |
| R2E-Gym SWE-agent-LM 轨迹 | SWE 轨迹（自回滚） | 数百条 | 30.4 / 70,847 | SWE-agent-LM-32B 在 R2E-Gym 上的 rollout（thought/action 步级 + reward，131k ctx 跑）；**H∞≈0.01 —— 32B 自产 rollout 呈模板退化签名，与 frontier-model OpenHands dump（H∞ 0.6–1.2）形成"生成器规模"对照** | 0.232 | 0.01 | [`AxT-dev/swe-agent-lm-32b-r2e-gym-trajectories`](https://huggingface.co/datasets/AxT-dev/swe-agent-lm-32b-r2e-gym-trajectories) |

## II. Web / GUI 智能体

| 数据集 | 类别 | 规模 | 轨迹长度 (turns / bytes·ep⁻¹) | 特点 | α | H∞ | HF 镜像 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AgentTrek | Web/GUI 轨迹 | 10k+ 条 | 3 / 25,167 | tutorial-引导合成的 web 轨迹（messages 含页面观测）；turns 少但单 turn 观测长 | 0.337 | 0.85 | [`xlangai/AgentTrek`](https://huggingface.co/datasets/xlangai/AgentTrek) |
| NNetNav-WA | Web 轨迹（per-step） | 万级 step 样本 | 3 / 18,519 | 无监督探索（WebArena）；⚠️ 每行是 per-*step* SFT 样本（messages=该步上文），前缀重叠会抬高冗余 → H∞=0.46 偏低需照此解读 | 0.309 | 0.46 | [`stanfordnlp/nnetnav-wa`](https://huggingface.co/datasets/stanfordnlp/nnetnav-wa) |
| NNetNav-live | Web 轨迹（per-step） | 万级 step 样本 | 3 / 31,732 | nnetnav-wa 的真实网站版；同 per-step 前缀重叠 caveat，但 H∞=1.37 远高于 WA 版 —— 真实网页观测多样性 ≫ WebArena 模拟站 | 0.426 | 1.37 | [`stanfordnlp/nnetnav-live`](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) |
| WebLINX (action 视图) | Web 轨迹 | 2.3k demos / 155 站点 | 25.2 / 1,646 | 真人导航对话，episode=按 `demo` 聚合的 compact action 视图（raw `clean_html` ~100KB/turn 已排除）；**H∞=1.95 全 registry 最高** —— 真人行为语义密度最大 | 0.489 | 1.95 | [`McGill-NLP/weblinx`](https://huggingface.co/datasets/McGill-NLP/weblinx) |
| Mind2Web (action 视图) | Web 轨迹 | 2,350 任务 / 137 站点 | 7.7 / 423 | **compact action 视图**（task + action_reprs），非全观测 | 0.420 | 1.70 | [`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web) |
| Mind2Web (全观测切片) | Web 轨迹（full-obs） | 同上，129 eps 采样 | 7.9 / 65,038 | 每步 `cleaned_html`（37–240KB/步）截断至 8KB；**与 action 视图对照：H∞ 1.70→0.30** —— HTML 观测样板主导长程冗余 | 0.427 | 0.30 | 同上 |
| WebLINX (全观测切片) | Web 轨迹（full-obs） | 同 WebLINX，116 eps 采样 | 29.1 / 72,353 | pruned DOM（~1.6–4.3KB/turn）全量 + action；**与 action 视图对照：H∞ 1.95→0.00** —— 相邻 turn 页面几乎不变，观测近乎纯冗余 | 0.203 | 0.00 | 同上 |

## III. 多环境 / 工具使用轨迹

| 数据集 | 类别 | 规模 | 轨迹长度 (turns / bytes·ep⁻¹) | 特点 | α | H∞ | HF 镜像 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AgentTraj-L | 多环境 SFT 轨迹 | 14k 条 / 14 envs | 26.5 / 3,170 | AgentGym 的混合环境轨迹集；episode 短，H∞=0 提示模板退化 | 0.100 | 0.00 | [`AgentGym/AgentTraj-L`](https://huggingface.co/datasets/AgentGym/AgentTraj-L) |
| AgentInstruct | 多环境 SFT 轨迹 | 1.8k 条 / 6 envs | 12.8 / 3,290 | AgentTuning 的 GPT-4 轨迹（OS/DB/ALFWorld/WebShop/Mind2Web/KG）；短 episode，模板性强 | 0.105 | 0.00 | [`THUDM/AgentInstruct`](https://huggingface.co/datasets/THUDM/AgentInstruct) |
| CodeActInstruct | 代码-动作轨迹 | 7k 条（`codeact` split） | 8.3 / 4,808 | code-as-action 多轮轨迹（含交互执行反馈），任务域多样但 episode 偏短；H∞=0 模板退化签名 | 0.133 | 0.00 | [`xingyaoww/code-act`](https://huggingface.co/datasets/xingyaoww/code-act) |
| AgentBank | 多环境轨迹 | 50k 条 / 19 configs | 67.3 / 10,396 | 16 任务全家桶（alfred…webshop，19 configs 合并采样）；turns 多但 H∞=0.15 —— 环境观测高度模板化 | 0.482 | 0.15 | [`Solaris99/AgentBank`](https://huggingface.co/datasets/Solaris99/AgentBank) |
| Lumos (ground-iterative) | 多环境轨迹 | 1.5k+ 条采样 | 5.2 / 1,762 | 统一 planning/grounding 格式的子目标→动作标注；episode 极短，H∞=0 模板退化 | 0.211 | 0.00 | [`ai2lumos/lumos_unified_ground_iterative`](https://huggingface.co/datasets/ai2lumos/lumos_unified_ground_iterative) |
| tau-bench traces (jkazdan) | 工具调用轨迹 | ⚠️ 仅 50 条 | 27.6 / 7,453 | 非官方 tau-bench SFT traces（航司/零售客服多轮工具调用）；content=None 的 tool_calls 载荷被丢（近似文本视图）；n=50 低样本 + H∞=0 | 0.263 | 0.00 | [`jkazdan/taubench_traces_training_data`](https://huggingface.co/datasets/jkazdan/taubench_traces_training_data) |
| ToolBench ToolLLaMA-DFS | 工具调用轨迹 | 18k+ 条（非官方镜像） | 7.2 / 9,757 | ToolLLaMA 的 DFS 决策树轨迹（AutoGPT 风格多步 API 调用）；H∞=0 —— API 响应模板主导 | 0.192 | 0.00 | [`Yhyu13/ToolBench_toolllama_G123_dfs`](https://huggingface.co/datasets/Yhyu13/ToolBench_toolllama_G123_dfs) |
| Nemotron-SFT-Agentic-v2 (interactive) | 客服/政策工具对话 | 大规模（采样 1158） | 11.0 / 7,251 | NVIDIA Nemotron 合成 agentic SFT（policy-条件化客服 + 工具） | 0.167 | 0.26 | [`nvidia/Nemotron-SFT-Agentic-v2`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) |
| Nemotron-SFT-Agentic-v2 (search) | 搜索智能体 SFT | 大规模（采样 111） | 24.8 / 76,289 | 多跳搜索轨迹（raw JSONL 直读绕过 `datasets` schema 推断失败；bad_lines=0 —— 数据本身无损）；**H∞=1.37 与 nnetnav-live 并列轨迹类第二** | 0.261 | 1.37 | 同上 |
| Nemotron-SFT-Agentic-v2 (tool_calling) | 工具调用 SFT | 大规模（采样 1073） | 12.1 / 7,825 | 同上 JSONL 直读；落入模板集群（H∞=0），与 interactive split 同签名 | 0.157 | 0.00 | 同上 |
| Nemotron-RL-Conv-Tool-Pivot | 工具对话 RL（per-step） | 大规模（采样 1429） | 10.7 / 5,873 | per-step RL 数据（context+expected_action）；α=0.05 + H∞=0 —— 几乎纯模板（同一 scenario 重复采样 ×32 rollouts 的结构性冗余） | 0.048 | 0.00 | [`nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) |
| APIGen-MT-5k | 多轮工具调用轨迹 | 5k 条 | 15.2 / 5,747 | Salesforce 多轮 API 调用对话（tau-bench 风格航司/零售域）；H∞=0 模板退化 | 0.178 | 0.00 | [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) |
| Agent-FLAN | 多环境 SFT 轨迹 | 7 splits（采样 1500，首 split 主导） | 11.6 / 3,681 | InternLM 的 agent SFT 混合（AgentInstruct react/tflan + ToolBench 变体）；与 AgentInstruct/AgentTraj-L 同签名（低 α + H∞=0） | 0.113 | 0.00 | [`internlm/Agent-FLAN`](https://huggingface.co/datasets/internlm/Agent-FLAN) |
| ScienceWorld expert 轨迹 | 具身环境轨迹 | 580 条（test） | 45.9 / 8,680 | ⚠️ repo 名叫 webshop 但实为 **ScienceWorld** expert 轨迹（think_act agent，reward 标注）；turns 多但环境观测模板化（H∞=0） | 0.251 | 0.00 | [`lclan/webshop_expert_trajectories`](https://huggingface.co/datasets/lclan/webshop_expert_trajectories) |
| FireAct (multitask-multimethod) | QA+搜索工具轨迹 | 2k+ 条 | 5.1 / 930 | ReAct/CoT/Reflexion 混合方法的 QA 搜索轨迹；episode 极短但 **H∞=1.80 全 registry 第二** —— 内容多样性高、模板低 | 0.397 | 1.80 | [`zwhe99/FireAct`](https://huggingface.co/datasets/zwhe99/FireAct) |
| glaive-FC-v2 | 函数调用对话 | 113k 条 | 5.6 / 2,272 | 合成函数调用对话（raw text turns）；**短 FC 对话 H∞=1.05 反而高于 ToolLLaMA/APIGen 等轨迹型工具集（H∞=0）**—— 无重复环境观测 | 0.280 | 1.05 | [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |
| Hermes-FC-v1 (func_calling) | 函数调用对话（多轮） | 数千条 | 4.6 / 6,788 | NousResearch XML-tools 风格多轮 FC；同上，H∞=0.78 健康区间 | 0.259 | 0.78 | [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) |
| II-Agent GAIA 轨迹 | 多跳研究/工具轨迹 | 165 条（GAIA val 全集） | 46.6 / 20,984 | II-Agent 跑 GAIA validation 的完整 tool_call trace（含 judge 标注）；**绕开 GAIA gate 的轨迹代理**；H∞=1.25 健康 —— 多跳搜索内容多样 | 0.197 | 1.25 | [`Intelligent-Internet/ii-agent_gaia-benchmark_validation`](https://huggingface.co/datasets/Intelligent-Internet/ii-agent_gaia-benchmark_validation) |
| deep-research SFT (0406) | 深度研究轨迹 | 数百条 | 32.5 / 52,522 | 多源调研 SFT（长工具链 + 综合报告）；episode 长但 H∞=0 —— 蒸馏签名 | 0.179 | 0.00 | [`kylemontgomery/deep-research-sft-0406`](https://huggingface.co/datasets/kylemontgomery/deep-research-sft-0406) |
| Fractal DeepResearch-SFT | 深度研究报告 | 72 条采样 | 1 / 118,099 | plan+report 单文档（118 KB/ep，全 registry 第 4 长）；H∞=0 —— 合成报告模板主导 | 0.180 | 0.00 | [`FractalAIResearch/DeepResearch-SFT`](https://huggingface.co/datasets/FractalAIResearch/DeepResearch-SFT) |

## IV. 基准任务（非轨迹，issue/task 描述本体）

| 数据集 | 类别 | 规模 | 轨迹长度 (turns / bytes·ep⁻¹) | 特点 | α | H∞ | HF 镜像 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SWE-bench Verified | SWE 基准 | 500 题 | 1 / 6,225 | 人工核验过的 SWE-bench 子集；非轨迹（problem+patch），H∞=1.57 在 NL 正常区间 | 0.334 | 1.57 | [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |
| SWE-Gym | SWE 训练任务 | 2.4k 任务 | 1 / 14,005 | 可执行 repo 训练环境的任务集（problem+patch 序列化）；**轨迹版见候选队列 OpenHands-\*-Trajectories** | 0.395 | 1.22 | [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) |
| SWE-rebench (test) | SWE 基准 | 21k+ 任务（rolling） | 1 / 34,235 | SWE-bench 的 rolling 扩充（含 install_config/docker 镜像元数据）；单题文本最长的基准型条目 | 0.299 | 0.84 | [`nebius/SWE-rebench`](https://huggingface.co/datasets/nebius/SWE-rebench) |
| R2E-Gym-Lite | SWE 训练任务（合成 issue） | 数千任务 | 1 / 21,992 | 合成 problem_statement + commit hunks（whole-file 内容已排除，hunks 截 8KB）；**合成 issue H∞=0.89 < 人写 SWE-bench-Verified 1.57**，与 SWE-rebench 持平 | 0.381 | 0.89 | [`R2E-Gym/R2E-Gym-Lite`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Lite) |

---

## V. 总览速查（α × H∞ × horizon，迭代 18 收尾时点，n=42 有效 / CSV 46 行含 4 项已剔除）

### Horizon 排行（bytes·ep⁻¹ 前五）

| # | 数据集 | bytes·ep⁻¹ | turns | α | H∞ |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | OpenHands feedback（真实用户会话） | 205,677 | 61.8 | 0.361 | 0.75 |
| 2 | SWE-rebench OpenHands 轨迹 | 195,763 | **127.9** | 0.294 | 0.67 |
| 3 | SWE-Hero OpenHands 轨迹 | 141,052 | 125.0 | 0.318 | 0.81 |
| 4 | SWE-smith-trajectories | 96,001 | 64.2 | 0.285 | 0.56 |
| 5 | SWE-Zero OpenHands 轨迹 | 79,325 | 63.7 | 0.315 | **1.21** |

（Fractal DeepResearch 单文档报告 118 KB/ep 按字节可排 #4，但 turns=1 非轨迹，不入此榜。）

**长 horizon 即 SWE + 搜索：** bytes·ep⁻¹ 前 5 全部是 repo-scale 代码智能体轨迹；真·长轨迹（>20 turns 且 >50KB）存在于 SWE/OpenHands 生态、真实用户会话、以及 iter 17 抢救的 Nemotron-v2 search split（24.8 turns / 76 KB·ep⁻¹ / H∞=1.37）。

### 签名集群（α × H∞ 象限，对照 `data_format.md` §XIII）

| 集群 | 签名 | 成员 | 解读 |
| :--- | :--- | :--- | :--- |
| **健康长轨迹** | α 0.29–0.36, H∞ 0.6–1.2 | OpenHands 全家（rebench/Hero/Zero/SFT/Sampled/feedback）、SWE-smith | 长程结构 + 真实语义密度并存；**长 context 训练首选** |
| **蒸馏 SFT 模板退化** | α 0.10–0.26, **H∞=0** | AgentInstruct、AgentTraj-L、Agent-FLAN、CodeAct、ToolLLaMA-DFS、APIGen-MT、Lumos、ScienceWorld、tau-bench traces、nebius-SWE-agent | 环境观测/系统提示模板主导，外推到无穷上下文几乎全可压；与 math 综述"低α+H∞≈0 退化签名"一致 |
| **compact 高密度** | α 0.40–0.49, H∞ 1.7–1.95 | WebLINX/Mind2Web action 视图、FireAct | 人写/多样内容、observation 已剥离；短 episode 但语义密度全场最高 |
| **观测坍缩对照** | 同源 H∞ 1.7→0.3 / 1.95→0.0 | mind2web/weblinx full-obs vs action 视图 | **观测并入即 H∞ 坍缩** —— web 观测在长程上近乎纯模板冗余 |
| **拟合 artifact（已剔除）** | α<0 或 H∞>8 | Nemotron-v1、rlenv-appworld | 近恒等文档集上 3-point 拟合失效；CSV 保留原始分，registry 不收 |

### 累积发现

1. **成功过滤不改变压缩结构**（iter 3）：OpenHands Sampled（含失败）vs SFT（成功过滤）Δα=0.015、ΔH∞=0.01 —— 但成功 episode 平均更长（39.0 vs 28.3 turns）。
2. **真实 > 模拟**（iter 3）：NNetNav real-web H∞=1.37 vs WebArena 版 0.46 —— 真实网页观测多样性约 3×。
3. **观测主导长程冗余**（iter 6）：full-obs 切片 H∞ 坍缩（1.70→0.30、1.95→0.00）；compact/full-obs 不可直接比较。
4. **蒸馏 SFT 混合有统一签名**（iter 1–8）：所有"多环境 GPT-4 蒸馏轨迹混合"落在 α≈0.1–0.2、H∞=0 —— 与 OpenHands 真实 rollout（H∞ 0.6–1.2）一刀切开；H∞ 可做"真实 rollout vs 蒸馏模板"零成本探针。
5. **per-step 释出格式抬高冗余**（iter 2/7）：NNetNav、Nemotron-RL-Pivot 类 per-step 行共享前缀，α/H∞ 与 episode 级 dump 不可混排。
6. **生成器规模信号**（iter 10）：32B 自产 rollout（R2E-Gym SWE-agent-LM，H∞≈0.01）落入模板退化集群，frontier-model OpenHands dump 为 H∞ 0.6–1.2 —— 与 math 综述实验 14"高模板 = 机器生成探针"同向。
7. **合成 issue 语义密度折扣**（iter 11）：R2E-Gym 合成 problem_statement H∞=0.89 < 人写 SWE-bench-Verified 1.57。
8. **观测重复才是 H∞=0 的根因**（iter 12）：短 FC 对话（glaive 1.05 / hermes 0.78）H∞ 反而高于轨迹型工具集（ToolLLaMA/APIGen=0）—— 杀死 H∞ 的不是"工具调用"本身，而是逐 turn 重复的环境观测/模板。
9. **同域不同签名**（iter 15–17）：同为"深度研究/搜索"域，蒸馏报告型 SFT（deep-research-sft / Fractal）H∞=0，而真实多跳搜索轨迹（II-Agent GAIA 1.25、Nemotron-v2 search 1.37）H∞ 健康 —— 域不决定签名，生成管线决定。
10. **加载失败 ≠ 数据损坏**（iter 17）：Nemotron-v2 两 split 的 CastError/JSON 错全是 `datasets` schema 推断问题，raw JSONL 直读 bad_lines=0 全量可用 —— 弃用判定前先试绕过加载器。

## 候选队列（按预期 horizon 长度排序，每轮从顶部取 2–3 个）

| 候选 | 预期类别 | 为什么值得收 | 状态 |
| :--- | :--- | :--- | :--- |
| `nvidia/Nemotron-Agentic-v1` | agentic SFT | 评分产出拟合 artifact（α=−0.08, H∞=15.2，同 rlenv-appworld 一类）→ 不入 registry；v2 已覆盖同域 | ⚠️ 拟合 artifact 剔除 (iter 7) |
| `Salesforce/xlam-function-calling-60k` | 函数调用 | glaive/hermes 已入 §III；xlam 为 gated，需页面 request access | ⛔ gated (iter 12) |
| terminal-bench 轨迹源（待新候选） | 终端智能体轨迹（新类别） | `yoonholee/*` 30/30 行 steps=null、`harithoppil/*` pass config 20/20 行 response="" —— 两个 dump 都只有元数据无轨迹内容，弃；类别本身有价值，待真正带内容的释出 | ⚠️ 已检均为空壳 (iter 5) |
| `OS-Copilot/OS-Genesis-web-data` / `-mobile-data` | GUI 轨迹 | OS-Genesis 反向合成的 GUI 轨迹 | ⛔ token 已配但仍 gated —— 需在数据集页面点"request access" (iter 12) |
| `smolagents/aguvis-stage-2` (Aguvis) | GUI 轨迹（多模态） | 文本侧切片实测为拟合 artifact（α=−0.03/H∞=34.6：每行单 step + 巨型重复 system prompt）→ 不入 registry；真·多模态协议仍待设计 | ⚠️ 文本侧 artifact (iter 14) |
| `webarena-x/webarena-infinity-trajectories` | WebArena 轨迹 | 实测仅截图 PNG，无文本轨迹 | ⚠️ 弃 (iter 15) |
| `Intelligent-Internet/swebench-pro-*-ii-agent-trajectories` | SWE 轨迹（frontier 模型） | SWE-bench-Pro 上 Sonnet-4.5/GPT-5 轨迹 | ⛔ gated，需页面 request access (iter 15) |
| `OpenGVLab/GUI-Odyssey` (24k 下载) | GUI 轨迹（多模态） | 跨 app 导航轨迹 | 多模态扩展 |
| AppWorld 完整 rollouts | 交互轨迹 | `satyakic/appworld-rollouts-*` 是 event-sourcing 日志（重建复杂），弃；`hamishivi/rlenv-appworld-train` 实测是 90 条任务 prompt 非 rollout，且 3-point 拟合 artifact（α=−0.17, H∞=9.9）→ 不入 registry | ⚠️ 官方轨迹 dump 未找到 |
| tau-bench 官方/更大轨迹源 | 工具调用轨迹 | `sammshen/taubench-sonnet-traces` 实测整个 dump 是**单次 run 的代理日志**（聚合后仅 1 episode，29KB < 32KB oracle 块）→ oracle 不适用，不入 registry；`annon124816/tau_bench`（2.3k 下载）实测是 parquet 校验清单非数据 (iter 10)；jkazdan 50 条已入 §III，更大源仍缺 | ⚠️ 待更大轨迹源 |
| `SWE-Gym/MoatlessTools-Sampled-Trajectories` / `ZeonLap/OpenHands-Trajectories` | SWE 轨迹 | 前者仅 eval 元数据无轨迹内容；后者是 webdataset tar 日志（首行 log 为空 bytes） | ⚠️ 均弃 (iter 10) |
| TheAgentCompany | 交互基准 | 仅 OS 镜像 + 第三方 planning 数据，无官方轨迹 dump | ⚠️ 无合适 HF 释出 |
| `McGill-NLP/weblinx-browsergym` | Web 轨迹（BrowserGym 格式） | WebLINX 的 BrowserGym 重打包；认证后重试仍 ReadTimeout（文件树过大，metadata 列举本身超时）；compact/fullobs 两视图已入 registry，边际价值低 | ⚠️ 搁置（认证后仍超时，iter 12） |
| `Hongliang1997/OpenWebVoyager-IL-Trajectories-GPT-4o` | Web 轨迹 | repo 布局损坏（SplitsNotFoundError），无法流式加载 | ⚠️ 弃 (iter 8) |
| `gaia-benchmark/GAIA` | 多跳工具基准 | gated | ⛔ token 已配但仍 gated —— 需在数据集页面点"request access" (iter 12) |
| OSWorld / AgentNet | GUI 轨迹 | 多模态（截图），需 schema 工作 | 多模态扩展 |

## 迭代日志

| 轮 | 日期 (UTC) | 动作 |
| :- | :--- | :--- |
| 1 | 2026-06-05 | 建立 SAMPLES.md；落入 pilot 全部 6 集（补跑 swe-smith α=0.285/H∞=0.56/96KB·ep⁻¹ 为最长 horizon；mind2web-actions α=0.420/H∞=1.70）；从 provenance 重建 `data/agentic_alpha_hinf.csv`（6 行）；候选队列 6 个候选确认存在于 Hub |
| 2 | 2026-06-05 | +8 集：SWE-Gym/SWE-rebench 探明为任务集（→ §IV，α=0.395/0.299）；CodeActInstruct → §III；**OpenHands 轨迹 dump ×3 → §I（swe-rebench-oh 127.9 turns/196 KB·ep⁻¹ 新最长 horizon，α=0.294/H∞=0.67；SWE-Zero H∞=1.21 轨迹类最高）**；AgentTrek + NNetNav-WA → §II；hub search 定位 tau-bench/ToolBench/AppWorld 候选镜像；R2E-Gym 探明为任务生成集（序列化待设计） |
| 3 | 2026-06-05 | +7 集：SWE-Hero (125 turns/141 KB·ep⁻¹ horizon 第二) + OpenHands Sampled/Verifier → §I（**成功过滤 ablation：Δα=0.015，几乎无影响**）；NNetNav-live (H∞=1.37 ≫ WA 版 0.46，真实网页>模拟站) + **WebLINX (H∞=1.95 全 registry 最高)** → §II；AgentBank 19-config 合并 (67.3 turns 但 H∞=0.15 模板化) + Lumos → §III；脚本新增 multi-config 与 group-by-demo episode 聚合支持 |
| 4 | 2026-06-05 | 工具调用轮：tau-bench traces (jkazdan, n=50) → §III；**2 个候选实测后剔除**（taubench-sonnet-proxy 聚合后单 episode、rlenv-appworld 拟合 artifact α=−0.17）；ToolBench 两镜像弃（Adorg 加载失败 / Maurus 是检索语料非轨迹），改队列 `Yhyu13/ToolBench_toolllama_G123_dfs`；AppWorld 官方轨迹 dump 缺位确认 |
| 5 | 2026-06-05 | +2 集：ToolLLaMA-DFS (α=0.192/H∞=0，API 模板主导) → §III；**OpenHands feedback 真实用户会话 → §I（205,677 bytes/ep 全 registry 最长，61.8 turns，α=0.361/H∞=0.75；新增 event-stream 序列化器）**；terminal-bench 两 dump 深检均为空壳（steps=null / response=""）弃；OS-Genesis gated；Aguvis 多模态延期 |
| 6 | 2026-06-05 | 全观测协议轮：mind2web-fullobs（每步 HTML 截 8KB）+ weblinx-fullobs（pruned DOM 全量）→ §II；**核心发现：观测并入后 H∞ 崩塌（Mind2Web 1.70→0.30，WebLINX 1.95→0.00）—— web 观测在长程上近乎纯模板冗余，compact/full-obs 两视图测的是不同对象**；发现 NVIDIA Nemotron agentic 家族 + APIGen-MT-5k 入队 |
| 7 | 2026-06-05 | Nemotron 轮：+4 集（SFT-v2 interactive → §III；RL-SWE-Pivot 90 turns per-step → §I；RL-Conv-Tool-Pivot α=0.05/H∞=0 近纯模板 → §III；APIGen-MT-5k → §III）；3 项剔除/暂缺（v1 拟合 artifact；v2 search/tool_calling splits 上游 schema 错误加载失败）；新增 `ser_nemotron_rl` 序列化器 |
| 8 | 2026-06-05 | +3 集 → §III：Agent-FLAN（7 splits 合并，α=0.113/H∞=0 与 AgentInstruct 同签名）；ScienceWorld expert（repo 名误标 webshop，45.9 turns，H∞=0）；**FireAct（H∞=1.80 全 registry 第二，短 episode 高多样性）**；OpenWebVoyager-IL repo 损坏弃；weblinx-browsergym 超时待重试 |
| 9 | 2026-06-05 | 新增 §V 总览速查：horizon 排行（前 5 全为 SWE/OpenHands 生态）、α×H∞ 签名集群（健康长轨迹 / 蒸馏模板退化 / compact 高密度 / 观测坍缩 / artifact）、5 条累积发现（H∞ 可探"真实 rollout vs 蒸馏模板"）；weblinx-browsergym 重试探针发出 |
| 10 | 2026-06-05 | +1 集：R2E-Gym SWE-agent-LM 轨迹 → §I（30.4 turns/71KB，H∞≈0.01 —— **32B 自产 rollout 落入模板退化集群，frontier vs 小模型生成器对照支持发现 4**）；4 候选剔除 3（annon tau_bench=校验清单、MoatlessTools=纯元数据、ZeonLap=空 webdataset）；weblinx-browsergym 二次挂起搁置 |
| 11 | 2026-06-05 | 中途遇 HF 未认证限流（429，browsergym 文件树列举耗尽配额）退避一轮；恢复后 +1 集：R2E-Gym-Lite 任务集 → §IV（`ser_r2e_task`：problem+commit hunks，whole-file 排除；**合成 issue H∞=0.89 vs 人写 Verified 1.57**） |
| 12 | 2026-06-05 | HF token 配置（限流解除）；GAIA/OS-Genesis/xlam 仍 gated（需页面 request access）、browsergym 认证后仍超时 → 均搁置；+2 集 → §III：glaive-FC-v2 + Hermes-FC-v1（**短 FC 对话 H∞ 1.05/0.78 > 轨迹型工具集 0 —— 发现 8：杀死 H∞ 的是重复观测非工具调用本身**）；修复 SAMPLES.md CJK 硬换行渲染问题 |
| 13 | 2026-06-05 | 首次收尾：§V 刷新至 n=37 有效 / CSV 40 行（3 项已剔除）；累积发现 8 条 |
| 14 | 2026-06-05 | 续 5 轮（用户指示）：gated ×3 复查仍 blocked；Aguvis 文本侧切片实测拟合 artifact（单 step + 重复巨型 system prompt）剔除 |
| 15–16 | 2026-06-05 | 新类别扫描（computer-use / deep-research / browser-use）：+3 集 → §III（**II-Agent GAIA 轨迹 H∞=1.25 —— 绕 gate 的 GAIA 代理**；deep-research-sft 52KB/ep 与 Fractal 118KB/ep 均 H∞=0 蒸馏签名）；webarena-infinity 纯截图弃；ii-agent SWE-Pro gated |
| 17 | 2026-06-05 | **JSONL 直读抢救 Nemotron-v2 两 split**（加载失败原是 `datasets` schema 推断，数据无损）：search 24.8 turns/76KB/H∞=1.37 轨迹类第二；tool_calling 模板集群；发现 9/10 落档 |
| 18 | 2026-06-05 | **终轮收尾**：§V 刷新至 n=42 有效 / CSV 46 行（4 项剔除）；发现扩至 10 条；遗留：gated ×5（GAIA/OS-Genesis/xlam/ii-SWE-Pro ×2）等页面授权、GUI-Odyssey/OSWorld 多模态协议、5-seed σ 升级。**循环结束** |
