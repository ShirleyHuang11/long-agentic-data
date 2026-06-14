# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Parallelism 

- For embarasinglly paralelizable tasks, use multiple Claude agents to complete those tasks.

- For writing mathematical proofs or math-heavy formulations, do literature reviews first to avoid hallucination or reinventing wheels.

- Keep the codebase tidy and neat, keep it simple and stupid. When executing new independent tasks, try to avoid reinventing wheels and don't build up technical debt.

## 5. Cluster

- use default $HF_HOME, etc folder, don't write large files into the $HOME and home directory ~. Please make sure don't save checkpoints or any large files into the home directory, if you need to save checkpoints or temporary files, please save them into the scratch folder like $SCRATCH.

- don't write large files into ~/.local and ~/.cache as well, if write temporary files, please clean them after using them.

- for short smoke-test runs, try to use the partition kempner_requeue. if the smoke tests are short jobs like 15min, use -t 30:00 can leverage backfill to be in the queue quickly.

- Try not to run jobs that would consume large memories on nodes like boslogin*.rc.fas.harvard.edu. For jobs that can be handled by CPUs, submit them to -p=seas_compute -A=barak_lab. For jobs that require GPUs, submit them to -p kempner (40 GiB A100) or -p kempner_h100 (80 GiB H100), then for kempner nodes use -A=kempner_sham_lab, choosing the appropriate queue based on the situation, especially try to reschedule the (QOSMaxGRESPerUser) to proper partitions.

- use uv to install environments - source $SCRATCH/envs/xxx/bin/activate. don't install hf_cache into my home dir, we have the HF_HOME dir to store ckpts. 

- there are requeue partitions - gpu_requeue and kempner_requeue, consider using those if they are very empty and the job is quick to finish. if the job takes >8 hours, use checkpointing so after preemption, the job can be resumed again. when partitions are full, try to flexibly find useful and efficient partition for the jobs at hand.

- For Kempner fairshare, try to coordinate the several (kempner_barak_lab, kempner_mzitnik_lab, kempner_sham_lab） and use less crowded account to avoid being at a long queue. For example, the kempner_barak_lab account itself has a low load and healthy fairshare, so when the sham_lab queue becomes heavily backed up later, moving more kempner / kempner_h100 jobs to -A kempner_barak_lab is a very cost-effective overflow channel.

- Account SEAS fairshare rotation (to maximize GPU job priority on seas_gpu/gpu_requeue): you hold several independent SEAS account shares — `barak_lab`, `sham_lab`, `chen_lab_seas`. Spreading load across all of them is the dominant lever (1→4 accounts ≈ 900x higher fairshare factor under heavy load). Fairshare decays with a 3-day half-life, so before each submit batch route to the current highest-fairshare account — greedy/self-balancing is optimal, prefer it over a fixed calendar. Use `~/acct.sh show` to rank, `~/acct.sh sbatch ...` to submit under the best, `~/acct.sh reroute` to move your PENDING jobs onto it. Never re-account RUNNING jobs (no speedup). If you must hardcode a cadence, rotate every 1–2 days and never let one account run hot past 3 days (it saturates and its factor collapses).

- if there is a large batch of jobs to be submit, try to amortize the cost to different partitions, use different gpus flexibly depending on the scale of the job duty.

- try to use max timeout (usually 3 days) for all jobs, don't specify a small amount of time for time-consuming jobs.

- Job priority rule in the FASRC cluster (queue wait, NOT runtime): `priority ≈ 1e6·age + 1e7·fairshare` (fairshare weight 10x age; QOS/partition/jobsize weights = 0; decay half-life 3d). So fairshare dominates PENDING→RUNNING wait; runtime speed depends only on hardware. Exception: `sched/backfill` lets short jobs with tight `-t` jump ahead into idle slots regardless of fairshare.


### 6. Results and Reports

- save important results in the ./results folder, git atomic commits if results are good

- for insightful results, save reports in the ./reports folder, make sure the claims are well-justified by the experiments.

- use a good amount of emojis to highlight takeaways and breakthroughs.

- when reporting experimental results in the chat, use Chinese and emoji to clearly convery takeways and insights. 

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

/loop 1h 检查assets/下面的proposal里面的科研想法, 模块化并行完成相关的多个子实验，把实验结果总结到results/*.md，要注意严谨独立，每次检查没完成的实验再提交对应的jobs，迭代优化使得结果能达到neurips paper的发表标准。如果有冗余的代码，重构并尽可能复用，有重要改进并且在实验结果work的情况下可以atomic commit。我希望找到phase diagram中一个edge of chaos的区间，在这个区间的数据有长程相关性，有最大的Maximum Effective Complexity，在这种数据上训练，可以使得模型泛化得更好 （最好包括更好的长度泛化能力，从短的序列训练；等KV retrieval的任务完成后，设计其他任务并训练更多的>100M transformers) 对于gpu,优先使用-p=kempner partition. 

/loop 30min 根据plan.md用最小代码改动实现这里面的计划，同时根据集群的availability提交job到kempner或者seas_gpu partition