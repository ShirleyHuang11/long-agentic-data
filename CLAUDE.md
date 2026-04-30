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

## 5. Cluster

- use default $HF_HOME, etc folder, don't write large files into the $HOME and home directory

- for short smoke-test runs, try to use the partition kempner_requeue. if the smoke tests are short jobs like 15min, use -t 30:00 can leverage backfill to be in the queue quickly.

- For jobs that can be handled by CPUs, submit them to -p=seas_compute -A=barak_lab. For jobs that require GPUs, submit them to -p kempner (40 GiB A100) or -p kempner_h100 (80 GiB H100), then for kempner nodes use -A=kempner_sham_lab, choosing the appropriate queue based on the situation.

- don't write large files into ~/.local and ~/.cache as well, if write temporary files, please clean them after using them.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

/loop 1h 检查assets/下面的proposal里面的科研想法, 模块化并行完成相关的多个子实验，把实验结果总结到results/*.md，要注意严谨独立，每次检查没完成的实验再提交对应的jobs，迭代优化使得结果能达到neurips paper的发表标准。如果有冗余的代码，重构并尽可能复用，有重要改进并且在实验结果work的情况下可以atomic commit。我希望找到phase diagram中一个edge of chaos的区间，在这个区间的数据有长程相关性，有最大的Maximum Effective Complexity，在这种数据上训练，可以使得模型泛化得更好 （最好包括更好的长度泛化能力，从短的序列训练；等KV retrieval的任务完成后，设计其他任务并训练更多的>100M transformers) 对于gpu,优先使用-p=kempner partition. 