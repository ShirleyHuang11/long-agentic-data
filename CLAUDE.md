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

- use default $HF_HOME, etc folder, don't write large files into the $HOME and home directory ~. Please make sure don't save checkpoints or any large files into the home directory, if you need to save checkpoints or temporary files, please save them into the scratch folder like $SCRATCH.

- don't write large files into ~/.local and ~/.cache as well, if write temporary files, please clean them after using them.

- for short smoke-test runs, try to use the partition kempner_requeue. if the smoke tests are short jobs like 15min, use -t 30:00 can leverage backfill to be in the queue quickly.

- Try not to run jobs that would consume large memories on nodes like boslogin*.rc.fas.harvard.edu. For jobs that can be handled by CPUs, submit them to -p=seas_compute -A=barak_lab. For jobs that require GPUs, submit them to -p kempner (40 GiB A100) or -p kempner_h100 (80 GiB H100), then for kempner nodes use -A=kempner_sham_lab, choosing the appropriate queue based on the situation.

- use uv to install environments - source $SCRATCH/envs/math/bin/activate. don't install envs and huggingface large files like hf_cache into my home dir, we have the HF_HOME dir to store ckpts. 

- there are requeue partitions - gpu_requeue and kempner_requeue, consider using those if they are very empty and the job is quick to finish. when partitions are full, try to flexibly find useful and efficient partition for the jobs at hand.

- if there is a large batch of jobs to be submit, try to amortize the cost to different partitions, use different gpus flexibly depending on the scale of the job duty.

- try to use max timeout (usually 3 days) for all jobs, don't specify a small amount of time for time-consuming jobs.


### 6. Results and Reports

- save important results in the ./results folder, git atomic commits if results are good

- for insightful results, save reports in the ./reports folder, make sure the claims are well-justified by the experiments.

- use a good amount of emojis to highlight takeaways and breakthroughs.

- when reporting experimental results in the chat, use Chinese and emoji to clearly convery takeways and insights. 

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
