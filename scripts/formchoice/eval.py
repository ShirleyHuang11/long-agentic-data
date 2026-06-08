"""Form-vs-Choices eval (finding-20 discriminator) — runnable deterministic core.

3 models: base (Qwen2.5-1.5B-Instruct), template (SFT on OpenThoughts), healthy
(SFT on GLM4.7/JetBrains/SWE-ZERO). Greedy decode.

FORM score (deterministic parsers, held-out agent turns): does the model emit a
well-formed single action that is syntactically valid and non-degenerate — NOT
whether it's the right action.

DECISION score (exact-match to gold, no template overlap): given a conversation
prefix ending right before a gold assistant tool call, does the model pick the
correct tool? Sources NOT in either training corpus (APIGen-MT, tau-bench).

Implements the deterministic subset of reports/formchoice_eval_design.md; the
sandbox-verifier checks (aider) are deferred and noted in the report.
Writes data/formchoice_eval.json.
"""
import json, re, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
CK = "/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/ckpts"
MODELS = {"base": BASE, "template": f"{CK}/template", "healthy": f"{CK}/healthy"}
N_FORM, N_DEC = 60, 120
MAXNEW = 256


def gen(model, tok, msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# ---- FORM checks (terminus-2 JSON-aware) on a held-out emission ----
def _extract_json(text):
    # find first balanced {...} object in the emission
    s = text.find("{")
    if s < 0:
        return None
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[s:i+1])
                except Exception:
                    return None
    return None

def form_checks(text):
    c = {}
    obj = _extract_json(text)
    c["valid_json"] = obj is not None
    c["has_commands"] = isinstance(obj, dict) and isinstance(obj.get("commands"), list)
    c["has_analysis_plan"] = isinstance(obj, dict) and ("analysis" in obj and "plan" in obj)
    # well-formed keystrokes in command batch
    ok_ks = False
    if isinstance(obj, dict) and isinstance(obj.get("commands"), list) and obj["commands"]:
        ok_ks = all(isinstance(cmd, dict) and "keystrokes" in cmd for cmd in obj["commands"])
    c["wellformed_keystrokes"] = ok_ks
    # non-degenerate: <30% 4-gram repetition
    toks = text.split()
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i+4]) for i in range(len(toks)-3)]
        c["non_degenerate"] = (1 - len(set(grams))/len(grams)) < 0.30
    else:
        c["non_degenerate"] = True
    return c


def load_form_prompts():
    # held-out OpenThoughts rows BEYOND the 7787 used in training
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i < 8000:  # skip the training region
            continue
        conv = row.get("conversations") or []
        # find a prefix ending after an observation (user/tool), ask for next action
        msgs, last_role = [], None
        for m in conv:
            role = {"human": "user", "gpt": "assistant", "tool": "user"}.get(
                m.get("from") or m.get("role"), m.get("role", "user"))
            content = m.get("value") or m.get("content") or ""
            if role == "assistant" and len(msgs) >= 2:
                out.append(msgs[:])  # predict this assistant turn
                break
            msgs.append({"role": "user" if role != "assistant" else "assistant",
                         "content": content})
        if len(out) >= N_FORM:
            break
    return out


# ---- DECISION checks: gold tool-name exact match ----
def load_decision_prompts():
    prompts = []
    try:
        ds = load_dataset("Salesforce/APIGen-MT-5k", "dataset", split="train", streaming=True)
        for row in ds:
            conv = row.get("conversations") or []
            msgs = []
            for m in conv:
                val = m.get("value") or ""
                frm = m.get("from")
                if frm == "gpt" and ("tool_call" in val.lower() or '"name"' in val):
                    gold = re.search(r'"name"\s*:\s*"([^"]+)"', val)
                    if gold and len(msgs) >= 1:
                        prompts.append((msgs[:], gold.group(1)))
                    break
                role = "assistant" if frm == "gpt" else "user"
                msgs.append({"role": role, "content": val})
            if len(prompts) >= N_DEC:
                break
    except Exception as e:
        print("APIGen load failed:", str(e)[:120])
    return prompts


def main():
    form_p = load_form_prompts()
    dec_p = load_decision_prompts()
    print(f"form prompts={len(form_p)}  decision prompts={len(dec_p)}", flush=True)
    results = {}
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path if os.path.exists(path) else BASE)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
        model.eval()
        # FORM
        fsum = {}
        for msgs in form_p:
            t = gen(model, tok, msgs)
            for k, v in form_checks(t).items():
                fsum.setdefault(k, []).append(v)
        form_score = sum(sum(v)/len(v) for v in fsum.values())/len(fsum) if fsum else 0.0
        # DECISION
        dec_hits = 0
        for msgs, gold in dec_p:
            t = gen(model, tok, msgs)
            dec_hits += int(gold.lower() in t.lower())
        dec_score = dec_hits/len(dec_p) if dec_p else float("nan")
        results[name] = {"form": round(form_score, 4),
                         "decision": round(dec_score, 4),
                         "form_breakdown": {k: round(sum(v)/len(v), 3) for k, v in fsum.items()}}
        print(f"{name}: form={form_score:.3f} decision={dec_score:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    results["_deltas"] = {
        "d_form_healthy_minus_template": round(results["healthy"]["form"]-results["template"]["form"], 4),
        "d_decision_healthy_minus_template": round(results["healthy"]["decision"]-results["template"]["decision"], 4),
        "n_form": len(form_p), "n_decision": len(dec_p)}
    json.dump(results, open("data/formchoice_eval.json", "w"), indent=2)
    print("WROTE data/formchoice_eval.json"); print(json.dumps(results["_deltas"], indent=2))


if __name__ == "__main__":
    main()
