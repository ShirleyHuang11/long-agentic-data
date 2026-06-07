"""TRL SFTTrainer for the form-vs-choices experiment (finding 20).

Trains Qwen2.5-1.5B-Instruct on a chat-format JSONL corpus (one
{"messages":[...]} per line). Two corpora are trained separately at matched
token budget: formchoice/template.jsonl vs formchoice/healthy.jsonl.

Model choice: Qwen2.5-1.5B-Instruct (not the base model). Rationale:
  - The Instruct model ships a chat template, so SFTTrainer applies per-turn
    role formatting directly from our {"messages"} rows with no hand-rolled
    template (which the base model lacks). Our two corpora are themselves
    terminus-2 / OpenHands chat traces.
  - Finding 20 is about *behavioral* shift (does the model learn FORM vs
    CHOICES). Starting from an instruction-following model isolates the
    agentic-trace contribution on top of a model that already follows
    instructions, which is the right control for the form/choices contrast.
  - 1.5B fits comfortably on one A100-40GB with packing + bf16 at seq 8192.

Usage:
  python train_sft.py --data <jsonl> --output_dir <dir> [--max_steps N] [--subset N]
"""
import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="chat JSONL ({'messages':[...]})")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="-1 = full epochs (default 1 epoch)")
    ap.add_argument("--subset", type=int, default=0,
                    help=">0 = take first N rows (smoke test)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.subset > 0:
        ds = ds.select(range(min(args.subset, len(ds))))
    print(f"Loaded {len(ds)} episodes from {args.data}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        packing=True,
        max_seq_length=args.seq_len,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
