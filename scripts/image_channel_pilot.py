"""Image-channel pilot (iter 44): visual observation redundancy.

For multimodal GUI trajectories, measure how much the screen actually changes
between consecutive steps: downscale each screenshot to 64x36 grayscale and
compute the mean absolute pixel delta between consecutive frames, normalized
to [0,1]. Low delta = consecutive screenshots are near-duplicates (the visual
analog of the text observation-collapse, finding 3). Also report a zstd
compression ratio of the stacked downscaled frames as a crude visual-entropy
proxy.

Pilot on mlfoundations-cua-dev/agentnet-gimp-trajectories (embedded PNGs).
Appends per-episode rows to data/image_channel_pilot.csv.
"""

import csv
import os

import numpy as np
import zstandard as zstd
from datasets import load_dataset

N_EPISODES = 40
SIZE = (64, 36)  # w, h after downscale


def main():
    ds = load_dataset("mlfoundations-cua-dev/agentnet-gimp-trajectories",
                      split="train", streaming=True)
    out = "data/image_channel_pilot.csv"
    exists = os.path.exists(out)
    cctx = zstd.ZstdCompressor(level=19)
    deltas_all = []
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "task_id", "n_frames", "mean_step_delta",
            "frame_stack_zstd_ratio"])
        if not exists:
            w.writeheader()
        for i, row in enumerate(ds):
            if i >= N_EPISODES:
                break
            frames = []
            for img in row.get("images") or []:
                g = np.asarray(img.convert("L").resize(SIZE), dtype=np.float32)
                frames.append(g)
            if len(frames) < 2:
                continue
            deltas = [float(np.abs(b - a).mean() / 255.0)
                      for a, b in zip(frames, frames[1:])]
            stack = np.stack(frames).astype(np.uint8).tobytes()
            ratio = len(cctx.compress(stack)) / len(stack)
            md = float(np.mean(deltas))
            deltas_all.append(md)
            w.writerow({"dataset": "cua-agentnet-gimp", "task_id": row.get("task_id"),
                        "n_frames": len(frames),
                        "mean_step_delta": round(md, 4),
                        "frame_stack_zstd_ratio": round(ratio, 4)})
            f.flush()
    print(f"episodes={len(deltas_all)}  "
          f"mean step-delta={np.mean(deltas_all):.4f}  "
          f"median={np.median(deltas_all):.4f}", flush=True)


if __name__ == "__main__":
    main()
