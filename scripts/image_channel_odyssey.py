"""Image-channel contrast (iter 45): GUI-Odyssey cross-app vs GIMP single-app.

Same metric as image_channel_pilot.py (mean consecutive-frame pixel delta on
64x36 grayscale + zstd ratio of the frame stack), on GUI-Odyssey episodes
whose screenshots live in shard data_0. Appends to data/image_channel_pilot.csv
with dataset="gui-odyssey".
"""

import collections
import csv
import os
import re

import numpy as np
import zstandard as zstd
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

REPO = "OpenGVLab/GUI-Odyssey"
N_EPISODES = 25
MIN_FRAMES = 6
SIZE = (64, 36)


def main():
    api = HfApi()
    # NB: an episode's frames are split ACROSS data_* shards — group over all.
    pngs = [s.rfilename for s in api.dataset_info(REPO).siblings
            if s.rfilename.startswith("screenshots/data_")
            and s.rfilename.endswith(".png")]
    eps = collections.defaultdict(list)
    pat = re.compile(r"data_\d+/(\d+)_(\d+)\.png$")
    for f in pngs:
        m = pat.search(f)
        if m:
            eps[m.group(1)].append((int(m.group(2)), f))
    chosen = [e for e in sorted(eps) if len(eps[e]) >= MIN_FRAMES][:N_EPISODES]
    print(f"episodes={len(eps)}, using {len(chosen)}", flush=True)

    cctx = zstd.ZstdCompressor(level=19)
    out = "data/image_channel_pilot.csv"
    deltas_all = []
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "task_id", "n_frames", "mean_step_delta",
            "frame_stack_zstd_ratio"])
        for ep in chosen:
            frames = []
            for _, fname in sorted(eps[ep]):
                try:
                    p = hf_hub_download(REPO, fname, repo_type="dataset")
                    g = np.asarray(Image.open(p).convert("L").resize(SIZE),
                                   dtype=np.float32)
                    frames.append(g)
                except Exception as e:
                    print(f"  skip {fname}: {type(e).__name__}", flush=True)
            if len(frames) < 2:
                continue
            deltas = [float(np.abs(b - a).mean() / 255.0)
                      for a, b in zip(frames, frames[1:])]
            stack = np.stack(frames).astype(np.uint8).tobytes()
            md = float(np.mean(deltas))
            deltas_all.append(md)
            w.writerow({"dataset": "gui-odyssey", "task_id": ep,
                        "n_frames": len(frames),
                        "mean_step_delta": round(md, 4),
                        "frame_stack_zstd_ratio":
                            round(len(cctx.compress(stack)) / len(stack), 4)})
            f.flush()
    print(f"episodes={len(deltas_all)}  mean delta={np.mean(deltas_all):.4f}  "
          f"median={np.median(deltas_all):.4f}", flush=True)


if __name__ == "__main__":
    main()
