import gc
import itertools
import os
import zlib
import numpy as np
from datasets import load_dataset
import random
import time

def get_lz_entropy_bpc(text_chunk):
    """Compute raw text entropy (BPC) using Raw Deflate."""
    data = text_chunk.encode('utf-8')
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
    compressed = compressor.compress(data) + compressor.flush()
    return (len(compressed) * 8) / len(text_chunk)

def estimate_alpha_via_lz(text_corpus, short_len=128, long_len=32000, num_samples=300):
    # Ensure corpus is long enough
    if len(text_corpus) < long_len * 2:
        text_corpus = text_corpus * (long_len * 2 // len(text_corpus) + 1)
        
    short_bpcs = []
    for _ in range(num_samples):
        idx = random.randint(0, len(text_corpus) - short_len - 1)
        short_bpcs.append(get_lz_entropy_bpc(text_corpus[idx : idx + short_len]))
    c_short = np.mean(short_bpcs)

    long_bpcs = []
    for _ in range(num_samples // 3):
        idx = random.randint(0, len(text_corpus) - long_len - 1)
        long_bpcs.append(get_lz_entropy_bpc(text_corpus[idx : idx + long_len]))
    c_long = np.mean(long_bpcs)

    # Compute log-log slope (scaling exponent)
    ratio_C = c_long / c_short
    ratio_N = long_len / short_len
    alpha_predict = abs(np.log(ratio_C) / np.log(ratio_N))

    return c_short, c_long, alpha_predict

# Benchmark: (name, HF ID, subset, split, joiner)
BENCHMARK_DATASETS = [
    ("BookCorpus (long-form literature)", "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English", None, "train", "\n\n"),
    ("TinyStories (minimal logic)", "roneneldan/TinyStories", None, "train", " <|eos|> "),
    ("WikiText-103 (high-quality wiki)", "Salesforce/wikitext", "wikitext-103-raw-v1", "train", " \n "),
    ("Python Code (structured code)", "transformersbook/codeparrot-train", None, "train", "\n<|endoffile|>\n"),
    ("OpenWebText (noisy web)", "Skylion007/openwebtext", None, "train", " <|endoftext|> "),
]

if __name__ == "__main__":
    print("🚀 Starting Data Physics Benchmark (LZ oracle)...\n")
    
    for name, path, subset, split, joiner in BENCHMARK_DATASETS:
        print("="*60)
        print(f"📚 Loading: {name} | {path}")
        t0 = time.time()
        
        try:
            text_key = 'code' if ('github-code' in path or 'codeparrot' in path) else 'text'
            text_key = 'context' if 'book' in path.lower() else text_key
            ds = load_dataset(path, subset, split=split, streaming=True)
            ds = ds.shuffle(buffer_size=10_000, seed=0)
            rows = itertools.islice(ds, 7000)
            valid_texts = []
            for r in rows:
                t = r.get(text_key) or r.get("text") or r.get("content")
                if t and len(t) > 50:
                    valid_texts.append(t)
                    if len(valid_texts) >= 5000:
                        break
            corpus = joiner.join(valid_texts)
            del ds, rows
            gc.collect()

            c_short, c_long, alpha = estimate_alpha_via_lz(corpus, short_len=128, long_len=32000)

            print(f"⏱️ Elapsed: {time.time() - t0:.2f} s")
            print(f"🔍 C_short (micro friction): {c_short:.3f} BPC")
            print(f"🔍 C_long  (macro entropy):  {c_long:.3f} BPC")
            print(f"🎯 Predicted Alpha (α):     {alpha:.4f}")
            
        except Exception as e:
            print(f"❌ Load or compute failed: {e}")
        print("="*60 + "\n")

    # Avoid PyGILState_Release abort at exit (pyarrow/datasets cleanup runs in wrong thread).
    gc.collect()
    os._exit(0)