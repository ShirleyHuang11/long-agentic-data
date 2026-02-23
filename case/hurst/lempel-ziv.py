import zlib
import numpy as np
from datasets import load_dataset
import random
import time

# ==========================================
# 1. Core LZ entropy detector (Raw Deflate–based)
# ==========================================
def get_lz_entropy_bpc(text_chunk):
    """Compute BPC (Bits Per Character) of text as a proxy for information entropy."""
    data = text_chunk.encode('utf-8')
    # Use wbits=-15 for Raw Deflate to avoid header overhead and measure pure entropy
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
    compressed = compressor.compress(data) + compressor.flush()
    # Return average bits per character (BPC)
    return (len(compressed) * 8) / len(text_chunk)

# ==========================================
# 2. Generalized scaling-law exponent estimator
# ==========================================
def estimate_alpha_via_lz(text_corpus, short_len=128, long_len=32000, num_samples=500):
    print(f"   -> Sampling {num_samples} short chunks (len {short_len}) and long chunks (len {long_len})...")
    
    # Measure micro entropy: C_short
    short_bpcs = []
    for _ in range(num_samples):
        idx = random.randint(0, len(text_corpus) - short_len - 1)
        chunk = text_corpus[idx : idx + short_len]
        short_bpcs.append(get_lz_entropy_bpc(chunk))
    c_short = np.mean(short_bpcs)

    # Measure macro entropy: C_long
    long_bpcs = []
    for _ in range(num_samples // 5):  # fewer long samples; stats are stable enough
        idx = random.randint(0, len(text_corpus) - long_len - 1)
        chunk = text_corpus[idx : idx + long_len]
        long_bpcs.append(get_lz_entropy_bpc(chunk))
    c_long = np.mean(long_bpcs)

    # Physical-quantity mapping
    beta_proxy = c_short                    # Micro friction (local unpredictability)
    gamma_proxy = c_short - c_long          # Macro gain (information gain from long context)
    
    # Cagnetta fast proxy formula
    alpha_proxy = gamma_proxy / (2 * beta_proxy)

    # Physical mapping: power-law exponent from slope in log-log (C vs N)
    ratio_C = c_long / c_short
    ratio_N = long_len / short_len
    alpha_true = abs(np.log(ratio_C) / np.log(ratio_N))

    return c_short, c_long, gamma_proxy, beta_proxy, alpha_proxy, alpha_true

# ==========================================
# 3. Dataset run and comparison
# ==========================================
DATASETS = [
    ("TinyStories", "roneneldan/TinyStories", None, "train"),
    ("WikiText-2", "Salesforce/wikitext", "wikitext-2-raw-v1", "train")
]

if __name__ == "__main__":
    print("🚀 Starting LZ complexity oracle (CPU-only)...\n")
    
    for display_name, path, subset, split in DATASETS:
        print("="*60)
        print(f"📚 Loading dataset: {display_name}")
        t0 = time.time()
        
        # Load ~5–10 MB of text for stable macro statistics
        if subset:
            ds = load_dataset(path, subset, split=f"{split}[:5000]")
        else:
            ds = load_dataset(path, split=f"{split}[:5000]")
            
        join_char = " <|endoftext|> " if "Tiny" in display_name else " "
        corpus = join_char.join([t for t in ds['text'] if len(t.strip()) > 20])
        
        # Ensure corpus is long enough
        corpus = corpus * 5 if len(corpus) < 100000 else corpus
        
        # Run LZ estimation
        c_short, c_long, gamma, beta, alpha_proxy, alpha_true = estimate_alpha_via_lz(corpus)
        
        t1 = time.time()
        print(f"⏱️ Elapsed: {t1 - t0:.2f} s (no GPU)")
        print("-" * 60)
        print(f"🔍 Measured quantities:")
        print(f"   • C_short (micro entropy / Beta):  {c_short:.3f} BPC (lower = simpler local grammar)")
        print(f"   • C_long  (macro entropy):         {c_long:.3f} BPC")
        print(f"   • Gamma   (long-range gain):       {gamma:.3f} BPC (larger drop = stronger long-range structure)")
        print(f"🎯 Predicted scaling exponent (Alpha proxy): {alpha_proxy:.4f}")
        print(f"🎯 Scaling exponent from log-log slope (Alpha true): {alpha_true:.4f}")
        print("="*60 + "\n")