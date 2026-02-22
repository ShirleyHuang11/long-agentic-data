import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 0. Base setup aligned with paper settings
# ==========================================
MODEL_NAME = "gpt2"
NUM_SAMPLES = 1000  # subset for statistical analysis
MAX_LAG_BETA = 100  # max lag for estimating beta and H

# Hurst signal: same embedding as language.py (Snowflake Arctic Embed M)
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-m"
CHUNK_SIZE = 128  # characters per chunk for sequence of embeddings
ENCODE_BATCH_SIZE = 16  # chunks per batch for progress bar

# Datasets: (display_name, path, subset_name or None, split, context_lengths for gamma)
# https://huggingface.co/datasets/Salesforce/wikitext
DATASET_CONFIGS = [
    # ("TinyStories", "roneneldan/TinyStories", None, f"train[:{NUM_SAMPLES}]", [64, 128, 256, 512]),
    ("WikiText-2", "Salesforce/wikitext", "wikitext-2-raw-v1", f"train[:{NUM_SAMPLES}]", [32, 64, 128, 256]), # 2, 4, 8, 16
]


def load_corpus(display_name, path, subset, split):
    """Load dataset and return single text corpus. TinyStories: join with EOS; WikiText: join with space."""
    if subset:
        ds = load_dataset(path, subset, split=split)
    else:
        ds = load_dataset(path, split=split)
    texts = ds["text"]
    if "TinyStories" in display_name:
        return " <|endoftext|> ".join(texts)
    return " ".join(texts)


print(f"Loading Model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ==========================================
# 1. Paper method A: Compute Gamma (conditional entropy decay with context)
# ==========================================
# Formula: H(n) - H_inf = A * n^(-gamma)
def compute_gamma(text, context_lengths):
    print("\n--- Computing Gamma (Entropy Decay) ---")
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    entropies = []
    
    # Simulate conditional entropy at each context length as in the paper
    model.eval()
    with torch.no_grad():
        for n in tqdm(context_lengths, desc="Evaluating Context Lengths"):
            losses = []
            # Sample positions to compute next-token loss at this context length
            for i in range(n, min(n + 500, len(tokens) - 1)):
                context = tokens[i-n : i].unsqueeze(0)
                target = tokens[i].unsqueeze(0)
                outputs = model(context, labels=context)
                # Approximate conditional entropy (last-token log-loss only)
                logits = outputs.logits[0, -1, :]
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
                losses.append(loss.item())
            entropies.append(np.mean(losses))
            
    # Fit Gamma
    def entropy_decay_func(n, A, gamma, H_inf):
        return A * np.power(n, -gamma) + H_inf
    
    try:
        # A must be positive; gamma typically in (0, 2); H_inf must not exceed the smallest observed entropy
        lower_bounds = [0.01, 0.01, 0.0]
        upper_bounds = [10.0, 2.0, min(entropies) + 0.1]
        popt, _ = curve_fit(entropy_decay_func, context_lengths, entropies, 
                            bounds=(lower_bounds, upper_bounds), maxfev=10000)
        gamma_val = popt[1]
    except Exception as e:
        print(f"Curve fit failed: {e}. Falling back to simple log-log regression.")
        # Fallback: log-linear regression
        H_inf_guess = min(entropies) * 0.95
        log_n = np.log(context_lengths)
        log_H = np.log([e - H_inf_guess for e in entropies])
        slope, _ = np.polyfit(log_n, log_H, 1)
        gamma_val = -slope

    print(f"-> Entropies over contexts: {[float(round(e, 3)) for e in entropies]}")
    print(f"-> Estimated Gamma: {gamma_val:.4f}")
    
    return gamma_val, entropies

# ==========================================
# 2. Continuous method B: Hurst exponent and Beta (correlation decay)
# ==========================================
# Hurst estimation aligned with language.py: integrate stationary signal (fGN) -> trajectory (fBM),
# then fit slope in log-log space; lags start at 2 to avoid boundary effects.
def get_hurst_exponent(time_series, max_lag=100):
    """Compute Hurst exponent by integrating the stationary signal first (same as language.py)."""
    trajectory = np.cumsum(time_series - np.mean(time_series))
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(trajectory[lag:], trajectory[:-lag])) for lag in lags]
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    return m[0]

# Extract 1D embedding signal using Snowflake Arctic Embed M (same as language.py)
def extract_embedding_signal(text):
    """Chunk text, encode with Snowflake Arctic Embed M, reduce to 1D via PCA."""
    print("Loading Snowflake Arctic Embed M and extracting embeddings for Hurst...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    chunks = [
        text[i : i + CHUNK_SIZE].strip()
        for i in range(0, len(text), CHUNK_SIZE)
        if text[i : i + CHUNK_SIZE].strip()
    ]
    if not chunks:
        raise ValueError("No non-empty chunks from text.")
    all_embeddings = []
    for i in tqdm(
        range(0, len(chunks), ENCODE_BATCH_SIZE),
        desc="Encoding chunks",
        unit="batch",
    ):
        batch = chunks[i : i + ENCODE_BATCH_SIZE]
        emb = emb_model.encode(batch)
        all_embeddings.append(np.asarray(emb, dtype=np.float64))
    embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=1)
    signal = pca.fit_transform(embeddings).flatten()
    return signal

# Paper: C(r) ~ r^(-beta); physically beta = 2 - 2H
def compute_hurst_and_beta(text):
    print("\n--- Computing Hurst (H) and Beta (Correlation Decay) ---")
    # Use Snowflake Arctic Embed M for embedding signal (same as language.py)
    signal = extract_embedding_signal(text)
    
    # 2.1 Compute Beta (direct fit to autocorrelation)
    lags_beta = np.arange(1, MAX_LAG_BETA)
    autocorr = [np.corrcoef(signal[:-lag], signal[lag:])[0, 1] for lag in lags_beta]
    
    def power_law(r, A, beta):
        return A * np.power(r, -beta)
    
    # Fit only the positive-correlation part
    valid_lags = [lags_beta[i] for i in range(len(lags_beta)) if autocorr[i] > 0]
    valid_corr = [autocorr[i] for i in range(len(lags_beta)) if autocorr[i] > 0]
    
    popt_beta, _ = curve_fit(power_law, valid_lags, valid_corr, p0=(1.0, 0.5))
    beta_direct = popt_beta[1]
    
    # beta_direct, beta_from_hurst = 0.0992, 0.9092
    
    # 2.2 Hurst via shared estimation (language.py style: integrate then log-log slope)
    hurst_val = get_hurst_exponent(signal, max_lag=MAX_LAG_BETA)
    
    # Theoretical relation
    beta_from_hurst = 2 - 2 * hurst_val
    
    print(f"-> Calculated Hurst Exponent (H): {hurst_val:.4f}")
    print(f"-> Beta (Derived from Hurst):     {beta_from_hurst:.4f}")
    print(f"-> Beta (Direct AutoCorr Fit):    {beta_direct:.4f}")
    
    return hurst_val, beta_direct, beta_from_hurst

# ==========================================
# 3. Results summary and theory alignment
# ==========================================
def run_and_report(display_name, text_corpus, context_lengths):
    """Run gamma + Hurst/beta on corpus and print report for this dataset."""
    gamma, _ = compute_gamma(text_corpus, context_lengths)
    H, beta_direct, beta_from_hurst = compute_hurst_and_beta(text_corpus)
    alpha_from_direct = gamma / (2 * beta_direct)
    alpha_from_hurst = gamma / (2 * beta_from_hurst)
    print("\n" + "=" * 50)
    print(f" 🎯 {display_name} — SCALING LAW PREDICTION (Cagnetta et al. 2026)")
    print("=" * 50)
    print(f"1. Dataset: {display_name}")
    print(f"2. Information Decay (Gamma):  {gamma:.4f}")
    print(f"3. Correlation Decay (Beta):   {beta_from_hurst:.4f} (via Hurst) vs {beta_direct:.4f} (Direct)")
    print("-" * 50)
    print(f"Predicted Scaling Exponent (alpha = gamma / 2*beta):")
    print(f" Alpha (from Cagnetta et al. 2026): {alpha_from_hurst:.4f}")
    print(f" => Alpha (Using Hurst logic):     {alpha_from_hurst:.4f}")
    print(f" => Alpha (Using Direct Fit):      {alpha_from_direct:.4f}")
    print("=" * 50)
    return gamma, H, beta_direct, beta_from_hurst, alpha_from_hurst, alpha_from_direct


if __name__ == "__main__":
    for display_name, path, subset, split, context_lengths in DATASET_CONFIGS:
        print(f"\n{'-'*60}\n# {display_name} ({path}), context_lengths={context_lengths}\n{'-'*60}")
        text_corpus = load_corpus(display_name, path, subset, split)
        run_and_report(display_name, text_corpus, context_lengths)
    print("\nInterpretation: A higher alpha means steeper, more efficient scaling.")