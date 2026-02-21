import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 0. Base setup aligned with paper settings
# ==========================================
# Use TinyStories dataset and GPT-2 architecture as in the paper
MODEL_NAME = "gpt2"
DATASET_NAME = "roneneldan/TinyStories"
NUM_SAMPLES = 1000  # subset of stories for statistical analysis
MAX_LAG_BETA = 100  # max lag for estimating beta and H
CONTEXT_LENGTHS = [64, 128, 256, 512]  # context horizons for gamma

print(f"Loading Model ({MODEL_NAME}) and Dataset ({DATASET_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
dataset = load_dataset(DATASET_NAME, split=f"train[:{NUM_SAMPLES}]")
text_corpus = " ".join(dataset['text'])

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
    
    # Initial guess: A=1.0, gamma=0.5, H_inf=min(entropies)
    popt, _ = curve_fit(entropy_decay_func, context_lengths, entropies, 
                        p0=(1.0, 0.5, min(entropies)), maxfev=5000)
    gamma_val = popt[1]
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

# Paper: C(r) ~ r^(-beta); physically beta = 2 - 2H
def compute_hurst_and_beta(text):
    print("\n--- Computing Hurst (H) and Beta (Correlation Decay) ---")
    # For speed, extract static embeddings by word/token chunks
    words = text.split()[:5000] 
    vocab_embeddings = model.transformer.wte.weight.detach().numpy()
    
    # Map text to a continuous embedding signal
    signal_embs = []
    for w in words:
        tok = tokenizer.encode(w)
        if tok:
            signal_embs.append(vocab_embeddings[tok[0]])
    signal_embs = np.array(signal_embs)
    
    # PCA to 1D stationary semantic fluctuation (fGN)
    pca = PCA(n_components=1)
    signal = pca.fit_transform(signal_embs).flatten()
    
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
if __name__ == "__main__":
    gamma, _ = compute_gamma(text_corpus, CONTEXT_LENGTHS)
    H, beta_direct, beta_from_hurst = compute_hurst_and_beta(text_corpus)
    
    # Paper prediction for scaling-law exponent alpha_D
    alpha_from_direct = gamma / (2 * beta_direct)
    alpha_from_hurst  = gamma / (2 * beta_from_hurst)
    
    print("\n" + "="*50)
    print(" 🎯 FINAL SCALING LAW PREDICTION (Cagnetta et al. 2026)")
    print("="*50)
    print(f"1. Dataset: TinyStories (Highly structured, logical language)")
    print(f"2. Information Decay (Gamma):  {gamma:.4f} (How fast context helps)")
    print(f"3. Correlation Decay (Beta):   {beta_from_hurst:.4f} (via Hurst) vs {beta_direct:.4f} (Direct)")
    print("-"*50)
    print(f"Predicted Scaling Exponent (alpha = gamma / 2*beta):")
    print(f" => Alpha (Using Hurst logic): {alpha_from_hurst:.4f}")
    print(f" => Alpha (Using Direct Fit):  {alpha_from_direct:.4f}")
    print("="*50)
    print("Interpretation: A higher alpha means steeper, more efficient scaling.")