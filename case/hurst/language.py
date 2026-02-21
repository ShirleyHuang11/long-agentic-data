import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import requests
from tqdm import tqdm

"""
Real Text: Tiny Shakespeare
Calculated Hurst Exponent (H): 0.7863
"""

# Snowflake Arctic Embed M: https://huggingface.co/Snowflake/snowflake-arctic-embed-m
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-m"
CHUNK_SIZE = 128  # characters per chunk for sequence of embeddings
ENCODE_BATCH_SIZE = 16  # chunks per batch for progress bar

# 1. Corrected Hurst exponent computation (with physical integration step)
def get_hurst_exponent(time_series, max_lag=100):
    """Compute Hurst exponent by integrating the stationary signal first."""
    
    # Core physical correction: integrate stationary signal (fGN) into random-walk trajectory (fBM).
    # Subtract mean to remove linear drift, then take cumulative sum.
    trajectory = np.cumsum(time_series - np.mean(time_series))
    
    lags = range(2, max_lag)
    # Standard deviation of differences along the integrated trajectory
    tau = [np.std(np.subtract(trajectory[lag:], trajectory[:-lag])) for lag in lags]
    
    # Fit slope in log-log space
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    return m[0]

# 2. Fetch real text (Shakespeare)
def get_shakespeare_text():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    return response.text[:200_000]  # Use first 20k characters as sample

# 3. Extract feature sequence from text using Snowflake Arctic Embed
def extract_embedding_signal(text):
    print("Loading Snowflake Arctic Embed M and extracting embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Chunk text so we get a sequence of embeddings (one per chunk)
    chunks = [
        text[i : i + CHUNK_SIZE].strip()
        for i in range(0, len(text), CHUNK_SIZE)
        if text[i : i + CHUNK_SIZE].strip()
    ]
    if not chunks:
        raise ValueError("No non-empty chunks from text.")

    # Encode chunks as documents (no query prompt)
    all_embeddings = []
    for i in tqdm(
        range(0, len(chunks), ENCODE_BATCH_SIZE),
        desc="Encoding chunks",
        unit="batch",
    ):
        batch = chunks[i : i + ENCODE_BATCH_SIZE]
        emb = model.encode(batch)
        all_embeddings.append(np.asarray(emb, dtype=np.float64))
    embeddings = np.vstack(all_embeddings)

    # Reduce to 1-dim signal via PCA for Hurst analysis
    pca = PCA(n_components=1)
    signal = pca.fit_transform(embeddings).flatten()
    return signal

# Run experiment
if __name__ == "__main__":
    text = get_shakespeare_text()
    signal = extract_embedding_signal(text)
    h_real = get_hurst_exponent(signal)
    
    print(f"\n" + "="*30)
    print(f"Real Text: Tiny Shakespeare")
    print(f"Calculated Hurst Exponent (H): {h_real:.4f}")
    print("="*30)

    # Theoretical alignment
    if h_real > 0.6:
        print(f"Prediction: Natural Language is PERSISTENT (H={h_real:.2f}).")
        print("It should align with the BLUE line (H=0.8) in your Scaling Law plot.")
        print("Expected Scaling Exponent (alpha) is relatively HIGH.")
    else:
        print("Prediction: Natural Language behaves like Noise/Random Walk.")