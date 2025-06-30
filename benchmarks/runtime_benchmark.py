import time
from pathlib import Path
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))

from dipd.explainer import DIP
from dipd.learners import EBM
from sklearn.datasets import make_regression


def main():
    # Generate synthetic regression dataset
    N_SAMPLES = 1000
    N_FEATURES = 5
    X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    feature_counts = list(range(2, N_FEATURES + 1, 2))
    min_times = []
    avg_times = []
    max_times = []
    error_occurred = False
    for k in feature_counts:
        times = []
        for trial in range(5):
            # Sample random subset of features
            features_k = list(np.random.choice(feature_names, k, replace=False))
            df_k = df[features_k + ["target"]]

            # Split features into two groups for decomposition
            half = k // 2
            comb = (features_k[:half], features_k[half:])

            try:
                dip = DIP(df_k, "target", learner=EBM)
                start = time.time()
                _ = dip.get(comb, order=2)
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"k={k}, trial={trial}, time={elapsed:.4f} seconds")
            except Exception as e:
                print(f"Error at k={k}, trial={trial}: {e}")
                error_occurred = True
                break
        if error_occurred:
            break
        min_times.append(min(times))
        avg_times.append(sum(times) / len(times))
        max_times.append(max(times))

    # Plot runtime measurements
    # Truncate feature_counts to match collected timings
    counts = feature_counts[: len(avg_times)]
    plt.figure(figsize=(10, 6))
    plt.plot(counts, avg_times, marker="o", label="Average")
    plt.fill_between(counts, min_times, max_times, color="gray", alpha=0.3, label="Min/Max range")
    plt.xlabel("Number of Features")
    plt.ylabel("Runtime (seconds)")
    plt.title("DIP Runtime vs. Number of Features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("runtime_vs_features.png")
    # plt.show()


if __name__ == "__main__":
    main()
