import time
from pathlib import Path
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random  # for shuffling results

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src")))

from dipd.explainer import DIP
from dipd.learners import EBM
from sklearn.datasets import make_regression

# Constants for dataset generation
N_SAMPLES = 1000
N_FEATURES = 30
N_TRIALS = 8
N_STEPS = 2


# Function to run a single trial for a given feature count
def run_trial(k):
    # Generate synthetic regression dataset for this trial
    X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=0.1)
    feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Sample random subset of features and split for decomposition
    features_k = list(np.random.choice(feature_names, k, replace=False))
    df_k = df[features_k + ["target"]]
    half = k // 2
    comb = (features_k[:half], features_k[half:])

    start = time.time()
    _ = DIP(df_k, "target", learner=EBM).get(comb, order=2)
    elapsed = time.time() - start
    print(f"k={k}, time={elapsed:.4f} seconds")
    return elapsed


def main():
    # Feature counts to test
    feature_counts = list(range(2, N_FEATURES + 1, N_STEPS))
    min_times = []
    avg_times = []
    max_times = []
    error_occurred = False

    print(f"Starting runtime benchmark for k={feature_counts}")
    print(f"Number of trials per feature count: {N_TRIALS}")

    # Write all raw results progressively to CSV
    csv_path = "runtime_vs_features.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_features", "trial", "time"])

        # Dummy warmup run for consistent timing
        print("Running dummy warmup ...")
        try:
            for i in range(2, 5):
                _ = run_trial(i)
        except Exception as e:
            print(f"Warmup run error: {e}")

        for k in feature_counts:
            # Run trials sequentially with fresh dataset each time
            times = []
            for trial in range(N_TRIALS):
                try:
                    elapsed = run_trial(k)
                    times.append(elapsed)
                    writer.writerow([k, trial, elapsed])  # Write each result immediately
                except Exception as e:
                    print(f"Error at k={k}, trial={trial}: {e}")
                    error_occurred = True
                    break
            if error_occurred:
                break
            min_times.append(min(times))
            avg_times.append(sum(times) / len(times))
            max_times.append(max(times))

    print(f"Raw results written progressively to {csv_path}")


if __name__ == "__main__":
    main()
