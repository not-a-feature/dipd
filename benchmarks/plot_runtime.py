import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load raw results CSV
csv_path = Path(__file__).parent / "runtime_vs_features.csv"
df = pd.read_csv(csv_path)

# Compute average times per feature count
df_avg = df.groupby("num_features")["time"].mean().reset_index()

# Compute min and max times per feature count
df_min_max = df.groupby("num_features")["time"].agg(["min", "max"]).reset_index()

# Prepare data for fitting
x = df_avg["num_features"].values
y = df_avg["time"].values

# Fit exponential model y = a * exp(b x)
# Only use positive y values
y_log = np.log(y)
b, log_a = np.polyfit(x, y_log, 1)
a = np.exp(log_a)
exp_model = lambda x: a * np.exp(b * x)

# Extrapolate up to 100 features
step = int(np.diff(sorted(df_avg["num_features"]))[0]) if len(df_avg) > 1 else 1
x_pred = np.arange(x.min(), 150, step)

y_exp_pred = exp_model(x_pred)

# Plot observed and fitted curves
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="black", label="Observed Avg Time")
plt.plot(x_pred, y_exp_pred, label="Exponential Fit", linestyle=":")

# Add min/max observed times as shaded regions
plt.fill_between(
    df_min_max["num_features"],
    df_min_max["min"],
    df_min_max["max"],
    color="gray",
    alpha=0.3,
    label="Observed Min/Max Range",
)

plt.xlabel("Number of Features")
plt.ylabel("Runtime (seconds)")
plt.title("DIP Runtime vs. Number of Features: Observed and Model Fits")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = Path(__file__).parent / "runtime_vs_features_model.png"
plt.savefig(plot_path)
plt.show()
