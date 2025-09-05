import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Loading excel
file_path = "TAIWAN FREQ-MAG4.0.xlsx"
df = pd.read_excel(file_path, usecols=["Date", "Time", "Latitude", "Longitude", "Depth", "Mw"])

# Keep valid magnitudes
df = df.dropna(subset=["Mw"]).copy()
df["Mw"] = pd.to_numeric(df["Mw"], errors="coerce")
df = df.dropna(subset=["Mw"])

# Round Mw to nearest 0.1 (for binning)
df["Mw_round"] = df["Mw"].round(1)

# Defining magnitude bins (from 4.0 up to maximum magnitude, step 0.1)
m_min = 4.0
m_max = float(np.floor(df["Mw_round"].max() * 10) / 10.0)
mag_grid = np.round(np.arange(m_min, m_max + 0.001, 0.1), 1)

counts = df["Mw_round"].value_counts().reindex(mag_grid, fill_value=0).sort_index()
cdf = counts.cumsum()
total = int(counts.sum())
n_ge = total - (cdf - counts)
log10_n = np.where(n_ge > 0, np.log10(n_ge), np.nan)

result = pd.DataFrame({
    "Magnitude": mag_grid,
    "count": counts.values,
    "cumulative_count": cdf.values,
    "N(>=M)": n_ge.values,
    "log10(N)": log10_n
})

# Drop NaNs for regression
fit_data = result.dropna(subset=["log10(N)"])

#Linear regression
slope, intercept, r_value, p_value, std_err = linregress(fit_data["Magnitude"], fit_data["log10(N)"])
a_value = intercept
b_value = -slope  # slope is negative, so b = -slope

print(f" fit: log10(N) = {a_value:.3f} - {b_value:.3f} * M")
print(f"R-squared = {r_value**2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(result["Magnitude"], result["log10(N)"], s=40, alpha=0.7, label="Observed data")

x_vals = np.linspace(result["Magnitude"].min(), result["Magnitude"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label=f"Fit: log10(N) = {a_value:.2f} - {b_value:.2f}M")

plt.xlabel("Magnitude (Mw)")
plt.ylabel("log10(N)")
plt.title("Frequency-Magnitude Distribution")
plt.grid(True)
plt.legend()
plt.show()
result.to_csv("TAIWAN_frequency_magnitude_processed.csv", index=False)