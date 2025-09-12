import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from openpyxl import load_workbook

# Load Excel
file_path = "/content/Taiwan DataSheet(2).xlsx"
df = pd.read_excel(file_path, usecols=["Date", "Time", "Latitude", "Longitude", "Depth", "Mw"], engine="openpyxl")

# Keep valid magnitudes
df = df.dropna(subset=["Mw"]).copy()
df["Mw"] = pd.to_numeric(df["Mw"], errors="coerce")
df = df.dropna(subset=["Mw"])

# Round Mw to nearest 0.1
df["Mw_round"] = df["Mw"].round(1)

# Magnitude bins
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
slope, intercept, r_value, p_value, std_err = linregress(fit_data["Magnitude"], fit_data["log10(N)"])
a_value = intercept
b_value = -slope

print(f" fit: log10(N) = {a_value:.3f} - {b_value:.3f} * M")
print(f"R-squared = {r_value**2:.4f}")

# Plot
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

# Load workbook to find sheet
book = load_workbook(file_path)
sheet_name = book.active.title   # use active sheet (or specify manually)

# Write result DataFrame starting from column N
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
    result.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        startcol=6  # column N (0-based index)
    )

df = pd.read_excel(file_path, engine='openpyxl')

# 1) Small? column: 1 if Mw < 6, else 0
df["Small?"] = (df["Mw"] < 6).astype(int)

# 2) Large? column: 1 if Mw >= 6, else 0
df["Large?"] = (df["Mw"] >= 6).astype(int)

# 3) Cumulative small earthquakes (reset after large event)
cumulative_counts = []
count = 0
for small, large in zip(df["Small?"], df["Large?"]):
    if large == 1:
        count = 0
        cumulative_counts.append(count)
    else:
        if small == 1:
            count += 1
        cumulative_counts.append(count)

df["CumulativeSmall"] = cumulative_counts

df.to_excel(file_path, index=False)

# Make sure the column name matches exactly
values = df["CumulativeSmall"].tolist()

df["Cycle_peak"] = np.nan

for i in range(1, len(values)):
    # Detect reset (when value goes to 0 or 1 after being higher)
    if values[i] in (0, 1) and values[i-1] > values[i]:
        df.loc[i-1, "Cycle_peak"] = values[i-1]  # mark the peak row

df.to_excel(file_path, index=False)