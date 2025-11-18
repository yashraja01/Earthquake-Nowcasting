import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/content/Taiwan Datasheet 3.5.xlsx"

# Load Excel
df = pd.read_excel(file_path)

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract year
df['year'] = df['Date'].dt.year

# Filter year range if needed
df = df[(df['year'] >= 1970) & (df['year'] <= 2030)]

# Split magnitudes
df_big = df[df['Mw'] >= 6.0]
df_small = df[df['Mw'] < 6.0]

plt.figure(figsize=(12, 6))

# Scatter for small magnitudes
plt.scatter(df_small['year'], df_small['Mw'], s=5, color='navy')

# Scatter for big magnitudes
plt.scatter(df_big['year'], df_big['Mw'], s=40, edgecolors='black', color='yellow')

# Axis limits
plt.xlim(1970, 2030)
plt.ylim(3, 8)

# X-axis ticks = each 10 years → compressed like your screenshot
plt.xticks(range(1970, 2031, 10))

# Y-axis ticks = each magnitude
plt.yticks(range(3, 9, 1))

# Grid
plt.grid(which='major', linestyle='--', alpha=0.7)

plt.xlabel("Time (Years)")
plt.ylabel("Magnitude")
plt.title("Earthquake Magnitudes Over Time (1970–2030)")

plt.show()
