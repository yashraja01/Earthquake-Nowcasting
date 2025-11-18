import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Excel file

file_path = "/content/Taiwan Datasheet 3.5a.xlsx"
df = pd.read_excel(file_path)

df = df.dropna(subset=['Cycle_peak'])
# 2. Create Bins and Labels

bins = range(0, 1901, 100)

# Labels: "1-100", "101-200", ... "1801-1900"
labels = [f'{i+1}-{i+100}' for i in range(0, 1900, 100)]

# 3. Categorize the data
# 'pd.cut' sorts the values into the bins defined above
df['Range'] = pd.cut(df['Cycle_peak'], bins=bins, labels=labels, include_lowest=True)

# 4. Calculate Frequency
frequency_data = df['Range'].value_counts().sort_index()

# 5. Plot the Bar Graph
plt.figure(figsize=(10, 6)) # Set figure size
bars = plt.bar(frequency_data.index.astype(str), frequency_data.values, color='skyblue', edgecolor='black')

# Add labels and title
plt.title('Natural Time Frequency Distribution', fontsize=14)
plt.xlabel('Natural Time Counts', fontsize=12)
plt.ylabel('Number of Earthquakes', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid lines for easier reading of values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Optional: Add the actual count numbers on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

plt.tight_layout()
plt.show()
