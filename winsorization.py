import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Returns.csv', parse_dates=['Date'], date_format='%d-%b-%y')
df.set_index('Date', inplace=True)

# Clean the 'Lend APY' column: remove "%" and convert to numeric
df['Lend APY'] = df['Lend APY'].str.strip().str.rstrip('%')
df['Lend APY'] = pd.to_numeric(df['Lend APY'], errors='coerce')
df.dropna(subset=['Lend APY'], inplace=True)

# Optional: Winsorize the data at the 1st and 99th percentiles
lower_bound = df['Lend APY'].quantile(0.01)
upper_bound = df['Lend APY'].quantile(0.99)
df['Lend APY_Winsorized'] = df['Lend APY'].clip(lower=lower_bound, upper=upper_bound)

# Plot original vs. Winsorized data for comparison
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Lend APY'], label='Original APY', alpha=0.6)
plt.plot(df.index, df['Lend APY_Winsorized'], label='Winsorized APY', alpha=0.8)
plt.title('Comparison: Original vs. Winsorized Lend APY')
plt.xlabel('Date')
plt.ylabel('Lend APY (%)')
plt.legend()
plt.show()