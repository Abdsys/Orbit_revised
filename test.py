#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch

# ----------------------------
# 1. Load & Clean Data
# ----------------------------
df = pd.read_csv(
    'Returns.csv',
    parse_dates=['Date'],        # Parse dates from the 'Date' column
    date_format='%d-%b-%y'       # Format: day-month-year (2-digit)
)
df.set_index('Date', inplace=True)

# Clean 'Lend APY' column: remove '%' and convert to numeric
df['Lend APY'] = df['Lend APY'].str.strip().str.rstrip('%')
df['Lend APY'] = pd.to_numeric(df['Lend APY'], errors='coerce')
df.dropna(subset=['Lend APY'], inplace=True)

yield_series = df['Lend APY']

# ----------------------------
# 2. Calculate Log Returns
# ----------------------------
# Log returns: r_t = log(y_t) - log(y_{t-1})
log_returns = np.log(yield_series).diff().dropna()

# Plot log returns for visual inspection
plt.figure(figsize=(10,6))
plt.plot(log_returns, label='Log Returns')
plt.title("Time Series of Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.show()

# ----------------------------
# 3. Check for ARCH Effects using Engle's ARCH LM Test
# ----------------------------
# We use 10 lags as an example; adjust nlags as appropriate for your data.
lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(yield_series, nlags=10)

print("Engle's ARCH LM Test Results on Log Returns:")
print(f"LM Statistic: {lm_stat:.4f}")
print(f"LM p-value: {lm_pvalue:.4f}")
print(f"F Statistic: {f_stat:.4f}")
print(f"F p-value: {f_pvalue:.4f}")

# Interpretation guidance:
# A low p-value (typically < 0.05) indicates significant ARCH effects,
# suggesting that the variance of log returns is time-varying.
