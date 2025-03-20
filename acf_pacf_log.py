import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- 1. Load and Clean the Data ---
df = pd.read_csv(
    'Returns.csv',
    parse_dates=['Date'],    # Indicate which column to parse as dates
    date_format='%d-%b-%y'   # Explicitly specify format: day-month-year (2-digit)
)

# Make 'Date' the index
df.set_index('Date', inplace=True)

# Clean 'Lend APY' column: remove '%' and convert to numeric
df['Lend APY'] = df['Lend APY'].str.strip().str.rstrip('%')
df['Lend APY'] = pd.to_numeric(df['Lend APY'], errors='coerce')
df.dropna(subset=['Lend APY'], inplace=True)

yield_series = df['Lend APY']

# --- 2. Calculate Log Returns ---
# Compute log returns: r_t = log(y_t) - log(y_{t-1})
log_returns = np.log(yield_series).diff().dropna()

# --- 3. Plot ACF & PACF for Log Returns ---
plt.figure(figsize=(12, 6))
plot_acf(log_returns, lags=40, alpha=0.05)
plt.title('ACF of Log Returns')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(log_returns, lags=40, alpha=0.05)
plt.title('PACF of Log Returns')
plt.show()

# --- 4. Ljung-Box Test on Log Returns ---
lb_test = acorr_ljungbox(log_returns, lags=[10, 20], return_df=True)
print("Ljung-Box Test Results on Log Returns:")
print(lb_test)