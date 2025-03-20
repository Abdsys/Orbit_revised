import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- 1. Read CSV and parse dates with a specified format ---
df = pd.read_csv(
    'Returns.csv',
    parse_dates=['Date'],        # Indicate which columns to parse as datetimes
    date_format='%d-%b-%y'       # Explicitly specify day-month-year
)

# Set the Date column as index
df.set_index('Date', inplace=True)

# --- 2. Clean the 'Lend APY' column ---
# Remove the "%" sign and extra whitespace
df['Lend APY'] = df['Lend APY'].str.strip().str.rstrip('%')
# Convert to numeric (float), coerce errors to NaN
df['Lend APY'] = pd.to_numeric(df['Lend APY'], errors='coerce')
# Drop any rows with NaN
df.dropna(subset=['Lend APY'], inplace=True)

yield_series = df['Lend APY']

# ----------------------------
# ORIGINAL SERIES TESTS & PLOTS
# ----------------------------

# ADF Test on original series
result_adf = adfuller(yield_series)
print('\nADF Test Results on Original Series:')
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
print('Critical Values:')
for key, value in result_adf[4].items():
    print(f'   {key}: {value}')

# KPSS Test on original series
result_kpss = kpss(yield_series, regression='c')
print('\nKPSS Test Results on Original Series:')
print('KPSS Statistic:', result_kpss[0])
print('p-value:', result_kpss[1])
print('Critical Values:')
for key, value in result_kpss[3].items():
    print(f'   {key}: {value}')

# Plot the original time series
plt.figure(figsize=(10, 6))
plt.plot(yield_series, label='Original Lend APY')
plt.title('Time Series of Lend APY')
plt.xlabel('Date')
plt.ylabel('Lend APY (%)')
plt.legend()
plt.show()

# ----------------------------
# CALCULATE LOG RETURNS
# ----------------------------
# Log returns: r_t = log(y_t) - log(y_{t-1})
log_returns = np.log(yield_series).diff().dropna()

# ADF Test on log returns
result_adf_log = adfuller(log_returns)
print('\nADF Test Results on Log Returns:')
print('ADF Statistic:', result_adf_log[0])
print('p-value:', result_adf_log[1])
print('Critical Values:')
for key, value in result_adf_log[4].items():
    print(f'   {key}: {value}')

# KPSS Test on log returns
result_kpss_log = kpss(log_returns, regression='c')
print('\nKPSS Test Results on Log Returns:')
print('KPSS Statistic:', result_kpss_log[0])
print('p-value:', result_kpss_log[1])
print('Critical Values:')
for key, value in result_kpss_log[3].items():
    print(f'   {key}: {value}')

# Plot the log returns time series
plt.figure(figsize=(10, 6))
plt.plot(log_returns, label='Log Returns (%)', color='green')
plt.title('Time Series of Log Returns')
plt.xlabel('Date')
plt.ylabel('Log Return (%)')
plt.legend()
plt.show()

# ----------------------------
# FURTHER CHARTS FOR INTERPRETATION (Rolling Statistics)
# ----------------------------
rolling_window = 30  # adjust window size as needed

# Rolling stats on the ORIGINAL series
rolling_mean_orig = yield_series.rolling(window=rolling_window).mean()
rolling_std_orig = yield_series.rolling(window=rolling_window).std()

plt.figure(figsize=(10, 6))
plt.plot(yield_series, color='blue', label='Original Lend APY')
plt.plot(rolling_mean_orig, color='red', label=f'Rolling Mean ({rolling_window} days)')
plt.plot(rolling_std_orig, color='black', label=f'Rolling Std ({rolling_window} days)')
plt.title('Original Series with Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Lend APY (%)')
plt.legend()
plt.show()

# Rolling stats on the LOG RETURNS
rolling_mean_log = log_returns.rolling(window=rolling_window).mean()
rolling_std_log = log_returns.rolling(window=rolling_window).std()

plt.figure(figsize=(10, 6))
plt.plot(log_returns, color='blue', label='Log Returns')
plt.plot(rolling_mean_log, color='red', label=f'Rolling Mean ({rolling_window} days)')
plt.plot(rolling_std_log, color='black', label=f'Rolling Std ({rolling_window} days)')
plt.title('Log Returns with Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Log Return (%)')
plt.legend()
plt.show()

# ----------------------------
# SUMMARY TABLE OF TEST RESULTS
# ----------------------------
results = {
    'Test': [
        'ADF - Original', 
        'KPSS - Original', 
        'ADF - Log Returns', 
        'KPSS - Log Returns'
    ],
    'Statistic': [
        result_adf[0], 
        result_kpss[0], 
        result_adf_log[0], 
        result_kpss_log[0]
    ],
    'p-value': [
        result_adf[1], 
        result_kpss[1], 
        result_adf_log[1], 
        result_kpss_log[1]
    ],
    'Critical Value (1%)': [
        result_adf[4]['1%'], 
        result_kpss[3]['1%'], 
        result_adf_log[4]['1%'], 
        result_kpss_log[3]['1%']
    ],
    'Critical Value (5%)': [
        result_adf[4]['5%'], 
        result_kpss[3]['5%'], 
        result_adf_log[4]['5%'], 
        result_kpss_log[3]['5%']
    ],
    'Critical Value (10%)': [
        result_adf[4]['10%'], 
        result_kpss[3]['10%'], 
        result_adf_log[4]['10%'], 
        result_kpss_log[3]['10%']
    ]
}

results_table = pd.DataFrame(results)
print("\nSummary Table of Test Results:")
print(results_table)