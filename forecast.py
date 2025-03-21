#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# ----------------------------
# 1. Load ARMA Residuals (Full Data)
# ----------------------------
resid_df = pd.read_csv('arma_residuals.csv', parse_dates=['Date'])
resid_df.set_index('Date', inplace=True)
resid_all = resid_df['residual']  # entire series

# Decide on a train-test split index in integer terms
split_index = int(len(resid_all) * 0.8)
train_resid = resid_all.iloc[:split_index]
test_resid = resid_all.iloc[split_index:]

print(f"Training observations: {len(train_resid)}, Testing observations: {len(test_resid)}")

# ----------------------------
# 2. Fit GARCH on Training Portion
# ----------------------------
garch_model = arch_model(
    train_resid,
    mean='Zero',
    vol='Garch',
    p=1, o=0, q=1,
    dist='skewt'
)
garch_result = garch_model.fit(disp='off')
print("\nGARCH(1,1) Model Summary (Fitted on Training Data):")
print(garch_result.summary())

# ----------------------------
# 3. Manual Rolling Forecast
# ----------------------------
# We'll perform a rolling forecast by re-fitting the model with additional observations as we move through the test period.
forecasted_variance = []  # to store one-step-ahead forecast variance

for i in range(len(test_resid)):
    # Extend the sample: training data + test data up to current point
    current_data = resid_all.iloc[:split_index + i]
    model = arch_model(current_data, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='skewt')
    result = model.fit(disp='off')
    # Forecast one step ahead
    fcast = result.forecast(horizon=1)
    # The forecasted variance for the next period is located in the last row, first column
    forecast_var = fcast.variance.iloc[-1, 0]
    forecasted_variance.append(forecast_var)

# Convert forecasted variance to a Series with test_resid index
forecast_variance = pd.Series(forecasted_variance, index=test_resid.index)
forecast_volatility = np.sqrt(forecast_variance)

print("\nFirst few rows of forecasted variance:")
print(forecast_variance.head())
print("\nFirst few rows of forecasted volatility:")
print(forecast_volatility.head())

# ----------------------------
# 4. Realized Volatility in Test Period
# ----------------------------
realized_vol = test_resid.pow(2).rolling(window=5).mean().apply(np.sqrt).dropna()

# ----------------------------
# 5. Plot Forecasted vs. Realized Volatility
# ----------------------------
plt.figure(figsize=(12,6))
plt.plot(forecast_volatility.index, forecast_volatility, label='Forecasted Volatility', lw=2)
plt.plot(realized_vol.index, realized_vol, label='Realized Volatility (5-day Rolling)', lw=2, alpha=0.8)
plt.title('Forecasted vs. Realized Volatility (Out-of-Sample)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# ----------------------------
# 6. Print Summaries
# ----------------------------
print("\nForecasted Volatility Summary:")
print(forecast_volatility.describe())

print("\nRealized Volatility Summary:")
print(realized_vol.describe())
