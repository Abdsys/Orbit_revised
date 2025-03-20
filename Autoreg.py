import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------
# 1. Load & Clean Data
# ----------------------------
df = pd.read_csv(
    'Returns.csv',
    parse_dates=['Date'],        # Parse the 'Date' column as datetimes
    date_format='%d-%b-%y'       # Specify the date format: day-month-year (2-digit)
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

# ----------------------------
# 3. Ljung-Box Test on Log Returns
# ----------------------------
lb_test = acorr_ljungbox(log_returns, lags=[10, 20], return_df=True)
print("Ljung-Box Test Results on Log Returns:")
print(lb_test)

# ----------------------------
# 4. Grid Search over ARMA(p,q) on Log Returns
# ----------------------------
p_values = range(0, 4)  # Try p = 0, 1, 2, 3
q_values = range(0, 4)  # Try q = 0, 1, 2, 3

best_aic = float('inf')
best_order = None
best_model = None
results_list = []  # Store all model results

for p in p_values:
    for q in q_values:
        try:
            # Fit ARMA(p,q) on the log returns (use ARIMA with d=0 because log_returns are already differenced)
            model = ARIMA(log_returns, order=(p, 0, q))
            res = model.fit()
            
            current_aic = res.aic
            current_bic = res.bic
            
            # Ljung-Box test at lags 10, 20 on residuals
            lb_test_model = acorr_ljungbox(res.resid, lags=[10, 20], return_df=True)
            lb_pval_20 = lb_test_model['lb_pvalue'].iloc[1]  # p-value at lag=20
            
            results_list.append({
                'p': p,
                'q': q,
                'AIC': current_aic,
                'BIC': current_bic,
                'LB_pval_lag20': lb_pval_20
            })
            
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = (p, q)
                best_model = res
            
        except Exception as e:
            print(f"ARMA({p},{q}) failed: {e}")

results_df = pd.DataFrame(results_list)
results_df.sort_values('AIC', inplace=True)
print("\nAll Model Results (sorted by AIC):")
print(results_df)

# ----------------------------
# 5. Inspect the Best Model
# ----------------------------
if best_model is not None:
    print("\nBest ARMA model based on AIC:")
    print(f"Order=(p,q)={best_order}")
    print(f"AIC={best_aic}, BIC={best_model.bic}")
    print(best_model.summary())
    
    # Residual Diagnostics for the Best Model
    residuals = best_model.resid

    # After fitting your best ARMA model and storing it as best_model:
    residuals.to_csv('arma_residuals.csv', header=['residual'])

    
    print("\nLjung-Box Test on the Best Model's Residuals (lags=10,20):")
    lb_test_best = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print(lb_test_best)
    
    # Plot the time series of residuals
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='Residuals')
    plt.title(f'Time Series of Residuals - Best ARMA{best_order}')
    plt.xlabel('Date')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.show()
    
    # Plot ACF/PACF of the best model residuals
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    plt.figure(figsize=(12, 6))
    plot_acf(residuals, lags=40, alpha=0.05)
    plt.title(f'ACF of Residuals - Best ARMA{best_order}')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plot_pacf(residuals, lags=40, alpha=0.05)
    plt.title(f'PACF of Residuals - Best ARMA{best_order}')
    plt.show()
else:
    print("\nNo model was successfully fit. Adjust p, q ranges or check data.")
