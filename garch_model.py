#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ----------------------------
# 1. Load ARMA Residuals
# ----------------------------
resid_df = pd.read_csv('arma_residuals.csv', parse_dates=['Date'])
resid_df.set_index('Date', inplace=True)
arma_resid = resid_df['residual']

# ----------------------------
# 2. Grid Search for GARCH Models
# ----------------------------
model_types = ['GARCH', 'GJR']  # We'll try Standard GARCH(1,1) and GJR-GARCH(1,1)
distributions = ['normal', 't', 'skewt', 'ged']  # Error distributions to test

results = []  # List to store results

for mtype in model_types:
    for dist in distributions:
        try:
            if mtype == 'GARCH':
                # Standard GARCH(1,1): p=1, o=0, q=1
                p_vol, o_vol, q_vol = 1, 0, 1
            elif mtype == 'GJR':
                # GJR-GARCH(1,1): p=1, o=1, q=1 captures asymmetry
                p_vol, o_vol, q_vol = 1, 1, 1

            # Fit the GARCH model on the ARMA residuals.
            # mean='Zero' because the ARMA has removed the mean component.
            model = arch_model(
                arma_resid,
                mean='Zero',
                vol='Garch',
                p=p_vol,
                o=o_vol,
                q=q_vol,
                dist=dist
            )
            res = model.fit(disp='off')

            # Retrieve AIC and BIC for model selection
            aic_val = res.aic
            bic_val = res.bic

            # Ljung-Box test on standardized residuals at lag 10
            lb_test = acorr_ljungbox(res.std_resid, lags=[10], return_df=True)
            lb_pval = lb_test['lb_pvalue'].iloc[0]

            results.append({
                'Model Type': mtype,
                'Distribution': dist,
                'Volatility Order (p,o,q)': f"({p_vol},{o_vol},{q_vol})",
                'AIC': aic_val,
                'BIC': bic_val,
                'Ljung-Box p-value': lb_pval,
                'Model': res  # Keep the model object for later use
            })
        except Exception as e:
            print(f"Error fitting model {mtype} with {dist} distribution: {e}")
            continue

results_df = pd.DataFrame(results)
results_df.sort_values('AIC', inplace=True)
print("\nTop GARCH Models Based on AIC and Diagnostics:")
print(results_df[['Model Type', 'Distribution', 'Volatility Order (p,o,q)', 'AIC', 'BIC', 'Ljung-Box p-value']])

# ----------------------------
# 3. Inspect the Best GARCH Model
# ----------------------------
if not results_df.empty:
    best_garch = results_df.iloc[0]['Model']
    best_config = results_df.iloc[0][['Model Type', 'Distribution', 'Volatility Order (p,o,q)', 'AIC', 'BIC', 'Ljung-Box p-value']]
    print("\nBest GARCH Model Configuration:")
    print(best_config)
    print("\nSummary of the Best GARCH Model:")
    print(best_garch.summary())

    # Diagnostic: Standardized residuals
    std_resid = best_garch.std_resid

    # 3a. Ljung-Box test on standardized residuals at lags=10, 20
    lb_test_best = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
    print("\nLjung-Box Test on Standardized Residuals:")
    print(lb_test_best)

    # 3b. Ljung-Box test on squared standardized residuals at lags=10, 20
    # This checks for any leftover ARCH effects.
    sq_std_resid = std_resid ** 2
    lb_test_sqr = acorr_ljungbox(sq_std_resid, lags=[10, 20], return_df=True)
    print("\nLjung-Box Test on SQUARED Standardized Residuals (checking for ARCH effects):")
    print(lb_test_sqr)

    # 3c. Plot time series of standardized residuals
    plt.figure(figsize=(12, 6))
    plt.plot(std_resid, label='Standardized Residuals')
    plt.title('Time Series of GARCH Standardized Residuals')
    plt.xlabel('Date')
    plt.ylabel('Standardized Residual')
    plt.legend()
    plt.show()
    
    # 3d. Plot ACF/PACF of standardized residuals
    plt.figure(figsize=(12, 6))
    plot_acf(std_resid, lags=40, alpha=0.05)
    plt.title('ACF of GARCH Standardized Residuals')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plot_pacf(std_resid, lags=40, alpha=0.05)
    plt.title('PACF of GARCH Standardized Residuals')
    plt.show()

    # 3e. Plot the Raw (Unstandardized) Residuals
    raw_resid = best_garch.resid
    plt.figure(figsize=(12, 6))
    plt.plot(raw_resid, label='Raw Residuals')
    plt.title('Time Series of GARCH Raw Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.show()
    
    # Calculate and print the mean of raw residuals
    mean_raw = raw_resid.mean()
    print(f"\nMean of Raw Residuals: {mean_raw:.6f}")

    # Optionally, plot ACF/PACF of raw residuals
    plt.figure(figsize=(12, 6))
    plot_acf(raw_resid, lags=40, alpha=0.05)
    plt.title('ACF of GARCH Raw Residuals')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plot_pacf(raw_resid, lags=40, alpha=0.05)
    plt.title('PACF of GARCH Raw Residuals')
    plt.show()
else:
    print("No GARCH model was successfully fit. Check your data or adjust model parameters.")

