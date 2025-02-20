# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def get_returns(prices):
    rets = pd.DataFrame(index=prices.index[1:],columns=prices.columns)
    for column in prices.columns:
        rets[column] = prices[column]/prices[column].shift(1) - 1
    return rets

def portfolio_return(weights, returns):
    return np.dot(weights.T, returns)

def portfolio_std(weights, cov_mat):
    return np.dot(weights.T, np.dot(cov_mat, weights))**0.5

def portfolio_var(weights, cov_mat):
    return np.dot(weights.T, np.dot(cov_mat, weights)) 

def annualize_rets(r, periods_per_year=12):
    comp_growth = (1+r).prod()
    n_periods = len(r)
    return comp_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year=12):
    return r.std()*(periods_per_year**0.5)

def semi_deviation(r, periods_per_year=12):
    return annualize_vol(r[r<=0])

def sharpe_ratio(r, risk_free, periods_per_year=12):
    """
    Annualized sharpe ratio - requires returns and risk free of similar periods
    """
    excess_ret = r - risk_free
    ann_excess_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_excess_ret/ann_vol
    
def drawdown(r):
    index = 100*(1+r).cumprod()
    prev_peaks = index.cummax()
    drawdowns = (index - prev_peaks)/prev_peaks
    return drawdowns.min()

def var_historic(r, lvl=5):
    return np.percentile(r, lvl)

def cvar_historic(r, lvl=5):
    return r[r<=var_historic(r, lvl)].mean()

def summary_stats(port_returns, risk_free, strat_name):
    """
    Returns a DataFrame that contains aggregated summary stats for a series of portfolio returns 
    
    Parameters 
    ----------
    port_return (pd.Series): portfolio timeseries of monthly returns
    riskfree_rate (pd.Series): timeseries of monthly risk free rate
    """
    port_returns = pd.Series(port_returns)
    risk_free = pd.Series(risk_free)
    return pd.DataFrame({
        "Annualized Return": annualize_rets(port_returns, periods_per_year=12),
        "Annualized Vol": annualize_vol(port_returns, periods_per_year=12),
        "Skewness": port_returns.skew(),
        "Kurtosis": port_returns.kurt(),
        "Historic VaR (5%)": var_historic(port_returns, lvl=5),
        "Historic CVaR (5%)": cvar_historic(port_returns, lvl=5),
        "Sharpe Ratio": sharpe_ratio(port_returns, risk_free),
        "Max Drawdown": drawdown(port_returns)
         }, index=[strat_name])

def plot_efficient_frontier(port_returns, port_vols, date):
    """
    Plot the efficient frontier given sets of portfolio return and risk
    
    Parameters
    ----------
    port_returns (array)
    port_vols (array)
    """
        
    # Pick largest sharpe
    sharpe = port_returns/port_vols
    idx = np.argmax(sharpe)
    max_sharpe_ret = port_returns[idx]
    max_sharpe_risk = port_vols[idx]
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(port_vols, port_returns, label='Efficient Frontier', color="blue", linewidth=2)
    plt.scatter(port_vols, port_returns, c=sharpe, cmap='viridis', s=10, label='Portfolios')
    
    # Highlight Max Sharpe Portfolio
    plt.scatter(max_sharpe_risk, max_sharpe_ret, color='red', cmap='viridis', s=100, label='Max Sharpe Portfolio', edgecolors="black")
    plt.annotate(f"Max Sharpe: {max_sharpe_ret/max_sharpe_risk:.4f}", (max_sharpe_risk, max_sharpe_ret), xytext=(max_sharpe_risk + 0.01, max_sharpe_ret - 0.01), arrowprops=dict(facecolor="black", arrowstyle="->"), fontsize=10)
    
    # Customize
    plt.title(f'Efficient Frontier {date}', fontsize=16)
    plt.xlabel('Standard Deviation',fontsize=14)
    plt.ylabel('Return',fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
def plot_performance(port_levels):
    """
    Plots historical performance of portfolios
    
    Parameters
    ----------
    port_levels (pd.DataFrame): Columns should have each strategy's return, indexed by 'Dates'
    """
    port_levels = port_levels/port_levels.iloc[0,:]*1000 #Rebase
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    
    # Plot 
    for column in port_levels:
        plt.plot(port_levels.index, port_levels[column], label= column, alpha=0.7, linestyle='--')
        
    # Customize
    plt.title('Strategy Performance')
    plt.xlabel('Date',fontsize=14)
    plt.ylabel('Portfolio Level',fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    