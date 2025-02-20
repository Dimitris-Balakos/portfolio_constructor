# -*- coding: utf-8 -*-

# In[0 - Imports]
import os

user_dir = r'C:\Users\Dimitris\Desktop\Financial Competency Assessment Deliverable' # Change accordingly
os.chdir(user_dir) 

import pandas as pd
import numpy as np
from scipy import optimize
import Helper as hlp

data = pd.read_excel(user_dir+'\DatasetExample.xlsx',sheet_name='Tab 1 (15 stocks - monthly)',skiprows=1)
data.set_index('Dates',inplace=True)


# In[1 - MVO Functions]

def monte_carlo_sim(exp_rets, cov_mat, num_simulations=100):
    """
    Simulates returns using Monte Carlo simulation.
    
    Parameters
    ----------
    exp_rets: Vector of expected returns (Nx1)
    cov_mat: Asset variance-covariance matrix (NxN)
    num_simulations: Number of Monte Carlo simulations (default 100)
    
    Returns
    -------
    Simulated returns matrix (num_simulations x N)
    """
    np.random.seed(300) 
    simulated_returns = np.random.multivariate_normal(exp_rets, cov_mat, size=num_simulations)
    return simulated_returns

def minimal_var_port(exp_rets, cov_mat, weight_collar=(0.00, 1.0), verbose=False, monte_carlo=False, num_simulations=100):
    """
    Outputs the minimal variance portfolio weights from a set of constituents
    
    Parameters
    -------
    daily_rets: Vector of asset daily returns (kxN)
    cov_mat (pd.DataFrame): Asset variance covariance matrix (NxN)
    weight_collar (tuple) : Tuple of minimum and maximum security level weights. The default is (0.0,1.0).
    verbose: log switch
    monte_carlo (bool): If True, use Monte Carlo simulation to generate returns
    num_simulations (int): Number of simulations for Monte Carlo (if monte_carlo==True)
    
    Returns
    -------
    Vector of optmized weights
    """
    if monte_carlo:
        # Simulate returns using Monte Carlo
        simulated_returns = monte_carlo_sim(exp_rets, cov_mat, num_simulations=num_simulations)
        # Recalculate expected returns and covariance matrix 
        exp_rets = simulated_returns.mean(axis=0)  # Expected returns from simulations
        cov_mat = np.cov(simulated_returns.T)      # Covariance matrix from simulated returns
        
    n = len(exp_rets)
    guess = np.repeat(0.1,n)
    
    #Set security level weight bands 
    bounds = (weight_collar,)*n
    #Set full investment constraint 
    is_weight_one = {
        'type':'eq',
        'fun':lambda weights: np.sum(weights) - 1.0}
    
    #Set objective function
    output_weights = optimize.minimize(fun=hlp.portfolio_std,
                                        x0=guess,
                                        args=(cov_mat),
                                        method='SLSQP',
                                        options = {'maxiter':100, 'disp': verbose},
                                        constraints = (is_weight_one), 
                                        bounds = bounds
                                        )
    
    return output_weights.x

def minimize_volatility(target_return, exp_rets, cov_mat, weight_collar=(0.00, 1.0), verbose=False):
    """
    Outputs the minimum volatility set of weights for a level of portfolio return
    
    Parameters
    ----------
    target_return 
    exp_rets: Vector of asset expected returns (Nx1)
    cov_mat (pd.DataFrame): Asset variance covariance matrix (NxN)
    weight_collar (tuple) : Tuple of minimum and maximum security level weights. The default is (0.0,1.0).
    verbose: optimizer log switch
    
    Returns
    -------
    Vector of optmized weights
    """
        
    n = len(exp_rets)
    guess = np.repeat(1/n,n)
    
    #Set security level weight bands 
    bounds = (weight_collar,)*n
    #Set target return constraint
    is_target_return = {
        'type':'eq',
        'args':(exp_rets,),
        'fun':lambda weights, er: target_return - hlp.portfolio_return(weights, exp_rets)}
    #Set full investment constraint 
    is_weight_one = {
        'type':'eq',
        'fun':lambda weights: np.sum(weights) - 1.0}
    
    #Optimization
    output_weights = optimize.minimize(fun=hlp.portfolio_std,
                                        x0=guess,
                                        args=(cov_mat),
                                        method='SLSQP',
                                        options = {'maxiter':100, 'disp': verbose},
                                        constraints = (is_target_return, is_weight_one), 
                                        bounds = bounds
                                        )
    
    
    return output_weights.x

def mvo_optimal_weights(port_points, exp_rets, cov_mat, weight_collar=(0.00,1.0),verbose=False,monte_carlo=False, num_simulations=100):
    """
    Calculates the efficient frontier portfolios between the minimal variance and maximum return range
    
    Parameters
    ----------
    exp_rets: Vector of asset expected returns (Nx1)
    cov_mat: Asset variance covariance matrix (NxN)
    weight_collar: Tuple of minimum and maximum security level weights
    verbose: log switch
    monte_carlo (bool): If True, use Monte Carlo simulation to generate returns
    num_simulations (int): Number of simulations for Monte Carlo (if monte_carlo==True)
    """
    if monte_carlo:
        simulated_returns = monte_carlo_sim(exp_rets, cov_mat, num_simulations=num_simulations)
        exp_rets = simulated_returns.mean(axis=0)
        cov_mat = np.cov(simulated_returns.T)
    #Fetch returns between minimal variance and maximum return portfolios
    min_var_return = np.dot(minimal_var_port(exp_rets, cov_mat, weight_collar, verbose).T,exp_rets) # Calc minimal var return 
    target_returns = np.linspace(min_var_return, exp_rets.max(), port_points)
    #Minimize variance for each return target between the range
    output_weights = [minimize_volatility(target_ret, exp_rets, cov_mat, weight_collar, verbose) for target_ret in target_returns]
    
    return output_weights

# In[2 - Risk Parity Functions]

def fetch_risk_model_factors(start_date, freq='M'):
    """
    Fetches monthly/annual Carhart factors from kenneth french library
    
    Parameters
    ---------
    start_date (str)
    freq (str): Frequency switch. Has to be 'A' or 'M'
    
    Returns
    -------
    (pd.DataFrame) ['mkt_excess', 'SMB', 'HML', 'Mom']: Factor returns 
    """
    import pandas_datareader as pdr
    ff_3f = pdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start=start_date).read()
    mom_f = pdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor',start=start_date).read()
    slcer = 0 if freq=='M' else 1 if freq=='A' else None
    if slcer is None:
        raise ValueError('Frequency must set as either "A" or "B"')
    factors = pd.merge(ff_3f[slcer], mom_f[slcer], how='inner', left_index=True, right_index=True) 
    factors.index = factors.index.strftime('%Y-%m') # manipulate date index for follow up merge
    
    return factors

def calc_asset_loadings(prices, factors, freq='M'):
    """
    Computes asset loadings via least squares optimization
    
    Parameters
    ----------
    prices (pd.DataFrame): Timeseries of asset prices
    """
    import statsmodels.api as sm
    import pdb
    #pdb.set_trace()
    # Clear column strings
    factors.columns = [column.strip() for column in factors.columns] 
    factors = factors/100 #transform to decimal
    asset_returns = hlp.get_returns(prices)
    asset_returns.index = asset_returns.index.strftime('%Y-%m')
    asset_loadings = pd.DataFrame()
    residual_var = pd.DataFrame(index=asset_returns.columns , columns=['ResVar'])
    #compute loadings for each asset
    for asset in asset_returns.columns:
        df = asset_returns[[asset]]
        df = pd.merge(df,factors, how='inner', left_index=True, right_index=True)
        df['ret_excess'] = df[asset] - df['RF']
        df['mkt_excess'] = df['Mkt-RF'] 
        betas = pd.DataFrame(sm.formula.ols(formula="ret_excess ~ mkt_excess + SMB + HML + Mom",data = df).fit().params) # fit carhart model
        betas.columns = [asset]
        res_var = sm.formula.ols(formula="ret_excess ~ mkt_excess + SMB + HML + Mom",data = df).fit().resid.var()
        asset_loadings = pd.concat([betas, asset_loadings],axis=1) # Store Rebal betas
        residual_var.loc[residual_var.index==asset, 'ResVar'] = res_var
    factors = factors.drop(columns=['RF']).rename(columns={'Mkt-RF':'mkt_excess'}) # exclude the intercept
    #Output factor covariance matrix of equivalent window as that of the loadings
    factors = factors[factors.index.isin(df.index)]
    
    return asset_loadings.transpose().iloc[:,1:], factors, residual_var

def risk_contribution(weights, cov_mat):
    """
    Calculates the risk contribution of each asset
    """
    import pdb
    #pdb.set_trace()
    portfolio_var = hlp.portfolio_var(weights, cov_mat) 
    marginal_contr = np.dot(cov_mat, weights) 
    risk_contr = np.multiply(weights, marginal_contr) / portfolio_var
    
    return risk_contr

def risk_parity(weights, cov_mat):
    """
    Defines the risk parity objective function to be minimized: Delta of risk contributions 
    """
    import pdb
    #pdb.set_trace()
    risk_contr = risk_contribution(weights, cov_mat)
    target_risk = np.mean(risk_contr)
   
    return np.sum((risk_contr-target_risk)**2)

def risk_parity_optimal_weights(cov_mat,weight_collar=(0.00,1.0),verbose=True):
    """
    Optimizer call that applies the risk parity objective function along with long only, full investment
    and asset level weight caps constraints.

    Parameters
    ----------
    cov_mat (pd.DataFrame) : Asset variance covariance matrix (NxN)
    weight_collar (tuple) : Tuple of minimum and maximum security level weights. The default is (0.0,1.0).
    verbose (bool) : optimizer log switch

    Returns
    -------
    Vector of optmized weights
    """
    import pdb
    #pdb.set_trace()
    n = len(cov_mat)
    guess = np.repeat(1/n,n)
    
    #Set security level weight bands 
    bounds = (weight_collar,)*n
    #Set full investment constraint 
    is_weight_one = {
        'type':'eq',
        'fun':lambda weights: np.sum(weights) - 1.0}
    
    #Optimization
    output_weights = optimize.minimize(fun=risk_parity,
                                        x0=guess,
                                        args=(cov_mat),
                                        method='SLSQP',
                                        options = {'maxiter':100, 'disp': verbose},
                                        constraints = (is_weight_one), 
                                        bounds = bounds
                                        )
    
    return output_weights.x

def verify_risk_parity(portfolio_returns, factors):
    """
    Checks if a time series of portfolio returns exhibit risk parity against the Carhart 4 factor model
    
    Parameters
    ----------
    portfolio_returns (pd.DataFrame): Index: Date, One column with returns
    factors (pd.DataFrame) ['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']
    """
    import pdb
    #pdb.set_trace()
    # Get factor loadings and return period factor returns
    factor_loadings, factors, residual_var = calc_asset_loadings(portfolio_returns, factors)
    factor_cov = factors.cov()
    # Portfolio variance
    risk_cont = risk_contribution(factor_loadings.T, factor_cov) + np.diag(residual_var['ResVar'])
    
    return risk_cont

# In[3 - MDP Functions]

def diversification_ratio(weights, volatilities, cov_mat):
    """
    Calculate the Diversification Ratio (DR)    
    """
    weighted_vols = weights*volatilities
    port_vol = hlp.portfolio_std(weights, cov_mat)
    return weighted_vols.sum()/port_vol

def most_diversified_portfolio(weights, volatilities, cov_mat):
    """
    Defines the objective function to be minimized: negative of divesrification ratio
    """
    return -diversification_ratio(weights, volatilities, cov_mat)

def mdp_optimal_weights(returns, weight_collar=(0.00,1.0),verbose=True):
    """
    Constructs the Most Diversified Portfolio (MDP) by applying the MDP objective function with long only, full investment
    and asset level weight caps constraints.
    
    Parameters
    ----------
    returns (pd.DataFrame) : Asset return timeseries
    weight_collar (tuple) : Tuple of minimum and maximum security level weights. The default is (0.0,1.0).
    
    Returns
    -------
    Vector of optmized weights
    Diversification Ratio of optimized weights' portfolio
    """
    
    # Calculate volatilites & covariance matrix
    volatilities = returns.std()
    cov_mat = returns.cov()
    
    n = len(cov_mat)
    guess = np.repeat(1/n,n)
    
    #Set security level weight bands 
    bounds = (weight_collar,)*n
    #Set full investment constraint 
    is_weight_one = {
        'type':'eq',
        'fun':lambda weights: np.sum(weights) - 1.0}
    
    #Optimization
    output_weights = optimize.minimize(fun=most_diversified_portfolio,
                                        x0=guess,
                                        args=(volatilities, cov_mat),
                                        method='SLSQP',
                                        options = {'maxiter':100, 'disp': verbose},
                                        constraints = (is_weight_one), 
                                        bounds = bounds
                                        )
    return output_weights.x, diversification_ratio(output_weights.x, volatilities, cov_mat)

# In[4 - Backtesting Functions]

def Rebalancer(prices, rebal_method, rebal_dates, weight_collar=(0.00,1.0),monte_carlo=False, num_simulations=1000):
    """
    Rebalances portfolio
    
    Parameters
    ----------
    prices: asset prices time series
    rebal_method: rebalance methodology to be implemented {'MVO', 'MVO_sim', 'RiskParity', 'EW', 'MDP'}
    rebal_dates: list of rebalance dates
     
    Returns
    -------
    (pd.DataFrame): Portfolio Baskets for each rebalance date cross section {'Weight', 'Date'}
    (pd.DataFrame): Efficient frontier for each rebalance (rebal_method=='MVO')
    (pd.DataFrame): Diversification ratio for each rebalance (rebal_method=='MDP')
    """
    final_port = pd.DataFrame()
    prices = prices.sort_index(ascending=True)
    import pdb
    #pdb.set_trace()
    if (rebal_method=='MVO') | (rebal_method=='MVO_sim'):
        eff_front = pd.DataFrame()
        print('Building Mean-Variance Optimized Portfolios') if rebal_method=='MVO' else print('Building Mean-Variance Optimized Portfolios with Monte Carlo estimates')
        for date in rebal_dates:
            print(f'Rebal date: {date}')
            prices_dt = prices[prices.index<=date]
            returns_dt = hlp.get_returns(prices_dt)
            cov_mat_dt = returns_dt.cov()
            exp_rets_dt = hlp.annualize_rets(returns_dt)
            # Calculate efficient frontier portfolios
            if rebal_method=='MVO':
                eff_front_dt_ports = mvo_optimal_weights(port_points=50, exp_rets=exp_rets_dt, cov_mat=cov_mat_dt, weight_collar=weight_collar, monte_carlo=monte_carlo, num_simulations=num_simulations)
            # Calculate efficient frontier portfolios with simulated expected returns (100 simulations)
            elif rebal_method=='MVO_sim':
                eff_front_dt_ports = mvo_optimal_weights(port_points=50, exp_rets=exp_rets_dt, cov_mat=cov_mat_dt, weight_collar=weight_collar, monte_carlo=monte_carlo, num_simulations=num_simulations)
            eff_front_dt = pd.DataFrame({"Return": [hlp.portfolio_return(eff_front_dt_ports[n],exp_rets_dt) for n in range(len(eff_front_dt_ports))],
                                      "Standard Deviation" : [hlp.portfolio_std(eff_front_dt_ports[n],cov_mat_dt)*12**0.5 for n in range(len(eff_front_dt_ports))],
                                      "Date": [date for n in range(len(eff_front_dt_ports))]
                                        })
            eff_front_dt['Sharpe Ratio'] = eff_front_dt['Return']/eff_front_dt['Standard Deviation']
            # Pick maximum sharpe portfolio
            index = eff_front_dt[eff_front_dt['Sharpe Ratio']==eff_front_dt['Sharpe Ratio'].max()].index[0]
            optimal_port = eff_front_dt_ports[index]
            optimal_port = pd.DataFrame(data=optimal_port,index=exp_rets_dt.index,columns=['Weight'])
            optimal_port['Date'] = date
            # Store rebalance and efficient frontier portfolios 
            final_port = pd.concat([optimal_port, final_port],axis=0)
            eff_front = pd.concat([eff_front_dt, eff_front],axis=0)
            
        return final_port, eff_front            
        
    elif rebal_method=='RiskParity':
        print('Building Risk Parity Portfolios')
        # Fetch factor returns
        factor_data = fetch_risk_model_factors(prices.index.min().strftime('%Y-%m'), freq='M') 
        for date in rebal_dates:
            print(f'Rebal date: {date}')
            # Fetch prices for rolling 60 month return
            prices_dt = prices[prices.index<=date].iloc[-61:, :]
            # Fetch asset_loadings and residual return
            asset_loadings, factors, residual_var = calc_asset_loadings(prices_dt,factor_data)
            factor_cov = factors.cov()
            # Compute asset variance covariance matrix
            asset_cov = pd.DataFrame(data=np.dot(asset_loadings, np.dot(factor_cov, asset_loadings.T)) + np.diag(residual_var['ResVar']), index=asset_loadings.index, columns=asset_loadings.index)
            # Compute risk parity weights
            risk_parity_weights = pd.DataFrame(data=risk_parity_optimal_weights(asset_cov,weight_collar),index=asset_loadings.index,columns=['Weight'])
            risk_parity_weights['Date'] = date    
            final_port = pd.concat([risk_parity_weights, final_port],axis=0)                               
    
    elif rebal_method=='EW':
        print('Building Equally Weighted Portfolios')
        for date in rebal_dates:
            port_dt = pd.DataFrame(index=prices.columns, columns=['Weight','Date'])
            port_dt['Weight'] = 1/len(port_dt.index)
            port_dt['Date'] = date
            final_port = pd.concat([port_dt, final_port],axis=0)     
            
    elif rebal_method=='MDP':
        # Initialize table to store portfolio Diversification Ratio at each rebal date
        dr_ts = pd.DataFrame(index=rebal_dates,columns=['DR'])
        print('Building Most Diversified Portfolios')
        for date in rebal_dates:
            print(f'Rebal date: {date}')
            prices_dt = prices[prices.index<=date]
            returns_dt = hlp.get_returns(prices_dt)
            mdp_weights, dr = mdp_optimal_weights(returns_dt, weight_collar,verbose=True)
            mdp_port = pd.DataFrame(data=mdp_weights, index=returns_dt.columns, columns=['Weight'])
            mdp_port['Date'] = date
            # Store rebal portfolios and DR
            final_port = pd.concat([mdp_port, final_port],axis=0)
            dr_ts.loc[dr_ts.index==date,'DR'] = float(dr)
            
        return final_port, dr_ts 
        
    else:
        raise ValueError('rebal_method has to be set as one of the following: ["MVO", "MVO_sim", "RiskParity", "EW", "MDP"]')
        
    return final_port 

def Backtest(final_port, prices):
    """
    Calculates portfolio levels
    
    Parameters
    ----------
    final_port (pd.DataFrame) [Weight, Date]: Rebal portfolios dataframe. 
    prices (pd.DataFrame): Asset prices time series 
    
    Returns
    -------
    (pd.DataFrame) [Weight, Date] : End of month portfolio baskets
    (pd.DataFrame) [Portfolio Return, Portfolio Level] : End of month portfolio returns
    """
    
    drifted_port = pd.DataFrame()
    # Set dataframe to store monthly portfolio returns
    port_return = pd.DataFrame(index=prices[prices.index>=final_port['Date'].min()].index,columns=['Portfolio Return'])
    # Fetch rebal baskets
    for n,date in enumerate(sorted(final_port['Date'].unique())):
        print(f'>>>>Drifting rebal portfolio: {date}')
        port_dt = final_port[final_port['Date']==date]
        # Set dataframe to store monthly drifted weights
        drifted_port = pd.concat([port_dt,drifted_port],axis=0)
        # Exception to fetch all following available prices if we are drifting last rebal portfolio
        try:
            prices_dt = prices[(prices.index>=rebal_dates[n])&(prices.index<=rebal_dates[n+1])]
        except:
            prices_dt = prices[(prices.index>=rebal_dates[n])]
        rets_dt = hlp.get_returns(prices_dt)
        # Fetch monthly returns and calculate portfolio return along with drifted weights for month ends
        for dt in sorted(rets_dt.index.unique()):
            print(f'To {dt}')
            # Compute and store monthly portfolio return
            rets_month = rets_dt[rets_dt.index==dt]
            port_return_dt = hlp.portfolio_return(pd.Series(port_dt['Weight']), rets_month.T) 
            port_return.loc[port_return.index==dt, 'Portfolio Return'] = port_return_dt
            rets_month = rets_month + 1
            # Adjust month end weights 
            port_dt = pd.merge(port_dt, rets_month.T.rename(columns={rets_month.T.columns[0]:'Return'}), how='inner',left_index=True,right_index=True) 
            port_dt['Date'] = dt
            port_dt['Weight'] = port_dt['Weight']*port_dt['Return']
            #Normalize Weights
            port_dt['Weight'] = port_dt['Weight']/port_dt['Weight'].sum()
            port_dt.drop(columns='Return',inplace=True)
            # Do not drift weights if it is a rebal month
            if dt not in [x for x in rebal_dates]:
                drifted_port = pd.concat([port_dt,drifted_port],axis=0)
    port_return['Portfolio Level'] = (port_return['Portfolio Return'] + 1).cumprod()
    port_return['Portfolio Level'] = port_return['Portfolio Level'].astype(float)
    port_return['Portfolio Return'] = port_return['Portfolio Return'].astype(float)
    
    return drifted_port, port_return.dropna()

# In[5 - Run Backtests]    

# Call Risk Parity Backtest
factors =  fetch_risk_model_factors(data.index.min().strftime('%Y-%m'), freq='M')
rebal_method = 'RiskParity'
rebal_dates = data.iloc[60:].index # Start building portfolios from 2005-12-30 to ensure timeseries sufficiency for regressions
rebal_dates = rebal_dates[rebal_dates.month.isin([3,6,9,12])] # Quarterly Rebals
risk_parity_ports = Rebalancer(data,'RiskParity', rebal_dates, weight_collar=(0.02,0.40)) # Call Risk Parity portfolio constructor
risk_parity_drifted_port, risk_parity_levels_rets = Backtest(risk_parity_ports, data) # Calculate portfolio levels/returns & intra review monthly drifted portfolios
risk_parity_levels = risk_parity_levels_rets[['Portfolio Level']].rename(columns={'Portfolio Level':'RiskParity'})
risk_parity_rets = risk_parity_levels_rets[['Portfolio Return']].rename(columns={'Portfolio Return':'RiskParity'})
risk_parity_rets.index = risk_parity_rets.index.strftime('%Y-%m')
risk_parity_rets = pd.merge(risk_parity_rets, factors['RF'], how='left', left_index = True, right_index=True)
# QC - Validate weight constraints & Check factor exposures of portfolio returns
verify_risk_parity(risk_parity_levels, factors=factors) # Verify risk parity
risk_parity_drifted_port.groupby('Date')['Weight'].sum() 
risk_parity_drifted_port.groupby('Date')['Weight'].count()
risk_parity_ports['Weight'].max()
risk_parity_ports['Weight'].min()

#Call MVO Backtest
rebal_dates = data.iloc[150:].index # Start building portfolios from 2013-06-28 to ensure timeseries sufficiency for asset covariance matrix
rebal_dates = rebal_dates[rebal_dates.month.isin([3,6,9,12])] # Quarterly Rebals
mvo_ports, eff_front = Rebalancer(data,'MVO', rebal_dates, weight_collar=(0.02,0.40)) # Call MVO portfolio constructor
mvo_drifted_port, mvo_levels_rets = Backtest(mvo_ports, data) # Calculate portfolio levels/returns & intra review monthly drifted portfolios
mvo_levels = mvo_levels_rets[['Portfolio Level']].rename(columns={'Portfolio Level':'MVO'})
mvo_rets = mvo_levels_rets[['Portfolio Return']].rename(columns={'Portfolio Return':'MVO'})
mvo_rets.index = mvo_rets.index.strftime('%Y-%m')
mvo_rets = pd.merge(mvo_rets, factors['RF'], how='left', left_index = True, right_index=True)
# QC -  Validate weight constraints
mvo_drifted_port.groupby('Date')['Weight'].sum() 
mvo_drifted_port.groupby('Date')['Weight'].count()
mvo_ports['Weight'].max()
mvo_ports['Weight'].min()

# Call EW Backtest
rebal_dates = data.index
rebal_dates = rebal_dates[rebal_dates.month.isin([3,6,9,12])] # Returns data not needed for construction - Start quarterly rebalances at first eligible data cut off date ('2000-12-29')
ew_ports = Rebalancer(data,'EW', rebal_dates) # Call EW portfolio constructor
ew_drifted_port, ew_levels_rets = Backtest(ew_ports, data) #Calculate portfolio levels/returns & intra review monthly drifted portfolios
ew_levels = ew_levels_rets[['Portfolio Level']].rename(columns={'Portfolio Level':'EW'})
ew_rets = ew_levels_rets[['Portfolio Return']].rename(columns={'Portfolio Return':'EW'})
ew_rets.index = ew_rets.index.strftime('%Y-%m')
ew_rets = pd.merge(ew_rets, factors['RF'], how='left', left_index = True, right_index=True)
# QC -  Validate weight constraints
ew_drifted_port.groupby('Date')['Weight'].sum() 
ew_drifted_port.groupby('Date')['Weight'].count()

# Call MDP Backtest
rebal_dates = data.iloc[40:].index 
rebal_dates = rebal_dates[rebal_dates.month.isin([3,6,9,12])] #Start building portfolios from 2004-06-30 to ensure desired timeseries window
mdp_ports, dr_ts = Rebalancer(data,'MDP', rebal_dates, weight_collar=(0.02,0.40)) # Call MDP portfolio constructor & Rebalance portfolios' Diversified Ratio
mdp_drifted_port, mdp_levels_rets = Backtest(mdp_ports, data) #Calculate portfolio levels/returns & intra review monthly drifted portfolios
mdp_levels = mdp_levels_rets[['Portfolio Level']].rename(columns={'Portfolio Level':'MDP'})
mdp_rets = mdp_levels_rets[['Portfolio Return']].rename(columns={'Portfolio Return':'MDP'})
mdp_rets.index = mdp_rets.index.strftime('%Y-%m')
mdp_rets = pd.merge(mdp_rets, factors['RF'], how='left', left_index = True, right_index=True)
# QC -  Validate weight constraints
mdp_drifted_port.groupby('Date')['Weight'].sum() 
mdp_drifted_port.groupby('Date')['Weight'].count()
mdp_ports['Weight'].max()
mdp_ports['Weight'].min()

# Plot Strategies' Performance
port_levels = pd.concat([mvo_levels, risk_parity_levels, ew_levels, mdp_levels],axis=1).dropna() # Keep common window of performance
hlp.plot_performance(port_levels)

# Plot historical optimal efficient frontier
hlp.plot_efficient_frontier(eff_front.loc[eff_front['Date']==eff_front['Date'].max(),'Return'], eff_front.loc[eff_front['Date']==eff_front['Date'].max(),'Standard Deviation'], eff_front['Date'].max())

# Summary Stats
common_start_date = max([risk_parity_rets.index.min(),mvo_rets.index.min(),ew_rets.index.min(),mdp_rets.index.min()]) # Start the comparison from the first common performance date
risk_parity_stats = hlp.summary_stats(risk_parity_rets.loc[risk_parity_rets.index>=common_start_date,'RiskParity'], risk_parity_rets.loc[risk_parity_rets.index>=common_start_date,'RF']/100, 'RiskParity')
mvo_stats = hlp.summary_stats(mvo_rets.loc[mvo_rets.index>=common_start_date,'MVO'], mvo_rets.loc[mvo_rets.index>=common_start_date,'RF']/100, 'MVO')
ew_stats = hlp.summary_stats(ew_rets.loc[ew_rets.index>=common_start_date,'EW'], ew_rets.loc[ew_rets.index>=common_start_date,'RF']/100, 'EW')
mdp_stats = hlp.summary_stats(mdp_rets.loc[mdp_rets.index>=common_start_date,'MDP'], mdp_rets.loc[mdp_rets.index>=common_start_date,'RF']/100, 'MDP')
stats = pd.concat([mvo_stats, risk_parity_stats, ew_stats, mdp_stats])
# Output formatting
stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']] = round(stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']]*100,2)
stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']].columns = [column + ' (%)' for column in stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']].columns]

# In[6 - Simulated MVO Backtest]    

#Call simulated MVO Backtest
rebal_dates = data.iloc[150:].index # Start building portfolios from 2013-06-28 to ensure timeseries sufficiency for asset covariance matrix
rebal_dates = rebal_dates[rebal_dates.month.isin([3,6,9,12])] # Quarterly Rebals
sim_mvo_ports, eff_front_sim = Rebalancer(data,'MVO_sim', rebal_dates, weight_collar=(0.02,0.40),monte_carlo=True,num_simulations=100) # Call sim_mvo portfolio constructor
sim_mvo_drifted_port, sim_mvo_levels_rets = Backtest(sim_mvo_ports, data) # Calculate portfolio levels/returns & intra review monthly drifted portfolios
sim_mvo_levels = sim_mvo_levels_rets[['Portfolio Level']].rename(columns={'Portfolio Level':'MVO_sim'})
sim_mvo_rets = sim_mvo_levels_rets[['Portfolio Return']].rename(columns={'Portfolio Return':'MVO_sim'})
sim_mvo_rets.index = sim_mvo_rets.index.strftime('%Y-%m')
sim_mvo_rets = pd.merge(sim_mvo_rets, factors['RF'], how='left', left_index = True, right_index=True)
# QC -  Validate weight constraints
sim_mvo_drifted_port.groupby('Date')['Weight'].sum() 
sim_mvo_drifted_port.groupby('Date')['Weight'].count()
sim_mvo_ports['Weight'].max()
sim_mvo_ports['Weight'].min()

# Plot simulated optimal efficient frontier
hlp.plot_efficient_frontier(eff_front_sim.loc[eff_front_sim['Date']==eff_front_sim['Date'].max(),'Return'], eff_front_sim.loc[eff_front_sim['Date']==eff_front_sim['Date'].max(),'Standard Deviation'], eff_front_sim['Date'].max())

# Plot MVO vs MVO_sim Performance
port_levels_sim = pd.concat([mvo_levels, sim_mvo_levels],axis=1).dropna() 
hlp.plot_performance(port_levels_sim)

# Summary Stats MVO vs MVO_sim
mvo_stats = hlp.summary_stats(mvo_rets['MVO'], mvo_rets['RF']/100, 'MVO')
sim_mvo_stats = hlp.summary_stats(sim_mvo_rets['MVO_sim'], sim_mvo_rets['RF']/100, 'MVO_sim')
sim_stats = pd.concat([mvo_stats, sim_mvo_stats])
# Output formatting
sim_stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']] = round(sim_stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']]*100,2)
sim_stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']].columns = [column + ' (%)' for column in sim_stats[['Annualized Return', 'Annualized Vol', 'Historic VaR (5%)', 'Historic CVaR (5%)', 'Max Drawdown']].columns]
