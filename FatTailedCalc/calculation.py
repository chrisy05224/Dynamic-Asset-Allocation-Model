import numpy as np
import yfinance as yf
from scipy.stats import scoreatpercentile, levy_stable

def martingale_condition(alpha, beta, gamma, T, r, q=0.0):
    ##     Enforces the martingale condition, even if assuming that prices are not independent, mostly for Arbitrage Prevention reasons. Quiet possibly to eliminate a free option.     
    if alpha == 2.0:
        sigma = gamma * np.sqrt(2 * T)
        result = (r - q - 0.5 * sigma**2) * T
        return result
        
    elif alpha == 1.0:
        result = (r - q) * T - (2/np.pi) * gamma * beta * T
        return result
        
    else:
        term = (gamma**alpha) / np.cos(np.pi * alpha / 2)
        result = (r - q) * T - term * np.tan(np.pi * alpha / 2) * beta * T
        return result

def levy_expectation(S0, K, T, r, q, alpha, beta, gamma, n_samples=100000):
    delta = martingale_condition(alpha, beta, gamma, T, r, q)
    
    if alpha == 2.0:
        sigma = gamma * np.sqrt(2) * np.sqrt(T)  # Proper annualization
        X = np.random.normal(delta, sigma, n_samples)
        
    else:
        scaled_gamma = gamma * (T ** (1/alpha))
        X = levy_stable.rvs(alpha, beta, loc=delta, scale=scaled_gamma, size=n_samples)
    
    S_T = S0 * np.exp(X)
    probability = np.mean(S_T > K)
    
    return probability, delta



def quantile_estimation(returns):
    """
    Estimates Lévy stable parameters using McCulloch's quantile method.
    """
    
    x_05 = scoreatpercentile(returns, 5)
    x_25 = scoreatpercentile(returns, 25)
    x_50 = scoreatpercentile(returns, 50)  # median
    x_75 = scoreatpercentile(returns, 75)
    x_95 = scoreatpercentile(returns, 95)
    
    phi_1 = (x_95 - x_05) / (x_75 - x_25)
    phi_2 = (x_95 + x_05 - 2*x_50) / (x_95 - x_05)
    
    if phi_1 <= 2.439:
        alpha_est = 0.10 + (2.439 - phi_1) / 0.8
    else:
        alpha_est = 2.0 - (phi_1 - 2.439) / 0.5
    
    alpha_est = np.clip(alpha_est, 1.1, 2.0)  # Constrain to reasonable range
    
    if alpha_est < 1.7:
        beta_est = 5.5 * (phi_2 + 0.25) * (alpha_est - 1.7) + phi_2
    else:
        beta_est = phi_2
    
    beta_est = np.clip(beta_est, -1.0, 1.0)  # Constrain to [-1, 1]
    
    if alpha_est < 1.7:
        gamma_est = (x_75 - x_25) / (1.654 + 0.288*alpha_est - 0.640*(alpha_est-1.7))
    else:
        gamma_est = (x_75 - x_25) / (1.654 + 0.288*alpha_est)
    
    delta_est = x_50 + gamma_est * beta_est * np.tan(np.pi * alpha_est / 2)
    
    return {
        'alpha': alpha_est,
        'beta': beta_est,
        'gamma': gamma_est,
        'delta': delta_est
    }

'''
## Debug cases
if __name__ == "__main__":

    ticker = "SPY"
    data = yf.download(ticker, period="50y", interval="1d")
    prices = data['Close']
    returns = np.log(prices / prices.shift(1)).dropna().values 
        quantile_params = quantile_estimation(returns)

    # Your estimated parameters
    alpha = quantile_params['alpha']
    beta = quantile_params['beta']
    gamma = quantile_params['gamma']
    
    # Call the integrated function
    expectation, calculated_delta = levy_expectation(
        S0=S0, 
        K=K, 
        T=T, 
        r=r, 
        q=q, 
        alpha=alpha, 
        beta=beta, 
        gamma=gamma,
        n_samples=100000
    )
    print(f"Martingale drift adjustment δ: {calculated_delta:.6f}")
    print(f"Monte Carlo Expectation E[1_{{S_T > K}}]: {expectation:.6f}")
    print(f"Probability OTM call becomes ITM: {expectation:.2%}") 
'''
