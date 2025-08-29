import numpy as np
from scipy.optimize import fsolve, minimize
import yfinance as yf
from scipy.stats import norm, scoreatpercentile, levy_stable
import matplotlib.pyplot as plt


def levy_expectation(S0, K, T, r, q, alpha, beta, annual_volatility, n_samples=100000):
    gamma = 0.015  # 1.5% daily scale (reasonable for equities)
    
    # CRITICAL: Proper risk-neutral drift adjustment
    # For Lévy stable, the martingale condition requires:
    # E[exp(X)] = exp(rT) where X ~ S(alpha, beta, gamma, delta; 0)
    
    if alpha == 2.0:
        # Normal distribution case
        delta = (r - q - 0.5 * (gamma*np.sqrt(2))**2) * T
        X = np.random.normal(delta, gamma * np.sqrt(2), n_samples)
    else:
        # Lévy stable case - we need to find delta such that E[exp(X)] = exp((r-q)T)
        # This is complex, so we use a reasonable approximation
        if alpha == 1.0:
            # Cauchy distribution
            delta = (r - q) * T - (2/np.pi) * gamma * beta * np.log(T)
        else:
            # General stable case - use first order approximation
            delta = (r - q) * T - gamma**alpha * (1 / np.cos(np.pi * alpha / 2)) * T
        
        # Generate stable random variables
        X = levy_stable.rvs(alpha, beta, scale=gamma, size=n_samples)
        X += delta  # Add the drift component
    
    # Calculate terminal prices
    S_T = S0 * np.exp(X)
    
    # Calculate probability
    probability = np.mean(S_T > K)
    
    return probability


def levy_characteristic_function(u, alpha, beta, gamma, delta):
    """
    Should be more accurate.
        """
    abs_u = np.maximum(np.abs(u), 1e-12)
    if alpha == 1.0:
        term = -gamma * abs_u * (1 + 1j * beta * (2/np.pi) * np.sign(u) * np.log(abs_u))
    else:
        term = -np.power(gamma * abs_u, alpha) * (1 - 1j * beta * np.sign(u) * np.tan(np.pi * alpha / 2))
    
    return np.exp(1j * delta * u + term)

def martingale_condition(alpha, beta, gamma, T, r):
    """
    Enforces the martingale condition, even if assuming that prices are not independent, mostly for Arbitrage Prevention reasons. Quiet possibly to eliminate a free option. 
    """
    if alpha == 2.0:
        # Normal case: delta = (r - 0.5*sigma^2)*T
        # But remember: for alpha=2, variance = 2*gamma^2
        variance = 2 * gamma**2
        return (r - 0.5 * variance) * T
    elif alpha == 1.0:
        # Cauchy case: special handling
        return r * T - (2/np.pi) * gamma * beta * T
    else:
        # General stable case
        return r * T - gamma**alpha * (1 / np.cos(np.pi * alpha / 2)) * np.tan(np.pi * alpha / 2) * beta * T

# Another sanity check
def black_scholes_probability(S0, K, T, r, q, sigma):
    """Analytical probability for normal distribution (alpha=2)"""
    d2 = (np.log(S0/K) + (r - q - 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d2)

def levy_call_price(S0, K, T, r, q, alpha, beta, gamma, U_max=100, n_points=5000):
    """
    Price a European call option using the Levy stable model via Fourier transform.
    Uses simple numerical integration with trapezoidal rule.
    """
    def characteristic_at_neg_i(delta_test):
        return levy_characteristic_function(-1j, alpha, beta, gamma, delta_test)
    
    def equation_to_solve(delta_val):
        char_value = characteristic_at_neg_i(delta_val)
        return np.real(char_value) - np.exp(r * T)
    
    if alpha == 2:
        delta_initial_guess = (r - 0.5 * gamma**2) * T
    else:
        delta_initial_guess = r * T
    
    delta_solution = fsolve(equation_to_solve, delta_initial_guess)[0]
    
    def log_price_characteristic_function(v):
        phase = 1j * v * (np.log(S0) + (r - q) * T)
        return np.exp(phase) * levy_characteristic_function(v, alpha, beta, gamma, delta_solution)
    
    k = np.log(K / S0)  # log-moneyness
    
    def integrand(u):
        """
        Complete integrand function for the Lewis-Lipton formula
        """
        char_func_val = log_price_characteristic_function(u - 0.5j)
        numerator = np.exp(-1j * u * k) * char_func_val
        denominator = u**2 + 0.25
        return numerator / denominator
    
    # Create integration grid
    u_values = np.linspace(-U_max, U_max, n_points)
    
    # Evaluate integrand at all points
    integrand_values = np.array([integrand(u) for u in u_values])
    
    # Separate real and imaginary parts
    real_integrand = np.real(integrand_values)
    imag_integrand = np.imag(integrand_values)
    
    # Integrate using trapezoidal rule (numpy.trapezoid instead of trapz)
    real_integral = np.trapezoid(real_integrand, u_values)
    imag_integral = np.trapezoid(imag_integrand, u_values)
    
    integral_value = real_integral + 1j * imag_integral
    
    # ===== 4. COMPUTE CALL PRICE =====
    pre_factor = np.sqrt(S0 * K) * np.exp(-0.5 * (r + q) * T) / (2 * np.pi)
    integral_part = pre_factor * integral_value
    
    # The call price formula
    call_price = S0 * np.exp(-q * T) - np.real(integral_part)
    
    return call_price, delta_solution

def quantile_estimation(returns):
    """
    Estimates Lévy stable parameters using McCulloch's quantile method.
    """
    
    # Calculate required quantiles
    x_05 = scoreatpercentile(returns, 5)
    x_25 = scoreatpercentile(returns, 25)
    x_50 = scoreatpercentile(returns, 50)  # median
    x_75 = scoreatpercentile(returns, 75)
    x_95 = scoreatpercentile(returns, 95)
    
    # Calculate quantile ratios
    phi_1 = (x_95 - x_05) / (x_75 - x_25)
    phi_2 = (x_95 + x_05 - 2*x_50) / (x_95 - x_05)
    
    # Estimate alpha (using approximation)
    if phi_1 <= 2.439:
        alpha_est = 0.10 + (2.439 - phi_1) / 0.8
    else:
        alpha_est = 2.0 - (phi_1 - 2.439) / 0.5
    
    alpha_est = np.clip(alpha_est, 1.1, 2.0)  # Constrain to reasonable range
    
    # Estimate beta
    if alpha_est < 1.7:
        beta_est = 5.5 * (phi_2 + 0.25) * (alpha_est - 1.7) + phi_2
    else:
        beta_est = phi_2
    
    beta_est = np.clip(beta_est, -1.0, 1.0)  # Constrain to [-1, 1]
    
    # Estimate gamma (scale)
    if alpha_est < 1.7:
        gamma_est = (x_75 - x_25) / (1.654 + 0.288*alpha_est - 0.640*(alpha_est-1.7))
    else:
        gamma_est = (x_75 - x_25) / (1.654 + 0.288*alpha_est)
    
    # Estimate delta (location)
    delta_est = x_50 + gamma_est * beta_est * np.tan(np.pi * alpha_est / 2)
    
    return {
        'alpha': alpha_est,
        'beta': beta_est,
        'gamma': gamma_est,
        'delta': delta_est
    }

## Debug cases
def run_sensible_tests():
    """Run tests with financially sensible parameters"""
    print("FINANCIALLY SENSIBLE Lévy Expectation")
    print("=" * 50)
    
    S0, T, r, q = 100.0, 1.0, 0.05, 0.02
    
    test_cases = [
        # (K, alpha, beta, description)
        (105, 2.0, 0.0, "OTM Call - Normal (Black-Scholes)"),
        (95, 2.0, 0.0, "ITM Call - Normal (Black-Scholes)"),
        (105, 1.8, -0.2, "OTM Call - Mild Fat Tails"),
        (105, 1.6, -0.3, "OTM Call - Fat Tails"),
        (105, 1.4, -0.4, "OTM Call - Very Fat Tails"),
        (115, 1.7, -0.3, "Deep OTM - Fat Tails"),
        (80, 1.7, -0.3, "Deep ITM - Fat Tails"),
    ]
    
    for K, alpha, beta, description in test_cases:
        probability = levy_expectation(S0, K, T, r, q, alpha, beta, 50000)
        
        # For comparison, show Black-Scholes with 20% volatility
        bs_prob = black_scholes_probability(S0, K, T, r, q, 0.20)
        
        print(f"{description}:")
        print(f"  K={K}, S0={S0} → Moneyness: {'OTM' if K > S0 else 'ITM'}")
        print(f"  Lévy (α={alpha}, β={beta}): {probability:.3f}")
        
        if alpha == 2.0:
            print(f"  Black-Scholes: {bs_prob:.3f}")
            print(f"  Difference: {abs(probability - bs_prob):.3f}")
        else:
            print(f"  Black-Scholes: {bs_prob:.3f} (reference)")
            print(f"  Excess probability: {probability - bs_prob:+.3f}")
        
        # Sanity checks
        if K > S0 and probability > 0.5:
            print(f"  ⚠️  Warning: OTM call probability > 0.5")
        elif K < S0 and probability < 0.5:
            print(f"  ⚠️  Warning: ITM call probability < 0.5")
        
        print()

def demonstrate_fat_tail_effects():
    """Show how fat tails affect OTM option probabilities"""
    print("FAT TAIL EFFECTS ON OTM OPTIONS")
    print("=" * 45)
    
    S0, K, T, r, q = 100.0, 110.0, 0.25, 0.05, 0.02  # 3-month OTM call
    
    alphas = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
    beta = -0.3  # Negative skew (typical for equities)
    
    bs_prob = black_scholes_probability(S0, K, T, r, q, 0.20)
    print(f"Black-Scholes (α=2.0) probability: {bs_prob:.4f}")
    print()
    
    print("Alpha\tProbability\tExcess vs BS")
    print("-" * 35)
    
    for alpha in alphas:
        prob = levy_expectation(S0, K, T, r, q, alpha, beta, 30000)
        excess = prob - bs_prob
        print(f"{alpha}\t{prob:.4f}\t\t{excess:+.4f}")

if __name__ == "__main__":
    # Run sensible tests
    run_sensible_tests()
    
    print("\n" + "=" * 50)
    demonstrate_fat_tail_effects()
    
    print("\n" + "=" * 50)
    print("EXPECTED PATTERNS:")
    print("1. OTM calls (K > S0): probability < 0.5")
    print("2. ITM calls (K < S0): probability > 0.5") 
    print("3. Fat tails (α < 2): increase OTM probabilities")
    print("4. Negative skew (β < 0): affects left tail more")
    print("5. Values should be financially reasonable")


'''
# Where
# S0: Current spot price of the underlying asset
# K: Strike price of the option contract
# T: Time to expiration (in years)
# r: Risk-free interest rate (annualized)
# q: Continuous dividend yield (annualized)
# alpha: Tail index (0 < α ≤ 2) - lower values = fatter tails
# beta: Skewness parameter (-1 ≤ β ≤ 1) - negative = left skew, positive = right skew
# gamma: Scale parameter (γ > 0) - similar to volatility, controls distribution width
# n_points: Number of integration points for numerical computation
# U_max: Maximum integration range for Fourier transform

# Main usage
if __name__ == "__main__":

    # Option parameters
    S0 = 23140    # Spot price
    K = 19000     # Strike price (OTM call)
    T = 0.167       # Time to maturity (1 year)
    r = .04399      # Risk-free rate (5%)
    q = 0.02      # Dividend yield (2%)
   # Download historical data
    ticker = "SPY"
    data = yf.download(ticker, period="50y", interval="1d")
    prices = data['Close']
    returns = np.log(prices / prices.shift(1)).dropna().values 
    quantile_params = quantile_estimation(returns)

    print("Quantile Estimates:", quantile_params) 

    # Levy stable distribution parameters
    alpha, beta, gamma = quantile_params['alpha'], quantile_params['beta'], quantile_params['gamma']

    
    print("Lévy Alpha Model - OTM to ITM Expectation:")
    print("=" * 50)
    
    # Calculate the expectation
    expectation = levy_otm_to_itm_expectation(S0, K, T, r, q, alpha, beta, gamma)
    
    print(f"Option Parameters:")
    print(f"  S0: {S0}, K: {K}, T: {T}, r: {r}, q: {q}")
    print(f"Lévy Parameters: α={alpha}, β={beta}, γ={gamma}")
    print()
    print(f"Risk-Neutral Expectation E[1_{{S_T > K}}]: {expectation:.6f}")
    print(f"Interpretation: Probability that OTM call becomes ITM = {expectation:.4f}")
    
    # Also calculate the call price for reference
    call_price, delta = levy_call_price(S0, K, T, r, q, alpha, beta, gamma)
    print(f"Lévy Call Option Price: {call_price:.6f}")
    print(f"Calculated Delta Parameter: {delta:.6f}")
    
    # Show how expectation varies with different parameters
    print("\n" + "=" * 50)
    print("Expectation Sensitivity Analysis:")
    print("α\tβ\tγ\tE[1_{S_T > K}]")
    print("-" * 40) # immediately
    
    # Test different alpha values
    for alpha_test in [1.5, 1.7, 1.9, 2.0]:
        expectation_test = levy_otm_to_itm_expectation(S0, K, T, r, q, alpha_test, beta, gamma, n_points=5000)
        print(f"{alpha_test}\t{beta}\t{gamma}\t{expectation_test:.6f}")
    
    # Test different beta values
    print()
    for beta_test in [-0.5, -0.3, 0.0, 0.3]:
        expectation_test = levy_otm_to_itm_expectation(S0, K, T, r, q, alpha, beta_test, gamma, n_points=5000)
        print(f"{alpha}\t{beta_test}\t{gamma}\t{expectation_test:.6f}")
    
    # Test different strikes to show moneyness effect
    print("\n" + "=" * 50)
    print("Moneyness Effect (Different Strikes):")
    print("K\tMoneyness\tE[1_{S_T > K}]")
    print("-" * 40)
    
    strikes = [90, 95, 100, 105, 110, 115] ## Fix this with the proper data.
    for strike in strikes:
        moneyness = "ITM" if strike < S0 else "ATM" if strike == S0 else "OTM"
        expectation_val = levy_otm_to_itm_expectation(S0, strike, T, r, q, alpha, beta, gamma, n_points=5000)
        print(f"{strike}\t{moneyness}\t\t{expectation_val:.6f}")
''' 

