import numpy as np
from scipy.optimize import fsolve
import yfinance as yf
from scipy.stats import scoreatpercentile

def levy_characteristic_function(u, alpha, beta, gamma, delta):
    """
    Should be more accurate.
    """
    if alpha == 1.0:
        term = -gamma * np.abs(u) * (1 + 1j * beta * (2/np.pi) * np.sign(u) * np.log(np.abs(u)))
        return np.exp(1j * delta * u + term)
    else:
        term = - (gamma * np.abs(u)) ** alpha * (1 - 1j * beta * np.sign(u) * np.tan(np.pi * alpha / 2))
        return np.exp(1j * delta * u + term)


def enforce_martingale_condition(alpha, beta, gamma, T, r):
    """
    Enforces the martingale condition, even if assuming that prices are not independent, mostly for Arbitrage Prevention reasons.
    """
    # For the martingale condition, we need phi(-i) = exp(r*T)
    def condition(delta):
        char_value = levy_characteristic_function(-1j, alpha, beta, gamma, delta)
        return np.real(char_value) - np.exp(r * T)
    
    # Solve for delta
    delta_guess = r * T  # Reasonable initial guess
    delta_solution = fsolve(condition, delta_guess, full_output=True)
    
    if delta_solution[2] != 1:  # Check if solution converged
        print(f"Warning: Martingale condition solver did not converge. Using fallback.")
        return r * T - 0.5 * gamma**2 * T  # Fallback to normal approximation
    
    return delta_solution[0][0]

def levy_otm_to_itm_expectation(S0, K, T, r, q, alpha, beta, gamma, n_points=10000):
    """
    Calculate the risk-neutral expectation that an OTM option becomes ITM under Lévy-alpha model.
    This calculates E[1_{S_T > K}] for a call option using Fourier inversion.
    
    Parameters:
    S0 : float - Current spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    q : float - Dividend yield
    alpha, beta, gamma : float - Lévy distribution parameters
    n_points : int - Number of integration points
    
    Returns:
    float: Risk-neutral expectation E[1_{S_T > K}]
    """
    # ===== 1. ENFORCE MARTINGALE CONDITION =====
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
    
    # ===== 2. CHARACTERISTIC FUNCTION OF LOG-PRICE =====
    def log_price_characteristic_function(v):
        phase = 1j * v * (np.log(S0) + (r - q) * T)
        return np.exp(phase) * levy_characteristic_function(v, alpha, beta, gamma, delta_solution)
    
    # ===== 3. CALCULATE EXPECTATION USING FOURIER INVERSION =====
    # Expectation E[1_{S_T > K}] = P(S_T > K) under risk-neutral measure
    k = np.log(K / S0)  # log-moneyness
    
    def expectation_integrand(u):
        """Integrand for expectation calculation using Fourier inversion"""
        if u == 0:
            return 0.5  # Handle the singularity at u=0
        char_func_val = log_price_characteristic_function(u)
        numerator = np.exp(-1j * u * k) * char_func_val
        return np.imag(numerator / (1j * u))
    
    # Create integration grid (avoid u=0)
    u_min, u_max = 1e-10, 1000
    u_values = np.linspace(u_min, u_max, n_points)
    
    # Evaluate integrand
    integrand_values = np.array([expectation_integrand(u) for u in u_values])
    
    # Calculate expectation using trapezoidal integration
    expectation = 0.5 - (1/np.pi) * np.trapezoid(integrand_values, u_values)
    
    return expectation

def levy_call_price(S0, K, T, r, q, alpha, beta, gamma, U_max=100, n_points=5000):
    """
    Price a European call option using the Levy stable model via Fourier transform.
    Uses simple numerical integration with trapezoidal rule.
    """
    # ===== 1. ENFORCE MARTINGALE CONDITION =====
    def characteristic_at_neg_i(delta_test):
        return levy_characteristic_function(-1j, alpha, beta, gamma, delta_test)
    
    def equation_to_solve(delta_val):
        char_value = characteristic_at_neg_i(delta_val)
        return np.real(char_value) - np.exp(r * T)
    
    # Initial guess based on distribution type
    if alpha == 2:
        delta_initial_guess = (r - 0.5 * gamma**2) * T
    else:
        delta_initial_guess = r * T
    
    delta_solution = fsolve(equation_to_solve, delta_initial_guess)[0]
    
    # ===== 2. CHARACTERISTIC FUNCTION OF LOG-PRICE =====
    def log_price_characteristic_function(v):
        phase = 1j * v * (np.log(S0) + (r - q) * T)
        return np.exp(phase) * levy_characteristic_function(v, alpha, beta, gamma, delta_solution)
    
    # ===== 3. FOURIER INTEGRATION (Lewis-Lipton Formula) =====
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



# ===== EXAMPLE USAGE =====
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
    mle_params = quantile_estimation(returns)
    quantile_params = quantile_estimation(returns)

    print("MLE Estimates:", mle_params)
    print("Quantile Estimates:", quantile_params) 

    # Levy stable distribution parameters
    alpha, beta, gamma = mle_params['alpha'], mle_params['beta'], mle_params['gamma']

    
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
