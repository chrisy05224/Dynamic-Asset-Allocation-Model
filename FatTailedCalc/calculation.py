import numpy as np
from scipy.optimize import fsolve, minimize
import yfinance as yf
from scipy.stats import norm, scoreatpercentile, levy_stable
import matplotlib.pyplot as plt


def calibrate_gamma_from_volatility(alpha, beta, target_volatility, T=1.0, n_samples=100000):
# For alpha=2, we have a direct relationship: variance = 2 * gamma^2
    if alpha == 2.0:
        return target_volatility / np.sqrt(2 * 252)  # Daily scale for 252 trading days
    
    # For alpha < 2, we need to calibrate numerically
    def volatility_error(gamma):
        # Generate stable returns
        returns = levy_stable.rvs(alpha, beta, scale=gamma, size=n_samples)
        
        # Calculate annualized volatility from samples
        sample_volatility = np.std(returns) * np.sqrt(252)
        return abs(sample_volatility - target_volatility)
    
    # Initial guess based on empirical relationships
    initial_gamma = target_volatility / (10 * np.sqrt(252))  # Conservative guess
    
    result = minimize(volatility_error, initial_gamma, method='Nelder-Mead', 
                     options={'maxiter': 50, 'xatol': 1e-6})
    
    if not result.success:
        print(f"Warning: Gamma calibration failed for α={alpha}, using fallback")
        return initial_gamma
    
    return result.x[0]

def levy_characteristic_function(u, alpha, beta, gamma, delta):
    """
    Should be more accurate.
        """
    abs_u = np.maximum(np.abs(u), 1e-12)
    if alpha == 1.0:
        # Cauchy-like case
        term = -gamma * abs_u * (1 + 1j * beta * (2/np.pi) * np.sign(u) * np.log(abs_u))
    else:
        # General case (0 < α < 2)
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

def levy_otm_to_itm_expectation(S0, K, T, r, q, alpha, beta, gamma, n_points=2000, U_max=50):
    # Enforce martingale condition to get correct delta
    delta = martingale_condition(alpha, beta, gamma, T, r)
    
    # Characteristic function of log-price s_T = ln(S_T)
    def log_price_char_func(u):
        mean_term = np.log(S0) + (r - q) * T
        return np.exp(1j * u * mean_term) * levy_characteristic_function(u, alpha, beta, gamma, delta)
    
    # Fourier inversion for probability P(S_T > K)
    k = np.log(K / S0)  # log-moneyness
    
    def integrand(u):
        if np.abs(u) < 1e-10:  # Handle singularity at u=0
            return 0.5
        char_val = log_price_char_func(u)
        return np.imag(np.exp(-1j * u * k) * char_val / (1j * u))
    
    # Numerical integration
    u_values = np.linspace(1e-6, U_max, n_points)  # Avoid u=0
    integrand_values = [integrand(u) for u in u_values]
    
    # Calculate probability using trapezoidal rule
    probability = 0.5 - (1/np.pi) * np.trapezoid(integrand_values, u_values)
    
    # Ensure probability is valid
    probability = np.clip(probability, 0.0, 1.0)
    
    return probability

# Another sanity check
def analytical_black_scholes_probability(S0, K, T, r, q, sigma):
    """Analytical probability for normal distribution (alpha=2)"""
    d2 = (np.log(S0/K) + (r - q - 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d2)

def levy_expectation(S0, K, T, r, q, alpha, beta, annual_volatility, n_samples=100000):
    """
    CORRECT implementation with proper volatility calibration.
    Returns both the probability and the calibrated gamma.
    """
    # Step 1: Calibrate gamma to match the target annual volatility
    gamma = calibrate_gamma_from_volatility(alpha, beta, annual_volatility, T, n_samples//10)
    
    # Step 2: Get proper delta for martingale condition
    delta = martingale_condition(alpha, beta, gamma, T, r)
    
    # Step 3: Generate terminal prices
    if alpha == 2.0:
        # Normal distribution
        X = np.random.normal(delta, gamma * np.sqrt(2), n_samples)
    else:
        # Lévy stable distribution
        X = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=n_samples)
    
    # Calculate terminal price (accounting for dividends)
    S_T = S0 * np.exp((r - q) * T + X)
    
    # Calculate probability
    probability = np.mean(S_T > K)
    
    return probability, gamma

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

## Debug cases
def run_corrected_tests():
    """Run tests with proper volatility calibration"""
    print("PROPERLY CALIBRATED Lévy Expectation Implementation")
    print("=" * 65)
    
    S0, T, r, q = 100.0, 1.0, 0.05, 0.02
    annual_volatility = 0.20  # 20% annual volatility
    
    test_cases = [
        # (K, alpha, beta, expected_range, description)
        (105, 2.0, 0.0, (0.35, 0.45), "OTM Call - Normal (should match BS)"),
        (95, 2.0, 0.0, (0.55, 0.65), "ITM Call - Normal (should match BS)"),
        (105, 1.7, -0.3, (0.38, 0.50), "OTM Call - Fat Tails"),
        (115, 1.7, -0.3, (0.15, 0.30), "Deep OTM - Fat Tails"),
        (80, 1.7, -0.3, (0.85, 0.98), "Deep ITM - Fat Tails"),
    ]
    
    for K, alpha, beta, expected_range, description in test_cases:
        # Calculate expectation with proper calibration
        result = levy_expectation(S0, K, T, r, q, alpha, beta, annual_volatility)
        expectation, calibrated_gamma = result  # Now this should work
        
        # For alpha=2, compare with analytical Black-Scholes
        if alpha == 2.0:
            bs_prob = analytical_black_scholes_probability(S0, K, T, r, q, annual_volatility)
            diff = abs(expectation - bs_prob)
            print(f"{description}:")
            print(f"  Lévy: {expectation:.6f}, BS: {bs_prob:.6f}, Diff: {diff:.6f}")
            print(f"  Calibrated gamma: {calibrated_gamma:.6f}")
        else:
            bs_prob = analytical_black_scholes_probability(S0, K, T, r, q, annual_volatility)
            diff = abs(expectation - bs_prob)
            print(f"{description}:")
            print(f"  Lévy: {expectation:.6f}, BS: {bs_prob:.6f}, Diff: {diff:.6f}")
            print(f"  Calibrated gamma: {calibrated_gamma:.6f}")
        
        # Validate reasonableness
        if not (0 <= expectation <= 1):
            print(f"  ❌ ERROR: Invalid probability {expectation:.6f}")
        elif expectation < expected_range[0] or expectation > expected_range[1]:
            print(f"  ⚠️  SUSPICIOUS: {expectation:.6f} outside expected range {expected_range}")
        else:
            print(f"  ✅ REASONABLE: {expectation:.6f} in expected range {expected_range}")
        
        print()

def simple_demonstration():
    """Simple demonstration without the complex calibration"""
    print("SIMPLE DEMONSTRATION")
    print("=" * 40)
    
    S0, K, T, r, q = 100.0, 105.0, 1.0, 0.05, 0.02
    annual_volatility = 0.20
    
    # Test normal case (alpha=2) - should match Black-Scholes
    alpha, beta = 2.0, 0.0
    expectation, gamma = levy_expectation(S0, K, T, r, q, alpha, beta, annual_volatility, 50000)
    bs_prob = analytical_black_scholes_probability(S0, K, T, r, q, annual_volatility)
    
    print(f"Normal case (α=2):")
    print(f"  Lévy: {expectation:.6f}")
    print(f"  BS:   {bs_prob:.6f}")
    print(f"  Diff: {abs(expectation - bs_prob):.6f}")
    print(f"  Gamma: {gamma:.6f}")
    print()
    
    # Test fat-tailed case
    alpha, beta = 1.7, -0.3
    expectation, gamma = levy_expectation(S0, K, T, r, q, alpha, beta, annual_volatility, 50000)
    
    print(f"Fat-tailed case (α=1.7):")
    print(f"  Lévy: {expectation:.6f}")
    print(f"  BS:   {bs_prob:.6f}") 
    print(f"  Diff: {abs(expectation - bs_prob):.6f}")
    print(f"  Gamma: {gamma:.6f}")

if __name__ == "__main__":
    # Run simple demonstration first
    simple_demonstration()
    
    print("\n" + "=" * 65)
    print("Note: The calibration might take a moment for α < 2 cases...")
    
    # Then run the full test suite
    try:
        run_corrected_tests()
    except Exception as e:
        print(f"Error in full test suite: {e}")
        print("Running simplified version instead...")
        simple_demonstration()

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

