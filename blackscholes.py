import numpy as np
import math
from scipy.stats import norm

class BlackScholes:
    
    value = 0.0
    probability = 0.0
    delta = 0.0
    gamma = 0.0
    theta = 0.0
    vega = 0.0
    rho = 0.0
    
    def __init__(self, S, K, T, r, sigma, q, type):
        self.S = S # spot price
        self.K = K # strike price
        self.T = T # time in years until expiration
        self.r = r # risk-free rate
        self.sigma = sigma # volatility of underlying, usually the annualized standard deviation
        self.q = q # annualized dividend yield
        self.type = type # the type of option, either 'call' or 'put'
        
        if self.T > 0 and self.sigma > 0:
            self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(self.T))
            self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        else:
            self.d1 = float('inf') if self.S > self.K else float('-inf')
            self.d2 = float('inf') if self.S > self.K else float('-inf')
        
    
    def call_price(self):
        '''
        Calculates the price of a call option.
        '''
        if self.T <= 0:
            return max(0.0, self.S - self.K)
        
        call_price = (self.S * math.exp(-self.q * self.T) * norm.cdf(self.d1) - 
                self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2))
        
        return call_price
    
    def put_price(self):
        '''
        Calculates the price of a put option.
        '''
        if self.T <= 0:
            return max(0.0, self.K - self.S)
            
        put_price = (self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
               self.S * math.exp(-self.q * self.T) * norm.cdf(-self.d1))
        
        return put_price
    
    def price(self):
        '''
        Prices the option by calling the appropriate pricing functions.
        '''
        if self.type == 'call':
            return self.call_price()
        elif self.type == 'put':
            return self.put_price()
        else:
            return 'Error! Type of option not stated properly.'
    
    def probability_of_exercise(self):
        '''
        Risk-neutral probability of the option expiring in the money. 
        This assumes that the probability of exercising the option is N(d2) for call options and N(-d2) for put options. This is according to the BPP University Actuarial course.
        '''
        if self.type == 'call':
            return norm.cdf(self.d2)
        elif self.type == 'put':
            return norm.cdf(-self.d2)
        
    def expectation(self):
        '''
        Calculates the expectation of the price, the price * the 
        '''
        return self.price() * self.probability_of_exercise()
        
# Example Usage
# bs = BlackScholes(100.0, 50.0, 1.0, 0.05, 0.1, 0.05, 'call')
# price = bs.price()
# print(f"Call Option Price: {round(price, 2)}")
# expectation = bs.expectation()
# print(f"Call Option Expectation: {round(expectation, 2)}")