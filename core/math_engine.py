# ALL shared math (N(d), greeks, integrals, etc.)
import numpy as np
from scipy.stats import norm

def d1_calc(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2_calc(S, K, T, r, sigma):
    return d1_calc(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_payoff(S, K):
    return np.maximum(S - K, 0)

def put_payoff(S, K):
    return np.maximum(K - S, 0)

def norm_cdf(x):
    return norm.cdf(x)

def norm_pdf(x):
    return norm.pdf(x)
