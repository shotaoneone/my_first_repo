#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:38:57 2024

@author: ichitakushouta
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
r = 0
S0 = 100
sigma = 0.25
T = 1  # Time to maturity (let's assume 1 year)
K = 100  # Strike price
N = 10000  # Number of simulations
M = 252  # Number of time steps (assuming 252 trading days in a year)

def simulate_paths():
    dt = T / M
    paths = np.zeros((N, M+1))
    paths[:, 0] = S0
    
    for i in range(1, M+1):
        z = np.random.normal(0, 1, N)
        paths[:, i] = paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

def european_put_payoff(S_T, K):
    return np.maximum(K - S_T, 0)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Simulate paths
paths = simulate_paths()

# (a) Calculate mean and variance of terminal values
terminal_values = paths[:, -1]
mean_terminal = np.mean(terminal_values)
var_terminal = np.var(terminal_values)

print(f"(a) Mean of terminal values: {mean_terminal:.2f}")
print(f"    Variance of terminal values: {var_terminal:.2f}")

# (b) Calculate put option payoffs
payoffs = european_put_payoff(terminal_values, K)
mean_payoff = np.mean(payoffs)
std_payoff = np.std(payoffs)

print(f"(b) Mean of put option payoffs: {mean_payoff:.2f}")
print(f"    Standard deviation of put option payoffs: {std_payoff:.2f}")

# Plot histogram of payoffs
plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, edgecolor='black')
plt.title("Histogram of European Put Option Payoffs")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.show()

# (c) Calculate simulated option price
simulated_price = np.mean(payoffs) * np.exp(-r * T)
print(f"(c) Simulated European put option price: {simulated_price:.2f}")

# (d) Calculate Black-Scholes price
bs_price = black_scholes_put(S0, K, T, r, sigma)
print(f"(d) Black-Scholes European put option price: {bs_price:.2f}")
print(f"    Difference: {abs(simulated_price - bs_price):.4f}")