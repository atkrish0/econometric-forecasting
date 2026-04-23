import numpy as np

def standard_monte_carlo(simulated_forward_rates, strike, discount_factors):
    discount_factors = discount_factors[:, np.newaxis, np.newaxis]
    payoffs = np.maximum(simulated_forward_rates - strike, 0)
    discounted_payoffs = np.sum(payoffs * discount_factors, axis=0)
    return np.mean(discounted_payoffs), np.std(discounted_payoffs)

def monte_carlo_with_antithetic(simulated_forward_rates, strike, discount_factors, initial_forward_rates, dt, volatility, n_steps, n_simulations):
    n_steps, n_maturities, n_simulations = simulated_forward_rates.shape
    discount_factors = discount_factors[:, np.newaxis, np.newaxis]  # Reshape for broadcasting
    payoffs = []

    for j in range(n_simulations):
        dW = np.random.normal(0, np.sqrt(dt), (n_steps, n_maturities)) 
        dW_antithetic = -dW  

        fwd_rate = np.zeros((n_steps, n_maturities))
        fwd_rate_antithetic = np.zeros((n_steps, n_maturities))
        fwd_rate[0, :] = fwd_rate_antithetic[0, :] = initial_forward_rates

        for i in range(1, n_steps):
            fwd_rate[i, :] = fwd_rate[i-1, :] * np.exp(volatility * dW[i, :])
            fwd_rate_antithetic[i, :] = fwd_rate_antithetic[i-1, :] * np.exp(volatility * dW_antithetic[i, :])
        
        payoff = np.maximum(fwd_rate - strike, 0)
        payoff_antithetic = np.maximum(fwd_rate_antithetic - strike, 0)
        
        discounted_payoff = np.sum(payoff * discount_factors)
        discounted_payoff_antithetic = np.sum(payoff_antithetic * discount_factors)
        
        average_payoff = 0.5 * (discounted_payoff + discounted_payoff_antithetic)
        payoffs.append(average_payoff)
    
    return np.mean(payoffs), np.std(payoffs)

def monte_carlo_with_control_variates(simulated_forward_rates, strike, discount_factors, analytical_price):
    discount_factors = discount_factors[:, np.newaxis, np.newaxis]
    payoffs = np.maximum(simulated_forward_rates - strike, 0)
    discounted_payoffs = np.sum(payoffs * discount_factors, axis=0)
    
    mc_price = np.mean(discounted_payoffs)
    control_correction = analytical_price - np.mean(analytical_price)
    adjusted_price = mc_price + control_correction
    return adjusted_price, np.std(discounted_payoffs)

def price_with_variance_reduction(method, simulated_forward_rates, strike, discount_factors, analytical_price=None, **kwargs):
    if method == "standard":
        return standard_monte_carlo(simulated_forward_rates, strike, discount_factors)
    elif method == "antithetic":
        return monte_carlo_with_antithetic(
            simulated_forward_rates,
            strike,
            discount_factors,
            kwargs.get("initial_forward_rates"),
            kwargs.get("dt"),
            kwargs.get("volatility"),
            kwargs.get("n_steps"),
            kwargs.get("n_simulations"),
        )
    elif method == "control":
        if analytical_price is None:
            raise ValueError("Analytical price is required for control variates.")
        return monte_carlo_with_control_variates(simulated_forward_rates, strike, discount_factors, analytical_price)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'standard', 'antithetic', 'control'.")