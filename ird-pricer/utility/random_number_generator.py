import numpy as np
from scipy.stats import qmc
from scipy.stats import norm

def gen_normal_rn(shape, dt=1/252):
    return np.random.normal(0, np.sqrt(dt), size=shape)

def gen_uniform_rn(shape, dt=1/252):
    uniform_samples = np.random.uniform(0, 1, size=shape)
    return norm.ppf(uniform_samples) * np.sqrt(dt)

def gen_sobol_rn(shape, dt=1/252):
    sobol = qmc.Sobol(d=shape[0], scramble=True)
    sobol_samples = sobol.random(n=shape[1]).T
    return sobol_samples * np.sqrt(dt)
    #return (sobol_samples - 0.5) * np.sqrt(12 * dt)

def gen_halton_rn(shape, dt=1/252):
    halton = qmc.Halton(d=shape[0], scramble=True)
    halton_samples = halton.random(n=shape[1]).T
    return norm.ppf(halton_samples) * np.sqrt(dt)

def gen_lhs_rn(shape, dt=1/252):
    lhs = qmc.LatinHypercube(d=shape[0])
    lhs_samples = lhs.random(n=shape[1]).T
    return norm.ppf(lhs_samples) * np.sqrt(dt)

def generate_random_numbers(method, shape, dt=1/252):
    if method == "normal":
        return gen_normal_rn(shape, dt)
    elif method == "uniform":
        return gen_uniform_rn(shape, dt)
    elif method == "sobol":
        return gen_sobol_rn(shape, dt)
    elif method == "halton":
        return gen_halton_rn(shape, dt)
    elif method == "lhs":
        return gen_lhs_rn(shape, dt)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'normal', 'uniform', 'sobol', 'halton', 'lhs'.")