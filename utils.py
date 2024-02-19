import numpy as np 
from numba import njit

@njit
def clip(x, c):
    x_norm = np.sqrt(x @ x)
    if x_norm < c: 
        return x
    return x * c / x_norm


def clip_all(x, c):
    norms = np.sqrt(np.sum(x**2, axis=1))
    renorm = np.array([min(1, c/n) for n in norms])
    return (renorm * x.T).T

def get_error(models, x):
    return np.sum([model.get_error(x) for model in models])