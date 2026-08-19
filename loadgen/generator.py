import numpy as np


def generate_task_sizes(n: int, mean_units: float = 120_000, sigma: float = 0.5, seed=None) -> list:
    """Genera n tamaños de tarea (en 'unidades de trabajo') con distribución
    log-normal, replicando el enfoque de carga sintética usado en el
    antecedente directo (Rahimov & Aghayev, 2026)."""
    rng = np.random.default_rng(seed)
    mu = np.log(mean_units) - (sigma ** 2) / 2
    sizes = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return sizes.tolist()
