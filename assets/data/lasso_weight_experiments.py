"""Reproduce the scalar-lasso allocation experiments used in blog Part 5.

The design has p=12 standardized scalar predictors and three nonzero generating
coefficients. Matrix solves use Woodbury's identity, so each auxiliary solve is
only p by p even though the definition of G(gamma) is n by n.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STRATEGIES = ("waterfill", "uniform", "random", "roundrobin")


def get_weights(alpha: np.ndarray, tau: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Algorithm getWeights from the supplementary material, scalar case."""
    delta = alpha - tau
    delta_sq = delta**2
    gamma_tau = gamma * tau
    eligible = ~((delta >= 0) & (gamma == 0))
    if not np.any(eligible):
        return np.zeros_like(gamma)

    d = delta[eligible]
    d_sq = delta_sq[eligible]
    gt = gamma_tau[eligible]
    order = np.argsort(-d_sq)
    w = np.zeros(len(order))

    for k in range(len(order)):
        level = d_sq[order[k]]
        if k:
            w[order[:k]] = gt[order[:k]] / np.sqrt(level)
        w[order[k]] = 1 - w[order[:k]].sum()
        knot = gt[order[k]] / d[order[k]] if d[order[k]] != 0 else np.inf
        if w[order[k]] > 0 and (w[order[k]] < knot or d[order[k]] <= 0):
            break

        level = gt[order[: k + 1]].sum() ** 2
        next_level = d_sq[order[min(len(order) - 1, k + 1)]]
        if d_sq[order[k]] >= level and (level >= next_level or k == len(order) - 1):
            w[order[: k + 1]] = gt[order[: k + 1]] / np.sqrt(level)
            break

    out = np.zeros_like(gamma)
    out[eligible] = w
    return out


def make_instance(seed: int, n: int = 300, p: int = 12, rho: float = 0.55):
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((n, p))
    shared = rng.standard_normal((n, 1))
    x = np.sqrt(1 - rho) * independent + np.sqrt(rho) * shared
    x -= x.mean(axis=0, keepdims=True)
    x /= np.linalg.norm(x, axis=0, keepdims=True)

    beta = np.zeros(p)
    beta[:3] = (2.0, -1.6, 1.2)
    signal = x @ beta
    noise = rng.standard_normal(n)
    noise -= noise.mean()
    noise *= np.linalg.norm(signal) / (2 * np.linalg.norm(noise))
    return x, signal + noise, beta


class AuxiliaryLasso:
    def __init__(self, x: np.ndarray, y: np.ndarray, lambda_fraction: float = 0.45):
        self.x = x
        self.y = y
        self.xtx = x.T @ x
        self.xty = x.T @ y
        threshold = lambda_fraction * np.max(np.abs(self.xty))
        self.alpha = np.full(x.shape[1], threshold)

    def gy(self, gamma: np.ndarray) -> np.ndarray:
        root = np.sqrt(np.maximum(gamma, 0))
        middle = np.eye(len(gamma)) + root[:, None] * self.xtx * root[None, :]
        correction = root * np.linalg.solve(middle, root * self.xty)
        return self.y - self.x @ correction

    def objective(self, gamma: np.ndarray) -> float:
        gy = self.gy(gamma)
        return float(self.y @ gy + np.sum(self.alpha**2 * gamma))

    def tau(self, gamma: np.ndarray) -> np.ndarray:
        return np.abs(self.x.T @ self.gy(gamma))

    def step(self, gamma: np.ndarray, w: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return np.maximum((gamma + w) * tau / self.alpha - w, 0)


def optimum(model: AuxiliaryLasso) -> tuple[np.ndarray, float]:
    gamma = np.zeros(model.x.shape[1])
    for _ in range(2000):
        tau = model.tau(gamma)
        w = get_weights(model.alpha, tau, gamma)
        if w.sum() == 0:
            break
        updated = model.step(gamma, w, tau)
        if np.max(np.abs(updated - gamma)) < 1e-14:
            gamma = updated
            break
        gamma = updated
    return gamma, model.objective(gamma)


def run_strategy(
    model: AuxiliaryLasso,
    strategy: str,
    g_star: float,
    iterations: int,
    seed: int,
) -> list[float]:
    gamma = np.zeros(model.x.shape[1])
    rng = np.random.default_rng(seed)
    gaps = []
    for t in range(iterations + 1):
        gaps.append(max(model.objective(gamma) - g_star, 1e-15))
        if t == iterations:
            break
        tau = model.tau(gamma)
        if strategy == "waterfill":
            w = get_weights(model.alpha, tau, gamma)
        elif strategy == "uniform":
            w = np.full(len(gamma), 1 / len(gamma))
        elif strategy == "random":
            w = rng.dirichlet(np.ones(len(gamma)))
        elif strategy == "roundrobin":
            w = np.zeros(len(gamma))
            w[t % len(gamma)] = 1
        else:
            raise ValueError(strategy)
        gamma = model.step(gamma, w, tau)
    return gaps


def summarize(n_seeds: int = 20, tolerance: float = 1e-6, max_iterations: int = 200):
    hits = {name: [] for name in STRATEGIES}
    for offset in range(n_seeds):
        x, y, _ = make_instance(100 + offset)
        model = AuxiliaryLasso(x, y)
        _, g_star = optimum(model)
        for index, strategy in enumerate(STRATEGIES):
            gaps = run_strategy(model, strategy, g_star, max_iterations, offset * 17 + index)
            hit = next((i for i, value in enumerate(gaps) if value < tolerance), max_iterations)
            hits[strategy].append(hit)

    result = {}
    for strategy, values in hits.items():
        array = np.asarray(values)
        result[strategy] = {
            "median": float(np.median(array)),
            "q25": float(np.percentile(array, 25)),
            "q75": float(np.percentile(array, 75)),
            "failures": int(np.sum(array >= max_iterations)),
        }
    return result


def main() -> None:
    x, y, beta = make_instance(11)
    model = AuxiliaryLasso(x, y)
    gamma_star, g_star = optimum(model)
    traces = {
        strategy: run_strategy(model, strategy, g_star, 60, 700 + index)
        for index, strategy in enumerate(STRATEGIES)
    }
    result = {
        "meta": {
            "n": int(x.shape[0]),
            "p": int(x.shape[1]),
            "generating_nonzero": np.flatnonzero(beta).tolist(),
            "solution_nonzero": np.flatnonzero(gamma_star > 1e-8).tolist(),
            "lambda_fraction": 0.45,
            "replications": 20,
            "tolerance": 1e-6,
        },
        "traces": traces,
        "summary": summarize(),
    }
    destination = Path(__file__).with_name("lasso_weight_results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
