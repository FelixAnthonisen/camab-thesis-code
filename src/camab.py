import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats


@dataclass
class CMAB:
    distributions: List

    @property
    def n_arms(self) -> int:
        return len(self.distributions)


@dataclass
class CAMAB:
    base: CMAB
    abstract: CMAB
    omega: np.ndarray

    def __post_init__(self):
        assert len(self.omega) == self.base.n_arms
        assert int(self.omega.max()) < self.abstract.n_arms


def _blocks(n_clusters, cluster_size) -> np.ndarray:
    return np.repeat(np.arange(n_clusters), cluster_size)


def _centroid(mu_base, omega, n_clusters) -> np.ndarray:
    return np.array([mu_base[omega == j].mean() for j in range(n_clusters)])


def _gaussian_camab(mu_base, omega, mu_abstract, sigma) -> CAMAB:
    return CAMAB(
        base=CMAB([stats.norm(m, sigma) for m in mu_base]),
        abstract=CMAB([stats.norm(m, sigma) for m in mu_abstract]),
        omega=np.asarray(omega, dtype=int),
    )


def mean_shifts(mu_base, omega, mu_abstract) -> np.ndarray:
    return np.abs(np.asarray(mu_base) - np.asarray(mu_abstract)[omega])


def true_mean_shifts(camab: CAMAB) -> np.ndarray:
    mu_base = np.array([d.mean() for d in camab.base.distributions])
    mu_abstract = np.array([d.mean() for d in camab.abstract.distributions])
    return mean_shifts(mu_base, camab.omega, mu_abstract)


def coarsening_means(n_clusters=20, cluster_size=5, spread=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    omega = _blocks(n_clusters, cluster_size)
    centres = rng.uniform(0.0, 1.0, n_clusters)
    mu = centres[omega] + rng.normal(0.0, spread, n_clusters * cluster_size)
    return mu, omega, _centroid(mu, omega, n_clusters)


def biased_means(n_clusters=20, cluster_size=5, spread=0.05, bias=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mu, omega, centroid = coarsening_means(n_clusters, cluster_size, spread, rng)
    return mu, omega, centroid + rng.normal(0.0, bias, n_clusters)


def misaligned_means(n_clusters=20, cluster_size=5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n_arms = n_clusters * cluster_size
    mu = rng.uniform(0.0, 1.0, n_arms)
    omega = _blocks(n_clusters, cluster_size)[rng.permutation(n_arms)]
    return mu, omega, _centroid(mu, omega, n_clusters)


def common_offset_means(
    n_clusters=20, cluster_size=5, spread=0.005, offset=0.2, rng=None
):
    rng = np.random.default_rng() if rng is None else rng
    mu, omega, centroid = coarsening_means(n_clusters, cluster_size, spread, rng)
    return mu, omega, centroid - offset


def make_coarsening_camab(
    n_clusters=20, cluster_size=5, spread=0.05, sigma=1.0, seed=0
) -> CAMAB:
    rng = np.random.default_rng(seed)
    return _gaussian_camab(
        *coarsening_means(n_clusters, cluster_size, spread, rng), sigma
    )


def make_biased_camab(
    n_clusters=20, cluster_size=5, spread=0.05, bias=0.05, sigma=1.0, seed=0
) -> CAMAB:
    rng = np.random.default_rng(seed)
    return _gaussian_camab(
        *biased_means(n_clusters, cluster_size, spread, bias, rng), sigma
    )


def make_misaligned_camab(n_clusters=20, cluster_size=5, sigma=1.0, seed=0) -> CAMAB:
    rng = np.random.default_rng(seed)
    return _gaussian_camab(*misaligned_means(n_clusters, cluster_size, rng), sigma)


def make_common_offset_camab(
    n_clusters=20, cluster_size=5, spread=0.005, offset=0.2, sigma=1.0, seed=0
) -> CAMAB:
    rng = np.random.default_rng(seed)
    return _gaussian_camab(
        *common_offset_means(n_clusters, cluster_size, spread, offset, rng), sigma
    )


def pruning_at(omega, mu_abstract, threshold, tol=0.0):
    mu_abstract = np.asarray(mu_abstract)
    pruned = (mu_abstract.max() - mu_abstract) > threshold + tol
    arms = pruned[omega]
    return float(arms.mean())
