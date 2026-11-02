"""Toy data shared by the tests: small count matrices drawn in memory, so no test downloads anything."""

import numpy as np
import pytest
import scipy.sparse as sp


def nb_counts(rng, mean, n_cells, theta=2.0, depth_sd=0.3):
    """Whole-number counts (cells x genes): a negative binomial per gene around `mean`, with varying cell depth."""
    depth = np.exp(rng.normal(0.0, depth_sd, n_cells))
    lam = rng.gamma(theta, (depth[:, None] * mean[None, :]) / theta)
    return rng.poisson(lam).astype(np.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def controls(rng):
    """A toy line's control cells: 300 cells x 120 genes, as CSR counts."""
    mean = np.exp(rng.normal(1.0, 1.0, 120))
    return sp.csr_matrix(nb_counts(rng, mean, 300))
