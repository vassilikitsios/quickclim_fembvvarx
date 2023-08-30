"""Provides routines for performing model selection."""

# License: MIT

from __future__ import absolute_import, division

import numpy as np


def aic(log_likelihood, df):
    """Calculate AIC."""
    return -2 * log_likelihood + 2 * df


def bic(log_likelihood, df, n_samples):
    """Calculate BIC."""
    return -2 * log_likelihood + df * np.log(n_samples)


def aicc(log_likelihood, df, n_samples):
    """Calculate AICc."""
    return -2 * log_likelihood + 2 * df * n_samples / (n_samples - df - 1)
