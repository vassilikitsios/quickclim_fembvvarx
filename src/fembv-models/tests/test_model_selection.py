"""Provides unit tests for model selection helper routines."""

# License: MIT

import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
from sklearn.utils import check_array, check_random_state

from fembv_models import aic, bic


def _fit_least_squares_univariate_linear_model(X, y, bias=False):
    """Fit univariate linear model to data."""
    params = sl.lstsq(X, y)[0]

    residuals = y - np.dot(X, params)

    if bias:
        df = 1.0 * X.shape[0]
    else:
        df = 1.0 * (X.shape[0] - X.shape[1])

    covariance = np.dot(residuals.T, residuals) / df

    if np.size(covariance) == 1:
        covariance = covariance.item()

    return {'parameters': params, 'residuals': residuals,
            'covariance': covariance}


def _univariate_normal_log_likelihood(y, means=None, variance=None):
    """Calculate log-likelihood assuming normally distributed data."""
    y = check_array(y, ensure_2d=False, allow_nd=False)

    n_samples = y.shape[0]

    if means is None:
        means = np.zeros_like(y)
    else:
        means = check_array(means, ensure_2d=False, allow_nd=False)

        assert means.shape == y.shape

    if variance is None:
        variance = 1.0
    else:
        assert variance > 0

    log_likelihood = 0
    for t in range(n_samples):
        log_likelihood += ss.norm.logpdf(
            y[t], loc=means[t], scale=np.sqrt(variance))

    return log_likelihood


def _multivariate_normal_log_likelihood(X, means=None, covariance=None):
    """Calculate log-likelihood assuming normally distributed data."""
    X = check_array(X)

    n_samples, n_features = X.shape

    if means is None:
        means = np.zeros_like(X)
    else:
        means = check_array(means)

        assert means.shape == X.shape

    if covariance is None:
        covariance = np.eye(n_features)
    else:
        covariance = check_array(covariance)

        assert covariance.shape == (n_features, n_features)

    log_likelihood = 0
    for t in range(n_samples):
        log_likelihood += ss.multivariate_normal.logpdf(
            X[t], mean=means[t], cov=covariance)

    return log_likelihood


def test_aic_univariate_linear_model_normal_residuals():
    """Test calculation of AIC matches formula for normal residuals."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 6
    n_samples = 100

    resid_variance = ss.invgamma.rvs(2.0, random_state=random_state)
    mean = ss.norm.rvs(scale=10.0, size=(n_features,),
                       random_state=random_state)

    X = ss.multivariate_normal.rvs(
        mean=mean, cov=np.eye(n_features), size=(n_samples,),
        random_state=random_state)
    beta = ss.norm.rvs(scale=1.0, size=(n_features,),
                       random_state=random_state)

    y = (np.dot(X, beta) +
         ss.norm.rvs(loc=0, scale=resid_variance, size=(n_samples,),
                     random_state=random_state))

    linear_model = _fit_least_squares_univariate_linear_model(X, y, bias=True)

    model_log_likelihood = _univariate_normal_log_likelihood(
        y, means=np.dot(X, linear_model['parameters']),
        variance=linear_model['covariance'])

    n_params = np.size(linear_model['parameters']) + 1
    model_aic = aic(model_log_likelihood, n_params)

    expected_log_likelihood = (
        -0.5 * n_samples * np.log(
            np.sum(linear_model['residuals']**2) / n_samples) -
        0.5 * n_samples * np.log(2 * np.pi) - 0.5 * n_samples)
    expected_aic = (-2 * expected_log_likelihood + 2 * n_params)

    assert np.abs(model_aic - expected_aic) < 1e-6


def test_bic_univariate_linear_model_normal_residuals():
    """Test calculation of BIC."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 10
    n_samples = 50

    resid_variance = ss.invgamma.rvs(1.0, random_state=random_state)
    mean = ss.norm.rvs(scale=10.0, size=(n_features,),
                       random_state=random_state)

    X = ss.multivariate_normal.rvs(
        mean=mean, cov=np.eye(n_features), size=(n_samples,),
        random_state=random_state)
    beta = ss.norm.rvs(scale=1.0, size=(n_features,),
                       random_state=random_state)

    y = (np.dot(X, beta) +
         ss.norm.rvs(loc=0, scale=resid_variance, size=(n_samples,),
                     random_state=random_state))

    linear_model = _fit_least_squares_univariate_linear_model(X, y, bias=True)

    model_log_likelihood = _univariate_normal_log_likelihood(
        y, means=np.dot(X, linear_model['parameters']),
        variance=linear_model['covariance'])

    n_params = np.size(linear_model['parameters']) + 1
    model_bic = bic(model_log_likelihood, n_params, n_samples)

    expected_log_likelihood = (
        -0.5 * n_samples * np.log(
            np.sum(linear_model['residuals']**2) / n_samples) -
        0.5 * n_samples * np.log(2 * np.pi) - 0.5 * n_samples)
    expected_bic = (-2 * expected_log_likelihood +
                    n_params * np.log(n_samples))

    assert np.abs(model_bic - expected_bic) < 1e-6
