"""Provides unit tests for FEM-BV-VARX model."""

# License: MIT

import os

import numpy as np
import pytest
import scipy.linalg as sl
import scipy.stats as ss
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state

from fembv_models import FEMBVVARX
from fembv_models.fembv_varx import (
    _compute_covariances,
    _compute_precision_cholesky,
)


TEST_DATA_PATH = os.path.realpath(os.path.dirname(__file__))


def test_throws_when_given_invalid_n_components():
    """Test throws exception when given invalid number of components."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 100
    n_features = 2

    X = random_state.uniform(size=(n_samples, n_features))

    model = FEMBVVARX(n_components=-1)

    with pytest.raises(ValueError):
        model.fit(X)


def test_precisions_correctly_calculated():
    """Test precisions correctly computed from covariance matrices."""
    random_seed = 0

    random_state = check_random_state(random_seed)

    n_components = 3
    n_features = 3

    covariances = ss.invwishart.rvs(df=n_features, scale=np.eye(n_features),
                                    size=(n_components,),
                                    random_state=random_state)

    precisions_chol = _compute_precision_cholesky(covariances)

    for k in range(n_components):
        expected_precision = np.linalg.inv(covariances[k])

        assert np.allclose(expected_precision,
                           np.dot(precisions_chol[k], precisions_chol[k].T))


def test_covariances_correctly_calculated():
    """Test covariances correctly computed from precision matrices."""
    random_seed = 0

    random_state = check_random_state(random_seed)

    n_components = 3
    n_features = 3

    covariances = ss.invwishart.rvs(df=n_features, scale=np.eye(n_features),
                                    size=(n_components,),
                                    random_state=random_state)

    precisions_chol = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        precision = np.linalg.inv(covariances[k])
        precisions_chol[k] = sl.cholesky(precision, lower=True)

    calculated_covariances = _compute_covariances(precisions_chol)

    for k in range(n_components):

        assert np.allclose(covariances[k], calculated_covariances[k])


def test_rmse_correctly_calculated():
    """Test RMSE correctly computed."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_components = 1
    n_features = 3
    n_samples = 100

    X = random_state.normal(size=(n_samples, n_features))

    p = 1
    model = FEMBVVARX(
        n_components=n_components, orders=p, loss='log_likelihood').fit(X)

    calculated_rmse = model.rmse_

    X_pred = model.means_[0] + np.dot(X[:-1, :], model.endog_coef_[0, 0].T)
    expected_rmse = mean_squared_error(X[1:], X_pred, squared=False)

    assert np.abs(expected_rmse - calculated_rmse) < 1e-6

    model.n_components = 2
    model.weights_ = np.zeros((n_samples - p, 2))
    model.weights_[:(n_samples - p) // 2, 0] = 1.0
    model.weights_[(n_samples - p) // 2:, 1] = 1.0

    duplicated_Y_k = {k: model._Y_k[0].copy() for k in range(2)}
    duplicated_Theta_k = {k: model._Theta_k[0].copy() for k in range(2)}

    model._Y_k = duplicated_Y_k
    model._Theta_k = duplicated_Theta_k

    calculated_rmse = model._rmse(X)

    assert np.abs(expected_rmse - calculated_rmse) < 1e-6


def test_linear_varx_leastsq_example_1():
    """Test gives correct least-squares estimate for example dataset."""
    p = 2

    data = np.genfromtxt(os.path.join(TEST_DATA_PATH, 'test_dataset_1.csv'),
                         delimiter=',', names=True)

    # Use data from 1960Q1 to 1978Q4
    mask = data['year'] <= 1978

    invest = data['invest'][mask]
    income = data['income'][mask]
    cons = data['cons'][mask]

    # Fits are performed on first differences of log-data
    log_invest = np.log(invest)
    log_income = np.log(income)
    log_cons = np.log(cons)

    y = np.vstack(
        [np.diff(log_invest), np.diff(log_income), np.diff(log_cons)]).T

    assert y.shape == (75, 3)

    model = FEMBVVARX(n_components=1, orders=p, loss='log_likelihood').fit(y)

    expected_mu = np.array([-0.017, 0.016, 0.013])
    expected_A = np.array([[[-0.320, 0.146, 0.961],
                            [0.044, -0.153, 0.289],
                            [-0.002, 0.225, -0.264]],
                           [[-0.161, 0.115, 0.934],
                            [0.050, 0.019, -0.010],
                            [0.034, 0.355, -0.022]]])
    expected_Sigma_LS = np.array([[21.30e-4, 0.72e-4, 1.23e-4],
                                  [0.72e-4, 1.37e-4, 0.61e-4],
                                  [1.23e-4, 0.61e-4, 0.89e-4]])

    df_bias = y.shape[0] / (y.shape[0] - 3 * p - 1.0)

    assert np.allclose(model.means_, expected_mu, atol=1e-3)
    assert np.allclose(model.endog_coef_, expected_A, atol=1e-3)
    assert model.exog_coef_ is None
    assert np.allclose(df_bias * model.covariances_,
                       expected_Sigma_LS, atol=1e-5)
    assert np.allclose(model.weights_, 1)


def test_finds_correct_weights_with_hidden_step_independent_noise():
    """Test converges to correct weights with two components."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 3
    n_samples = 100

    # generate time series corresponding to random perturbations of a
    # step function
    switch_point = n_samples // 2
    step_height = 5
    component_one_scale = 1.0
    component_two_scale = 0.5

    step = np.zeros((n_samples, n_features))
    step[switch_point:] = step_height

    component_one_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_one_scale * np.eye(n_features)),
        size=(switch_point,))
    component_two_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_two_scale * np.eye(n_features)),
        size=(switch_point,))

    X = step.copy()
    X[:switch_point] += component_one_noise
    X[switch_point:] += component_two_noise

    model = FEMBVVARX(
        n_components=2, orders=0, n_init=100, loss='log_likelihood').fit(X)

    expected_weights = np.zeros((n_samples, 2))
    expected_weights[:switch_point, 0] = 1
    expected_weights[switch_point:, 1] = 1

    # note state ordering is not fixed
    if model.weights_[0, 0] > model.weights_[0, 1]:
        assert np.allclose(model.weights_, expected_weights)
    else:
        assert np.allclose(np.fliplr(model.weights_), expected_weights)


def test_finds_correct_weights_with_hidden_ar1_process_different_levels():
    """Test converges to correct weights with two components."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 4
    n_samples = 500

    # generate time series corresponding to single switch between two AR(1)
    # processes with different levels
    switch_point = n_samples // 2
    x0 = random_state.uniform(size=(n_features,))

    X = np.zeros((n_samples, n_features))
    X[0] = x0

    component_one_eigvals = np.diag(random_state.uniform(size=(n_features,)))
    component_one_eigvecs = random_state.normal(size=(n_features, n_features))
    component_one_A = np.dot(
        np.linalg.inv(component_one_eigvecs),
        np.dot(component_one_eigvals, component_one_eigvecs))

    component_two_eigvals = np.diag(random_state.uniform(size=(n_features,)))
    component_two_eigvecs = random_state.normal(size=(n_features, n_features))
    component_two_A = np.dot(
        np.linalg.inv(component_two_eigvecs),
        np.dot(component_two_eigvals, component_two_eigvecs))

    # check stationarity
    assert np.all(np.abs(np.linalg.eigvals(component_one_A)) < 1)
    assert np.all(np.abs(np.linalg.eigvals(component_two_A)) < 1)

    component_one_mean = np.ones((n_features,))
    component_two_mean = 10.0 * np.ones((n_features,))

    component_one_scale = 1.0
    component_two_scale = 0.5

    component_one_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_one_scale * np.eye(n_features)),
        size=(switch_point,))
    component_two_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_two_scale * np.eye(n_features)),
        size=(switch_point,))

    for t in range(1, n_samples):
        if t < switch_point:
            X[t, :] = (component_one_mean +
                       np.dot(component_one_A, X[t - 1]) +
                       component_one_noise[t - 1])
        else:
            X[t, :] = (component_two_mean +
                       np.dot(component_two_A, X[t - 1]) +
                       component_two_noise[t - switch_point])

    state_length = n_samples // 2
    model = FEMBVVARX(
        n_components=2, orders=1, state_length=state_length, n_init=100,
        loss='log_likelihood',
        weights_solver_kwargs={'max_iters': 1000}).fit(X)

    expected_weights = np.zeros((n_samples, 2))
    expected_weights[:switch_point, 0] = 1
    expected_weights[switch_point:, 1] = 1
    # the first observation is dropped as a presample value
    expected_weights = expected_weights[1:]

    # note state ordering is not fixed
    if model.weights_[0, 0] > model.weights_[0, 1]:
        assert np.allclose(model.weights_, expected_weights)
    else:
        assert np.allclose(np.fliplr(model.weights_), expected_weights)

    if model._max_tv_norm is not None:
        assert np.all(
            np.sum(
                np.abs(
                    np.diff(
                        model.weights_, axis=0)), axis=0) < model._max_tv_norm)


def test_predicts_correct_weights_with_hidden_step_independent_noise():
    """Test predicts correct weights with two components."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 3
    n_samples = 100

    # generate time series corresponding to random perturbations of a
    # step function
    switch_point = n_samples // 2
    step_height = 5
    component_one_scale = 1.0
    component_two_scale = 0.5

    step = np.zeros((n_samples, n_features))
    step[switch_point:] = step_height

    component_one_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_one_scale * np.eye(n_features)),
        size=(switch_point,))
    component_two_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_two_scale * np.eye(n_features)),
        size=(switch_point,))

    X = step.copy()
    X[:switch_point] += component_one_noise
    X[switch_point:] += component_two_noise

    # generate test set sample with underlying model given by the
    # model that is active at the end of the training set
    n_test_samples = 50
    X_test = (step_height +
              random_state.multivariate_normal(
                mean=np.zeros((n_features,)),
                cov=(component_two_scale * np.eye(n_features)),
                size=(n_test_samples,)))

    model = FEMBVVARX(
        n_components=2, orders=0, n_init=100, loss='log_likelihood').fit(X)

    expected_weights = np.zeros((n_samples, 2))
    expected_weights[:switch_point, 0] = 1
    expected_weights[switch_point:, 1] = 1

    # note state ordering is not fixed
    if model.weights_[0, 0] > model.weights_[0, 1]:
        states_flipped = False
        state_index = 1
        assert np.allclose(model.weights_, expected_weights)
    else:
        states_flipped = True
        state_index = 0
        assert np.allclose(np.fliplr(model.weights_), expected_weights)

    fitted_means = model.means_.copy()
    fitted_covariances = model.covariances_.copy()
    fitted_prec_chol = model.precisions_cholesky_.copy()

    predicted_weights, predicted_cost, predicted_log_like, predicted_rmse = \
        model.predict(X_test)

    assert np.allclose(model.means_, fitted_means)
    assert np.allclose(model.covariances_, fitted_covariances)
    assert np.allclose(model.precisions_cholesky_, fitted_prec_chol)

    if states_flipped:
        assert np.allclose(predicted_weights[:, 1], 0)
        assert np.allclose(predicted_weights[:, 0], 1)
    else:
        assert np.allclose(predicted_weights[:, 0], 0)
        assert np.allclose(predicted_weights[:, 1], 1)

    expected_log_like = 0.0
    for t in range(n_test_samples):
        expected_log_like += ss.multivariate_normal.logpdf(
            X_test[t], mean=model.means_[state_index],
            cov=model.covariances_[state_index])

    assert np.abs(predicted_cost + expected_log_like) < 1e-6
    assert np.abs(predicted_log_like - expected_log_like) < 1e-6


def test_predicts_correct_weights_with_hidden_ar1_process_different_levels():
    """Test predicts correct weights with two components."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 4
    n_samples = 500

    # generate time series corresponding to single switch between two AR(1)
    # processes with different levels
    switch_point = n_samples // 2
    x0 = random_state.uniform(size=(n_features,))

    X = np.zeros((n_samples, n_features))
    X[0] = x0

    component_one_eigvals = np.diag(random_state.uniform(size=(n_features,)))
    component_one_eigvecs = random_state.normal(size=(n_features, n_features))
    component_one_A = np.dot(
        np.linalg.inv(component_one_eigvecs),
        np.dot(component_one_eigvals, component_one_eigvecs))

    component_two_eigvals = np.diag(random_state.uniform(size=(n_features,)))
    component_two_eigvecs = random_state.normal(size=(n_features, n_features))
    component_two_A = np.dot(
        np.linalg.inv(component_two_eigvecs),
        np.dot(component_two_eigvals, component_two_eigvecs))

    # check stationarity
    assert np.all(np.abs(np.linalg.eigvals(component_one_A)) < 1)
    assert np.all(np.abs(np.linalg.eigvals(component_two_A)) < 1)

    component_one_mean = np.ones((n_features,))
    component_two_mean = 10.0 * np.ones((n_features,))

    component_one_scale = 1.0
    component_two_scale = 0.5

    component_one_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_one_scale * np.eye(n_features)),
        size=(switch_point,))
    component_two_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_two_scale * np.eye(n_features)),
        size=(switch_point,))

    for t in range(1, n_samples):
        if t < switch_point:
            X[t, :] = (component_one_mean +
                       np.dot(component_one_A, X[t - 1]) +
                       component_one_noise[t - 1])
        else:
            X[t, :] = (component_two_mean +
                       np.dot(component_two_A, X[t - 1]) +
                       component_two_noise[t - switch_point])

    # generate test set sample with underlying model given by the
    # model that is active at the end of the training set
    n_test_samples = 50

    test_noise = random_state.multivariate_normal(
        mean=np.zeros((n_features,)),
        cov=(component_two_scale * np.eye(n_features)),
        size=(n_test_samples,))

    X_test = np.zeros((n_test_samples, n_features))
    for t in range(1, n_test_samples):
        X_test[t, :] = (component_two_mean +
                        np.dot(component_two_A, X_test[t - 1]) +
                        test_noise[t])

    state_length = n_samples // 2
    model = FEMBVVARX(
        n_components=2, orders=1, state_length=state_length, n_init=100,
        loss='log_likelihood',
        weights_solver_kwargs={'max_iters': 1000}).fit(X)

    expected_weights = np.zeros((n_samples, 2))
    expected_weights[:switch_point, 0] = 1
    expected_weights[switch_point:, 1] = 1
    # the first observation is dropped as a presample value
    expected_weights = expected_weights[1:]

    # note state ordering is not fixed
    if model.weights_[0, 0] > model.weights_[0, 1]:
        states_flipped = False
        state_index = 1
        assert np.allclose(model.weights_, expected_weights)
    else:
        states_flipped = True
        state_index = 0
        assert np.allclose(np.fliplr(model.weights_), expected_weights)

    if model._max_tv_norm is not None:
        assert np.all(
            np.sum(
                np.abs(
                    np.diff(
                        model.weights_, axis=0)), axis=0) < model._max_tv_norm)

    init_max_tv_norm = model._max_tv_norm
    fitted_means = model.means_.copy()
    fitted_endog_coef = model.endog_coef_.copy()
    fitted_covariances = model.covariances_.copy()
    fitted_prec_chol = model.precisions_cholesky_.copy()

    predicted_weights, predicted_cost, predicted_log_like, predicted_rmse = \
        model.predict(X_test)

    if model._max_tv_norm is not None:
        assert np.abs(init_max_tv_norm - model._max_tv_norm) < 1e-8

    assert np.allclose(model.means_, fitted_means)
    assert np.allclose(model.endog_coef_, fitted_endog_coef)
    assert np.allclose(model.covariances_, fitted_covariances)
    assert np.allclose(model.precisions_cholesky_, fitted_prec_chol)

    if states_flipped:
        assert np.allclose(predicted_weights[:, 1], 0)
        assert np.allclose(predicted_weights[:, 0], 1)
    else:
        assert np.allclose(predicted_weights[:, 0], 0)
        assert np.allclose(predicted_weights[:, 1], 1)

    expected_log_like = 0.0
    for t in range(1, n_test_samples):
        X_hat = (model.means_[state_index] +
                 np.dot(model.endog_coef_[state_index, 0], X_test[t - 1]))
        expected_log_like += ss.multivariate_normal.logpdf(
            X_test[t], mean=X_hat, cov=model.covariances_[state_index])

    assert np.abs(predicted_cost + expected_log_like) < 1e-5
    assert np.abs(predicted_log_like - expected_log_like) < 1e-5


def test_runs_with_exogeneous_factor():
    """Test fitting converges when given exogeneous factors."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 6
    n_samples = 500

    X = random_state.normal(size=(n_samples, n_features))

    u = random_state.normal(size=(n_samples,))

    state_length = 0

    with pytest.raises(ValueError):
        model = FEMBVVARX(
            n_components=2, orders=1, state_length=state_length, n_init=100,
            loss='log_likelihood',
            weights_solver_kwargs={'max_iters': 1000}).fit(X, exog=u)

    n_exog = 1
    u = random_state.normal(size=(n_samples, n_exog))

    model = FEMBVVARX(
        n_components=2, orders=1, state_length=state_length, n_init=100,
        loss='log_likelihood',
        weights_solver_kwargs={'max_iters': 1000}).fit(X, exog=u)

    assert model.converged_

    n_exog = 3
    u = random_state.normal(size=(n_samples, n_exog))

    model = FEMBVVARX(
        n_components=2, orders=3, state_length=state_length, n_init=100,
        loss='log_likelihood',
        weights_solver_kwargs={'max_iters': 1000}).fit(X, exog=u)

    assert model.converged_
