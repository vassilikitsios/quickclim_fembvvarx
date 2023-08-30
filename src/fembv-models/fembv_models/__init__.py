"""Provides routines for fitting FEM-BV models."""

# License: MIT

from .fembv_varx import FEMBVVARX
from .fembv_weights import FEMBVWeights
from .model_selection import aic, aicc, bic
from .stochastic_matrix import left_stochastic_matrix, right_stochastic_matrix
from .validation import (
    check_array_shape,
    check_unit_axis_sums,
    is_integer,
    is_scalar,
)


__version__ = '0.0.1'


__all__ = [
    "FEMBVVARX",
    "FEMBVWeights",
    "aic",
    "aicc",
    "bic",
    "check_array_shape",
    "check_unit_axis_sums",
    "is_integer",
    "is_scalar",
    "left_stochastic_matrix",
    "right_stochastic_matrix"
]
