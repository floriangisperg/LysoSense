"""Unit tests for the per-model math helpers in ``lysosense.analysis``.

These lock down the corrected lognormal FWHM (algorithm-review Finding 3) and
the true per-model mean (Finding 4). The gaussian-only regression suite in
``test_analysis`` does not exercise the lognormal/splitgaussian paths, so
without these the fixes could regress unnoticed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lysosense.analysis import _calculate_fwhm_from_params, _mean_from_params

_SQRT_2LN2 = math.sqrt(2.0 * math.log(2.0))


# --- _mean_from_params (Finding 4: report the true mean, not the mode) -------


def test_mean_gaussian_equals_mode():
    # params: (amplitude, mu, sigma) -> symmetric, mean == mode == mu
    assert _mean_from_params("gaussian", np.array([2.0, 0.5, 0.1])) == pytest.approx(0.5)


def test_mean_splitgaussian_shifts_toward_the_wider_side():
    # params: (amplitude, mu, sigma_left, sigma_right); right side wider -> mean > mu
    mean = _mean_from_params("splitgaussian", np.array([1.0, 0.5, 0.05, 0.15]))
    expected = 0.5 + math.sqrt(2.0 / math.pi) * (0.15 - 0.05)
    assert mean == pytest.approx(expected)
    assert mean > 0.5


def test_mean_lognormal_exceeds_mode():
    # params: (amplitude, mode, shape); true mean = mode * exp(1.5 * shape^2)
    mode, shape = 0.48, 0.4
    mean = _mean_from_params("lognormal", np.array([1.0, mode, shape]))
    assert mean == pytest.approx(mode * math.exp(1.5 * shape * shape))
    # The old code reported params[1] (the mode), under-counting the mean.
    assert mean > mode


# --- _calculate_fwhm_from_params (Finding 3: exact lognormal FWHM) -----------


def test_fwhm_gaussian_is_2355_sigma():
    assert _calculate_fwhm_from_params("gaussian", np.array([1.0, 0.5, 0.1])) == pytest.approx(
        2.355 * 0.1
    )


def test_fwhm_lognormal_uses_exact_formula_not_the_small_shape_approx():
    # Exact: 2 * mode * sinh(shape * sqrt(2 ln 2)). This is the inverse of
    # _shape_cap_from_fwhm, so fit-time bounds and post-fit checks agree.
    mode, shape = 0.5, 0.6
    fwhm = _calculate_fwhm_from_params("lognormal", np.array([1.0, mode, shape]))
    expected = 2.0 * mode * math.sinh(shape * _SQRT_2LN2)
    assert fwhm == pytest.approx(expected)
    # ... and it must differ from the old approximation 2.355 * shape * mode.
    old_approx = 2.355 * shape * mode
    assert fwhm != pytest.approx(old_approx, abs=1e-6)
