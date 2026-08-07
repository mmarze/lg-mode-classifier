import numpy as np
import pytest
from math import factorial
from scipy.special import eval_genlaguerre

from lgbeam.laguerre import C_lp, L_lp 


# -------------------------
# Tests for C_lp
# -------------------------

def test_C_lp_basic():
    """Check normalization constant for simple known values."""

    # p=0, l=0:
    # sqrt(2*0!/(pi*0!)) = sqrt(2/pi)
    expected = np.sqrt(2 / np.pi)

    result = C_lp(0, 0)

    assert np.isclose(result, expected)


def test_C_lp_nonzero_indices():
    """Compare against direct formula."""

    p = 2
    l = 3

    expected = np.sqrt(
        2 * factorial(p) /
        (np.pi * factorial(p + abs(l)))
    )

    result = C_lp(p, l)

    assert np.isclose(result, expected)


def test_C_lp_negative_p():
    """p cannot be negative."""

    with pytest.raises(ValueError):
        C_lp(-1, 0)


@pytest.mark.parametrize(
    "p, l",
    [
        (1.5, 0),
        ("1", 0),
        (1, 2.5),
        (1, "2"),
    ],
)
def test_C_lp_invalid_types(p, l):
    """p and l must be integers."""

    with pytest.raises(TypeError):
        C_lp(p, l)


# -------------------------
# Tests for L_lp
# -------------------------

def test_L_lp_basic():
    """Compare against scipy implementation."""

    p = 0
    l = 0
    r = np.array([0.0, 1e-3, 2e-3])
    w_z = 1e-3

    x = 2 * r**2 / w_z**2

    expected = eval_genlaguerre(p, abs(l), x)

    result = L_lp(p, l, r, w_z)

    assert np.allclose(result, expected)


def test_L_lp_nonzero_indices():
    """Test higher order Laguerre polynomial."""

    p = 2
    l = -3
    r = np.array([0.0, 0.5, 1.0])
    w_z = 2.0

    expected = eval_genlaguerre(
        p,
        abs(l),
        2 * r**2 / w_z**2
    )

    result = L_lp(p, l, r, w_z)

    assert np.allclose(result, expected)


def test_L_lp_negative_radius():
    """Radius cannot contain negative values."""

    r = np.array([-1.0, 0.0])

    with pytest.raises(ValueError):
        L_lp(0, 0, r, 1.0)


def test_L_lp_zero_width():
    """Beam width must be positive."""

    r = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        L_lp(0, 0, r, 0)


@pytest.mark.parametrize(
    "p, l, r, w_z",
    [
        (1.2, 0, np.array([1.0]), 1.0),
        (1, 2.3, np.array([1.0]), 1.0),
        (1, 2, [1.0], 1.0),
        (1, 2, np.array([1.0]), "1"),
    ],
)
def test_L_lp_invalid_types(p, l, r, w_z):
    """Check type validation."""

    with pytest.raises(TypeError):
        L_lp(p, l, r, w_z)
        