import numpy as np
import pytest

from lgbeam.beams import LG


# ==========================================================
# Correct output
# ==========================================================

def test_LG_returns_complex_array():
    """LG should return a complex ndarray of the same shape."""

    r = np.linspace(0, 1e-3, 50)
    phi = np.zeros_like(r)

    U = LG(
        p=0,
        l=0,
        r=r,
        phi=phi,
        z=0,
        w0=1e-3,
        wavelength=532e-9,
    )

    assert isinstance(U, np.ndarray)
    assert U.dtype == np.complex128
    assert U.shape == r.shape


def test_LG_no_nan_or_inf():
    """Output should be finite."""

    r = np.linspace(0, 2e-3, 100)
    phi = np.linspace(-np.pi, np.pi, 100)

    U = LG(
        1,
        2,
        r,
        phi,
        1e-2,
        1e-3,
        532e-9,
    )

    assert np.all(np.isfinite(U))


# ==========================================================
# Physical properties
# ==========================================================

def test_LG_fundamental_mode_nonzero_at_origin():
    """LG00 has maximum amplitude at r=0."""

    r = np.array([0.0])
    phi = np.array([0.0])

    U = LG(
        0,
        0,
        r,
        phi,
        0,
        1e-3,
        532e-9,
    )

    assert np.abs(U[0]) > 0


def test_LG_vortex_zero_at_origin():
    """Modes with |l|>0 vanish at the origin."""

    r = np.array([0.0])
    phi = np.array([0.0])

    U = LG(
        0,
        2,
        r,
        phi,
        0,
        1e-3,
        532e-9,
    )

    assert np.isclose(U[0], 0)


def test_LG_negative_l_same_amplitude():
    """Changing sign of l changes phase but not amplitude."""

    r = np.linspace(0, 1e-3, 20)
    phi = np.linspace(-np.pi, np.pi, 20)

    U1 = LG(
        1,
        2,
        r,
        phi,
        1e-3,
        1e-3,
        532e-9,
    )

    U2 = LG(
        1,
        -2,
        r,
        phi,
        1e-3,
        1e-3,
        532e-9,
    )

    assert np.allclose(np.abs(U1), np.abs(U2))


# ==========================================================
# Type checking
# ==========================================================

@pytest.mark.parametrize(
    "r, phi",
    [
        ([0, 1], np.array([0, 0])),
        (np.array([0, 1]), [0, 0]),
        (1, np.array([0])),
        (np.array([0]), 1),
    ],
)
def test_LG_type_errors(r, phi):

    with pytest.raises(TypeError):
        LG(
            0,
            0,
            r,
            phi,
            0,
            1e-3,
            532e-9,
        )


# ==========================================================
# Value checking inside LG
# ==========================================================

@pytest.mark.parametrize(
    "r, phi",
    [
        (np.array([np.nan]), np.array([0])),
        (np.array([np.inf]), np.array([0])),
        (np.array([0]), np.array([np.nan])),
        (np.array([0]), np.array([np.inf])),
        (np.array([-1]), np.array([0])),
        (np.array([0]), np.array([np.pi + 0.1])),
        (np.array([0]), np.array([-np.pi - 0.1])),
    ],
)
def test_LG_value_errors(r, phi):

    with pytest.raises(ValueError):
        LG(
            0,
            0,
            r,
            phi,
            0,
            1e-3,
            532e-9,
        )


# ==========================================================
# Errors propagated from helper functions
# ==========================================================

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(p=-1),
        dict(w0=0),
        dict(wavelength=50e-9),
        dict(n=0),
    ],
)
def test_LG_helper_validation(kwargs):

    params = dict(
        p=0,
        l=0,
        r=np.array([0]),
        phi=np.array([0]),
        z=0,
        w0=1e-3,
        wavelength=532e-9,
        n=1,
    )

    params.update(kwargs)

    with pytest.raises(ValueError):
        LG(**params)


# ==========================================================
# Boundary values
# ==========================================================

def test_LG_phi_boundaries():

    r = np.array([1e-4, 2e-4])

    phi = np.array([-np.pi, np.pi])

    U = LG(
        0,
        1,
        r,
        phi,
        0,
        1e-3,
        532e-9,
    )

    assert np.all(np.isfinite(U))


def test_LG_zero_radius_allowed():

    r = np.zeros(5)
    phi = np.zeros(5)

    U = LG(
        0,
        0,
        r,
        phi,
        0,
        1e-3,
        532e-9,
    )

    assert np.all(np.isfinite(U))
