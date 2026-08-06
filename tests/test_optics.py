import numpy as np
import pytest

from lgbeam.beams import zR, R_z, w_z, wavenumber, psi


# ==========================================================
# zR
# ==========================================================

def test_zR_correct_value():
    w0 = 1e-3
    n = 1.5
    wavelength = 500e-9

    expected = np.pi * w0**2 * n / wavelength

    assert np.isclose(zR(w0, n, wavelength), expected)


@pytest.mark.parametrize(
    "wavelength",
    [100e-9, 1e-3],
)
def test_zR_boundary_wavelengths(wavelength):
    assert np.isfinite(zR(1e-3, 1.0, wavelength))


@pytest.mark.parametrize(
    "w0, n, wavelength",
    [
        ("1", 1.0, 500e-9),
        (1e-3, "1", 500e-9),
        (1e-3, 1.0, 500),
        (1e-3, 1.0, "500e-9"),
    ],
)
def test_zR_type_errors(w0, n, wavelength):
    with pytest.raises(TypeError):
        zR(w0, n, wavelength)


@pytest.mark.parametrize(
    "w0, n, wavelength",
    [
        (0, 1.0, 500e-9),
        (-1e-3, 1.0, 500e-9),
        (1e-3, 0, 500e-9),
        (1e-3, -1, 500e-9),
        (1e-3, 1.0, 99e-9),
        (1e-3, 1.0, 1.1e-3),
        (np.inf, 1.0, 500e-9),
        (1e-3, np.nan, 500e-9),
    ],
)
def test_zR_value_errors(w0, n, wavelength):
    with pytest.raises(ValueError):
        zR(w0, n, wavelength)


# ==========================================================
# R_z
# ==========================================================

def test_R_z_correct_value():
    z = 2.0
    zr = 5.0

    expected = z * (1 + (zr / z) ** 2)

    assert np.isclose(R_z(z, zr), expected)


def test_R_z_zero_returns_inf():
    assert np.isinf(R_z(0, 1.0))


@pytest.mark.parametrize(
    "z, zr",
    [
        ("1", 1.0),
        (1.0, "1"),
    ],
)
def test_R_z_type_errors(z, zr):
    with pytest.raises(TypeError):
        R_z(z, zr)


@pytest.mark.parametrize(
    "z, zr",
    [
        (1.0, 0),
        (1.0, -1),
        (np.nan, 1),
        (1, np.inf),
    ],
)
def test_R_z_value_errors(z, zr):
    with pytest.raises(ValueError):
        R_z(z, zr)


# ==========================================================
# w_z
# ==========================================================

def test_w_z_at_focus():
    assert np.isclose(
        w_z(1e-3, 0, 2.0),
        1e-3,
    )


def test_w_z_correct_value():
    w0 = 1e-3
    z = 2.0
    zr = 5.0

    expected = w0 * np.sqrt(1 + (z / zr) ** 2)

    assert np.isclose(
        w_z(w0, z, zr),
        expected,
    )


@pytest.mark.parametrize(
    "w0, z, zr",
    [
        ("1", 1, 1),
        (1, "1", 1),
        (1, 1, "1"),
    ],
)
def test_w_z_type_errors(w0, z, zr):
    with pytest.raises(TypeError):
        w_z(w0, z, zr)


@pytest.mark.parametrize(
    "w0, z, zr",
    [
        (0, 1, 1),
        (-1, 1, 1),
        (1, 1, 0),
        (1, 1, -1),
        (np.nan, 1, 1),
        (1, np.inf, 1),
    ],
)
def test_w_z_value_errors(w0, z, zr):
    with pytest.raises(ValueError):
        w_z(w0, z, zr)


# ==========================================================
# wavenumber
# ==========================================================

def test_wavenumber_correct_value():
    wavelength = 532e-9

    expected = 2 * np.pi / wavelength

    assert np.isclose(
        wavenumber(wavelength),
        expected,
    )


@pytest.mark.parametrize(
    "wavelength",
    [
        100e-9,
        1e-3,
    ],
)
def test_wavenumber_boundaries(wavelength):
    assert np.isfinite(
        wavenumber(wavelength)
    )


@pytest.mark.parametrize(
    "wavelength",
    [
        500,
        "532e-9",
    ],
)
def test_wavenumber_type_errors(wavelength):
    with pytest.raises(TypeError):
        wavenumber(wavelength)


@pytest.mark.parametrize(
    "wavelength",
    [
        np.nan,
        np.inf,
        99e-9,
        1.1e-3,
    ],
)
def test_wavenumber_value_errors(wavelength):
    with pytest.raises(ValueError):
        wavenumber(wavelength)


# ==========================================================
# psi
# ==========================================================

def test_psi_correct_value():
    l = -2
    p = 1
    z = 2.0
    zr = 5.0

    expected = (
        (abs(l) + 2 * p + 1)
        * np.arctan(z / zr)
    )

    assert np.isclose(
        psi(l, p, z, zr),
        expected,
    )


def test_psi_at_focus():
    assert np.isclose(
        psi(3, 2, 0, 1),
        0.0,
    )


@pytest.mark.parametrize(
    "l, p, z, zr",
    [
        ("1", 0, 1, 1),
        (1.5, 0, 1, 1),
        (1, "0", 1, 1),
        (1, 0, "1", 1),
        (1, 0, 1, "1"),
    ],
)
def test_psi_type_errors(l, p, z, zr):
    with pytest.raises(TypeError):
        psi(l, p, z, zr)


@pytest.mark.parametrize(
    "l, p, z, zr",
    [
        (1, -1, 1, 1),
        (1, 0, 1, 0),
        (1, 0, 1, -1),
        (1, 0, np.nan, 1),
        (1, 0, 1, np.inf),
    ],
)
def test_psi_value_errors(l, p, z, zr):
    with pytest.raises(ValueError):
        psi(l, p, z, zr)
        