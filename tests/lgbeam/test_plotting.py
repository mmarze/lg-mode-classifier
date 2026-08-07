import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lgbeam.plotting import (
    plot_intensity,
    plot_phase,
    plot_complex,
    plot_beam,
)


PLOT_FUNCS = [
    plot_intensity,
    plot_phase,
    plot_complex,
    plot_beam,
]


@pytest.fixture
def field():
    return np.array(
        [
            [1 + 1j, 2 - 1j],
            [3 + 2j, 4 - 3j],
        ]
    )


#############################
# Successful execution
#############################

@pytest.mark.parametrize("func", PLOT_FUNCS)
def test_default_plot(func, field):
    fig, ax = func(field)

    assert fig is not None
    assert ax is not None

    plt.close(fig)


@pytest.mark.parametrize("func", PLOT_FUNCS)
def test_plot_with_physical_coordinates(func, field):
    fig, ax = func(field, dx=1e-6, dy=2e-6)

    assert fig is not None
    assert ax is not None

    plt.close(fig)


@pytest.mark.parametrize("func", PLOT_FUNCS)
def test_accepts_numpy_scalars(func, field):
    fig, _ = func(
        field,
        dx=np.float64(1e-6),
        dy=np.int64(2),
    )

    plt.close(fig)


#############################
# field validation
#############################

@pytest.mark.parametrize("func", PLOT_FUNCS)
def test_field_must_be_ndarray(func):
    with pytest.raises(TypeError, match="field must be an np.ndarray"):
        func([[1 + 1j]])


@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize(
    "bad",
    [
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1 + np.inf * 1j]]),
    ],
)
def test_field_must_be_finite(func, bad):
    with pytest.raises(ValueError, match="field must be finite"):
        func(bad)


#############################
# title validation
#############################

@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", [1, 1.5, [], {}, object()])
def test_title_type(func, field, bad):
    with pytest.raises(TypeError, match="title"):
        func(field, title=bad)


#############################
# cmap validation
#############################

@pytest.mark.parametrize(
    "func, bad",
    [
        (plot_intensity, 123),
        (plot_phase, 123),
        (plot_complex, 123),
        (plot_beam, "viridis"),
    ],
)
def test_cmap_wrong_type(func, field, bad):
    with pytest.raises(TypeError, match="cmap"):
        func(field, cmap=bad)


#############################
# dx validation
#############################

@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", ["1", [], {}, object()])
def test_dx_type(func, field, bad):
    with pytest.raises(TypeError, match="dx"):
        func(field, dx=bad)


@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_dx_must_be_finite(func, field, bad):
    with pytest.raises(ValueError, match="dx must be finite"):
        func(field, dx=bad)


@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", [0, -1, -1e-6])
def test_dx_positive(func, field, bad):
    with pytest.raises(ValueError, match="dx must be positive"):
        func(field, dx=bad)


#############################
# dy validation
#############################

@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", ["1", [], {}, object()])
def test_dy_type(func, field, bad):
    with pytest.raises(TypeError, match="dy"):
        func(field, dy=bad)


@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_dy_must_be_finite(func, field, bad):
    with pytest.raises(ValueError, match="dy must be finite"):
        func(field, dy=bad)


@pytest.mark.parametrize("func", PLOT_FUNCS)
@pytest.mark.parametrize("bad", [0, -2, -1e-9])
def test_dy_positive(func, field, bad):
    with pytest.raises(ValueError, match="dy must be positive"):
        func(field, dy=bad)


#############################
# Plot-specific behaviour
#############################

def test_plot_complex_returns_two_axes(field):
    fig, ax = plot_complex(field, dx=1, dy=1)

    assert ax.shape == (2,)

    plt.close(fig)


def test_plot_beam_returns_four_axes(field):
    fig, ax = plot_beam(field, dx=1, dy=1)

    assert ax.shape == (2, 2)

    plt.close(fig)


def test_plot_beam_requires_three_colormaps(field):
    with pytest.raises(IndexError):
        plot_beam(field, cmap=["inferno"])
