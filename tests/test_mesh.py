import numpy as np
import pytest

from lgbeam.mesh import create_mesh


def test_create_mesh_shape():
    r, phi = create_mesh(1.0, 10)

    assert r.shape == (10, 10)
    assert phi.shape == (10, 10)


def test_create_mesh_radius_values():
    r, _ = create_mesh(1.0, 3)

    expected_r = np.array([
        [np.sqrt(2), 1, np.sqrt(2)],
        [1, 0, 1],
        [np.sqrt(2), 1, np.sqrt(2)]
    ])

    np.testing.assert_allclose(r, expected_r)


def test_create_mesh_phi_range():
    _, phi = create_mesh(1.0, 100)

    assert np.all(phi <= np.pi)
    assert np.all(phi >= -np.pi)


@pytest.mark.parametrize(
    "L",
    [-1, 0, np.inf, np.nan]
)
def test_create_mesh_invalid_L(L):

    with pytest.raises(ValueError):
        create_mesh(L, 10)


def test_create_mesh_invalid_N():

    with pytest.raises(ValueError):
        create_mesh(1.0, 1)


def test_create_mesh_type_error():

    with pytest.raises(TypeError):
        create_mesh("1", 10)

    with pytest.raises(TypeError):
        create_mesh(1.0, 2.5)
        