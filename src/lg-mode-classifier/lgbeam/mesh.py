import numpy as np


def create_mesh(L: int | float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a polar coordinate mesh over the square [-L, L] x [-L, L].

    Parameters
    ----------
    L : int or float
        Half-length of the domain (in meters). Must be positive.
    N : int
        Number of points along each axis. Must be at least 2.

    Returns
    -------
    r : np.ndarray
        Radius values with shape (N, N).
    phi : np.ndarray
        Angle values in radians with shape (N, N), in the range [-pi, pi].

    Raises
    ------
    TypeError
        If L is not a real number or N is not an integer.
    ValueError
        If L <= 0 or N < 2.
    """

    # ---------- Type checking ----------
    if not isinstance(L, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"L must be a real number, got {type(L).__name__}."
        )

    if not isinstance(N, (int, np.integer)):
        raise TypeError(
            f"N must be an integer, got {type(N).__name__}."
        )

    # ---------- Value checking ----------
    if not np.isfinite(L):
        raise ValueError("L must be finite.")

    if L <= 0:
        raise ValueError("L must be positive.")

    if N < 2:
        raise ValueError("N must be at least 2.")

    # ---------- Create Cartesian mesh ----------
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)

    X, Y = np.meshgrid(x, y, indexing="xy")

    # ---------- Convert to polar ----------
    r = np.hypot(X, Y)
    phi = np.arctan2(Y, X)

    return r, phi
