import numpy as np
from math import factorial
from scipy.special import eval_genlaguerre


def C_lp(p: int, l: int) -> float:
    """
    Calculate the required normalization constant for the generalized Laguerre polynomial.

    Parameters
    ----------
    p : int
        The radial index.  Must be non-negative.
    l : int
        The azimuthal index. Must be integer.

    Returns
    -------
    C_lp : float
        The normalization constant for the generalized Laguerre polynomial.

    Raises
    ------
    TypeError
        If l or p is not a real number or is not an integer.
    ValueError
        If p < 0.
    """
    
    # ---------- Type checking ----------
    if not isinstance(p, (int, np.integer)):
            raise TypeError(
                f"p must be an integer, got {type(p).__name__}."
            )

    if not isinstance(l, (int, np.integer)):
            raise TypeError(
                f"l must be an integer, got {type(l).__name__}."
            )
    
    # ---------- Value checking ----------
    if not np.isfinite(p):
        raise ValueError("p must be finite.")

    if not np.isfinite(l):
        raise ValueError("l must be finite.")

    if p < 0:
        raise ValueError("p must be non-negative.")
    
    # ---------- Calculate the normalization constant ----------
    
    clp = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + abs(l))))

    return clp


def L_lp(p: int, l: int, r, w_z) -> float:
    """
    Calculate the required normalization constant for the generalized Laguerre polynomial.

    Parameters
    ----------
    p : int
        The radial index.  Must be non-negative.
    l : int
        The azimuthal index. Must be integer.
    r : int or float
        The radial coordinate (in meters).
    w_z: int or float
        The beam width w(z) at the position z from the beam waist.

    Returns
    -------
    L_lp : float
        The normalization constant for the generalized Laguerre polynomial.

    Raises
    ------
    TypeError
        If l or p is not a real number or is not an integer.
    ValueError
        If p < 0.
        If r < 0.
        If w_z < 0.
    """
    
    # ---------- Type checking ----------
    if not isinstance(p, (int, np.integer)):
            raise TypeError(
                f"p must be an integer, got {type(p).__name__}."
            )

    if not isinstance(l, (int, np.integer)):
            raise TypeError(
                f"l must be an integer, got {type(l).__name__}."
            )

    if not isinstance(r, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"r must be a real number, got {type(r).__name__}."
        )

    if not isinstance(w_z, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"w(z) must be a real number, got {type(w_z).__name__}."
        )

    # ---------- Value checking ----------
    if not np.isfinite(p):
        raise ValueError("p must be finite.")

    if not np.isfinite(l):
        raise ValueError("l must be finite.")

    if not np.isfinite(r):
        raise ValueError("r must be finite.")

    if not np.isfinite(w_z):
        raise ValueError("w(z) must be finite.")

    if p < 0:
        raise ValueError("p must be non-negative.")

    if r < 0:
        raise ValueError("r must be non-negative.")

    if w_z <= 0:
        raise ValueError("w(z) must be positive.")   
    
    # ---------- Calculate the normalization constant ----------
    
    x = 2 * r**2 / w_z**2
    llp = eval_genlaguerre(p, abs(l), x)
    
    return llp
