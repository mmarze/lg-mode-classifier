import numpy as np

from .optics import (
    Rayleigh_range,
    radius_of_curvature,
    beam_width_z,
    wavenumber,
    psi,
)

from .laguerre import (
    C_lp,
    L_lp
)


def LaguerreGauss(p: int, l: int, r: np.ndarray, phi: np.ndarray, z: int | float, w0: int | float, wavelength: float, n: int | float = 1.0) -> np.ndarray:
    """
    # Calculate the Laguerre-Gaussian mode.

    Parameters
    ----------
    p : int
        The radial index. Must be non-negative.
    l : int
        The azimuthal index.
    r : np.ndarray
        The radial coordinate (in meters). Must be non-negative.
    phi: np.ndarray
        The angular coordinate (in radians). Must be in the range [-pi, pi]
    z : int or float
        The axial distance from the beam's focus (or 'waist') in meters.
    w0 : int or float
        The waist radius in meters. Must be positive.
    wavelength: float
        The wavelength of the beam in meters (range from 100 nm to 1 mm).
    n : int or float
        The index of refraction of the medium. 

    Returns
    -------
    LG : np.ndarray
        The complex electric-field amplitude of the Laguerre-Gaussian beam.

    Raises
    ------
    TypeError
        If l or p is not a real number or is not an integer.
        If r or phi or z or w0 or wavelength or n is not a real number or is not a float or is not an integer.
    ValueError
        If p < 0.
        If r < 0.
        If phi > pi or phi < -pi.
        If w0 <= 0.
        If wavelength > 1 mm or wavelength < 100 nm.
    """
    # ---------- Type checking ----------

    if not isinstance(r, np.ndarray):
        raise TypeError(
            f"r must be a np.ndarray, got {type(r).__name__}."
        )

    if not isinstance(phi, np.ndarray):
        raise TypeError(
            f"phi must be a np.ndarray, got {type(phi).__name__}."
        )

    # ---------- Value checking ----------
   
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite.")

    if not np.all(np.isfinite(phi)):
        raise ValueError("phi must be finite.")

    if np.any(r < 0):
        raise ValueError("r must be non-negative.")

    if np.any(phi > np.pi) or np.any(phi < -np.pi):
        raise ValueError("phi must be in the range [-pi, pi].")  
          
    # ---------- Calculate the Laguerre-Gaussian mode ----------

    zr = Rayleigh_range(w0, n, wavelength)
    wz = beam_width_z(w0, z, zr)
    kz = wavenumber(wavelength)
    Rz = radius_of_curvature(z, zr)

    C = C_lp(p, l)

    rho = np.sqrt(2.0) * r / wz

    U = (
        C / wz
        * rho**abs(l)
        * L_lp(p, l, r, wz)
        * np.exp(-r**2 / wz**2)
        * np.exp(-1j * kz * r**2 / (2 * Rz))
        * np.exp(-1j * l * phi)
        * np.exp(1j * psi(l, p, z, zr))
        * np.exp(-1j * kz * z)
    )

    return U
