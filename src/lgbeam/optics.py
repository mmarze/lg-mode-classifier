import numpy as np


def Rayleigh_range(w0: int | float, n: int | float, wavelength: float) -> float:
    """
    Calculate the beam's Rayleigh range.
    
    Parameters
    ----------
    w0 : int or float
        The waist radius in meters. Must be positive.
    n : int or float
        The index of refraction of the medium. 
    wavelength: float
        The wavelength of the beam in meters (range from 100 nm to 1 mm).

    Returns
    -------
    zR : int or float
        The beam's Rayleigh range (in meters).

    Raises
    ------
    TypeError
        If w0, n, or wavelength is not a real number or is not a float.
    ValueError
        If w0 or n <= 0.
        If wavelength > 1 mm or wavelength < 100 nm.
    """

    # ---------- Type checking ----------
    if not isinstance(w0, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Beam waist radius w0 must be a real number, got {type(w0).__name__}."
        )

    if not isinstance(n, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Refractive index n must be a real number, got {type(n).__name__}."
        )

    if not isinstance(wavelength, (float, np.floating)):
        raise TypeError(
            f"Wavelength must be a real number, got {type(wavelength).__name__}."
        )

    # ---------- Value checking ----------
    if not np.isfinite(w0):
        raise ValueError("Beam waist radius w0 must be finite.")

    if not np.isfinite(n):
        raise ValueError("Refractive index n must be finite.")

    if not np.isfinite(wavelength):
        raise ValueError("Wavelength must be finite.")

    if w0 <= 0:
        raise ValueError("Beam waist radius w0 must be positive.")

    if n <= 0:
        raise ValueError("Refractive index n must be positive.")

    if not (100e-9 <= wavelength <= 1e-3):
        raise ValueError(
            "wavelength must correspond to optical radiation "
            "(approximately 100 nm to 1 mm, expressed in meters)."
        )

    # ---------- Calculate the Rayleigh range ----------
    zr = np.pi * w0 **2 * n / wavelength
    
    return zr


def radius_of_curvature(z: int | float, z_r: int | float) -> float:
    """
    Calculate the radius of curvature R(z) of the beam's wavefronts at z.
    
    Parameters
    ----------
    z : int or float
        The axial distance from the beam's focus (or 'waist') in meters.
    z_r : int or float
        The Rayleigh range of the beam in meters.  Must be positive.

    Returns
    -------
    R(z) : float
        The radius of curvature R(z) of the beam's wavefronts at z (in meters).

    Raises
    ------
    TypeError
        If z_r or z is not a real number or is not a float or is not an integer.
    ValueError
        If z_r <= 0.
    """

    # ---------- Type checking ----------
    if not isinstance(z_r, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Rayleigh range z_R must be a real number, got {type(z_r).__name__}."
        )

    if not isinstance(z, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Distance from beam waist z must be a real number, got {type(z).__name__}."
        )

    # ---------- Value checking ----------
    if not np.isfinite(z_r):
        raise ValueError("Rayleigh range z_r must be finite.")

    if not np.isfinite(z):
        raise ValueError("Distance from beam waist z must be finite.")

    if z_r <= 0:
        raise ValueError("Rayleigh range z_r must be positive.")
    
    # ---------- Calculate radius of curvature R(z) ----------
    
    if z == 0:
        return np.inf
        
    rz = z * (1 + (z_r / z)**2)

    return rz


def beam_width_z(w0: int | float, z: int | float, z_r: int | float) -> float:
    """
    Calculate the beam width w(z) at the position z from the beam waist.
    
    Parameters
    ----------
    w0 : int or float
        The waist radius in meters. Must be positive.
    z : int or float
        The axial distance from the beam's focus (or 'waist').
    z_r : int or float
        The Rayleigh range of the beam. Must be positive.

    Returns
    -------
    w(z) : float
        The beam width w(z) at the position z from the beam waist.

    Raises
    ------
    TypeError
        If z_r or z or w0 is not a real number or is not a float or is not an integer.
    ValueError
        If z_r <= 0.
        If w0 <= 0.
    """

    # ---------- Type checking ----------
    if not isinstance(w0, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Beam waist radius w0 must be a real number, got {type(w0).__name__}."
        )

    if not isinstance(z_r, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Rayleigh range z_r must be a real number, got {type(z_r).__name__}."
        )

    if not isinstance(z, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Distance from beam waist z must be a real number, got {type(z).__name__}."
        )

    # ---------- Value checking ----------
    if not np.isfinite(w0):
        raise ValueError("Beam waist radius w0 must be finite.")

    if not np.isfinite(z_r):
        raise ValueError("Rayleigth range z_r must be finite.")

    if not np.isfinite(z):
        raise ValueError("Distance from beam waist z must be finite.")

    if z_r <= 0:
        raise ValueError("Rayleigth range z_r must be positive.")

    if w0 <= 0:
        raise ValueError("Beam waist w0 must be positive.")
    
    # ---------- Calculate the beam radius w(z) ----------

    wz = w0 * np.sqrt(1 + (z / z_r)**2)

    return wz


def wavenumber(wavelength: float) -> float:
    """
    Calculate an angular wavenumber, defined as the number of radians per unit distance.

    Parameters
    ----------
    wavelength : float
        The wavelength of the beam in meters (range from 100 nm to 1 mm).

    Returns
    -------
    k : float
        The angular wavenumber (in radians per meter).

    Raises
    ------
    TypeError
        If wavelength is not a real number or is not a float.
    ValueError
        If wavelength > 1 mm or wavelength < 100 nm.

    """

    # ---------- Type checking ----------
    if not isinstance(wavelength, (float, np.floating)):
        raise TypeError(
            f"Wavelength must be a real number, got {type(wavelength).__name__}."
        )

    # ---------- Value checking ----------
    if not np.isfinite(wavelength):
        raise ValueError("Wavelength must be finite.")

    if not (100e-9 <= wavelength <= 1e-3):
        raise ValueError(
            "Wavelength must correspond to optical radiation "
            "(approximately 100 nm to 1 mm, expressed in meters)."
        )
    
    # ---------- Calculate the wavenumer ----------

    k = 2 * np.pi / wavelength

    return k


def psi(l: int, p: int, z: int | float, z_r: int | float) -> float:
    """
    Calculate the magnitude of the Laguerre-Gaussian modes' Gouy phase shift.
    
    Parameters
    ----------
    l : int
        The azimuthal index. Must be integer.
    p : int
        The radial index. Must be non-negative.
    z : int or float
            The axial distance from the beam's focus (or 'waist') in meters.
    z_r : int or float
        The Rayleigh range of the beam in meters.  Must be positive.

    Returns
    -------
    Psi(z) : float
        The magnitude of the Laguerre-Gaussian modes' Gouy phase shift (in radians).

    Raises
    ------
    TypeError
        If l or p is not a real number or is not an integer
        If z_r or z is not a real number or is not a float or is not an integer.
    ValueError
        If p < 0.
        If z_r <= 0.
    """

    # ---------- Type checking ----------
    if not isinstance(z_r, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Rayleigh range z_r must be a real number, got {type(z_r).__name__}."
        )

    if not isinstance(z, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"Distance from beam waist z must be a real number, got {type(z).__name__}."
        )

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

    if not np.isfinite(z_r):
        raise ValueError("Rayleigh range z_r must be finite.")

    if not np.isfinite(z):
        raise ValueError("Distance from beam waist z must be finite.")

    if p < 0:
        raise ValueError("p must be non-negative.")

    if z_r <= 0:
        raise ValueError("Rayleigh range z_r must be positive.")
    
    # ---------- Calculate the magnitude of the Laguerre-Gaussian modes' Gouy phase shift ----------
    psi_z = (np.abs(l) + 2 * p + 1) * np.arctan(z / z_r)

    return psi_z
