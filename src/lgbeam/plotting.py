import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def plot_intensity(field: np.ndarray, title="Intensity", cmap="inferno", 
                   dx: int | float | None = None, dy: int | float | None = None):
    """
    Plot the intensity of a complex electric field.

    Parameters
    ----------
    field : np.ndarray
        Complex electric-field amplitude.
    title : str
        Plot title (default: "Intensity").
    cmap : str
        Matplotlib colormap (default: inferno).
    dx : int or float
        X pixel size in meters. If not provided, the axis is in pixels. Musy be positive.
    dy : int or float
        Y pixel size in meters. If not provided, the axis is in pixels. Must be positive.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.
    ax : matplotlib.axes.Axes
        The Matplotlib axes containing the plotted intensity.
    """

    # ---------- Type checking ----------
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"field must be an np.ndarray, got {type(field).__name__}."
        )

    if not isinstance(title, (str,type(None))):
        raise TypeError(
            f"title must be a string, got {type(title).__name__}."
        )

    if not isinstance(cmap, (str, type(None))):
        raise TypeError(
            f"cmap must be a string, got {type(cmap).__name__}."
        )

    if not isinstance(dx, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dx must be a real number, got {type(dx).__name__}."
        )

    if not isinstance(dy, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dy must be a real number, got {type(dy).__name__}."
        )

    # ---------- Value checking ----------
    if not np.all(np.isfinite(field)):
        raise ValueError("field must be finite.")

    if dx is not None and not np.isfinite(dx):
        raise ValueError("dx must be finite.")

    if dy is not None and not np.isfinite(dy):
        raise ValueError("dy must be finite.")

    if dx is not None and dx <= 0:
        raise ValueError("dx must be positive.")

    if dy is not None and dy <= 0:
        raise ValueError("dy must be positive.")

    # ---------- Plot ----------

    intensity = np.abs(field)**2

    fig, ax = plt.subplots(figsize=(6, 5))

    if dx is not None and dy is not None:
        ny, nx = intensity.shape
        extent = [- nx * dx / 2, nx * dx / 2, - ny * dy / 2, ny * dy / 2]
        im = ax.imshow(intensity, cmap=cmap, extent=extent)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax.yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax.tick_params(axis='x', labelrotation=45)
    else:
        im = ax.imshow(intensity, cmap=cmap)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

    fig.colorbar(im, label="Intensity $|U|^2$")
    plt.title(title)
    plt.tight_layout()

    return fig, ax


def plot_phase(field: np.ndarray, title="Phase", cmap="inferno",
               dx: int | float | None = None, dy: int | float | None = None):
    """
    Plot the phase distribution of a complex electric field.

    Parameters
    ----------
    field : np.ndarray
        Complex electric-field amplitude.
     title : str
            Plot title (default: "Phase").
    cmap : str
        Matplotlib colormap (default: inferno).
    dx : int or float
        X pixel size in meters. If not provided, the axis is in pixels. Musy be positive.
    dy : int or float
        Y pixel size in meters. If not provided, the axis is in pixels. Must be positive.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.
    ax : matplotlib.axes.Axes
        The Matplotlib axes containing the plotted intensity.
    """

    # ---------- Type checking ----------
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"field must be an np.ndarray, got {type(field).__name__}."
        )

    if not isinstance(title, (str, type(None))):
        raise TypeError(
            f"title must be a string, got {type(title).__name__}."
        )

    if not isinstance(cmap, (str, type(None))):
        raise TypeError(
            f"cmap must be a string, got {type(cmap).__name__}."
        )

    if not isinstance(dx, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dx must be a real number, got {type(dx).__name__}."
        )

    if not isinstance(dy, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dy must be a real number, got {type(dy).__name__}."
        )

    # ---------- Value checking ----------
    if not np.all(np.isfinite(field)):
        raise ValueError("field must be finite.")

    if dx is not None and not np.isfinite(dx):
        raise ValueError("dx must be finite.")

    if dy is not None and not np.isfinite(dy):
        raise ValueError("dy must be finite.")

    if dx is not None and  dx <= 0:
        raise ValueError("dx must be positive.")

    if dy is not None and  dy <= 0:
        raise ValueError("dy must be positive.")

    # ---------- Plot ----------

    phase = np.angle(field)

    fig, ax = plt.subplots(figsize=(6, 5))

    if dx is not None and dy is not None:
        ny, nx = phase.shape
        extent = [- nx * dx / 2, nx * dx / 2, - ny * dy / 2, ny * dy / 2]
        im = ax.imshow(phase, cmap=cmap, extent=extent)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax.yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax.tick_params(axis='x', labelrotation=45)
    else:
        im = ax.imshow(phase, cmap=cmap)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

    fig.colorbar(im, label="Phase [rad]")
    plt.title(title)
    plt.tight_layout()

    return fig, ax


def plot_complex(field: np.ndarray, title="Complex field", cmap="RdBu",
               dx: int | float | None = None, dy: int | float | None = None):
    """
    Plot real and imaginary components of a complex field.

    Parameters
    ----------
    field : np.ndarray
            Complex electric-field amplitude.
    title : str
        Plot title (default: "Complex field").
    cmap : str
        Matplotlib colormap (default: RdBu).
    dx: int or float
        X pixel size in meters. If not provided, the axis is in pixels. Musy be positive.
    dy: int or float
        Y pixel size in meters. If not provided, the axis is in pixels. Must be positive.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.

    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of Matplotlib axes containing the generated subplots.
    """

    # ---------- Type checking ----------
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"field must be an np.ndarray, got {type(field).__name__}."
        )

    if not isinstance(title, (str, type(None))):
        raise TypeError(
            f"title must be a string, got {type(title).__name__}."
        )

    if not isinstance(cmap, (str, type(None))):
        raise TypeError(
            f"cmap must be a string, got {type(cmap).__name__}."
        )

    if not isinstance(dx, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dx must be a real number, got {type(dx).__name__}."
        )

    if not isinstance(dy, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dy must be a real number, got {type(dy).__name__}."
        )

    # ---------- Value checking ----------
    if not np.all(np.isfinite(field)):
        raise ValueError("field must be finite.")

    if dx is not None and not np.isfinite(dx):
        raise ValueError("dx must be finite.")

    if dy is not None and not np.isfinite(dy):
        raise ValueError("dy must be finite.")

    if dx is not None and dx <= 0:
        raise ValueError("dx must be positive.")

    if dy is not None and  dy<= 0:
        raise ValueError("dy must be positive.")

    # ---------- Plot ----------

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    real = np.real(field)
    imag = np.imag(field)

    if dx is not None and dy is not None:
        ny, nx = real.shape
        extent = [- nx * dx / 2, nx * dx / 2, - ny * dy / 2, ny * dy / 2]
        im0 = ax[0].imshow(real, cmap=cmap, extent=extent)
        ax[0].set_title("Real part")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")
        ax[0].xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[0].yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[0].tick_params(axis='x', labelrotation=45)
        fig.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(imag, cmap=cmap, extent=extent)
        ax[1].set_title("Imaginary part")
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Y")
        ax[1].xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[1].yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[1].tick_params(axis='x', labelrotation=45)
        fig.colorbar(im1, ax=ax[1])

    else:
        im0 = ax[0].imshow(real, cmap=cmap)
        ax[0].set_title("Real part")
        fig.colorbar(im0, ax=ax[0])
    
        im1 = ax[1].imshow(imag, cmap=cmap)
        ax[1].set_title("Imaginary part")
        fig.colorbar(im1, ax=ax[1])

    plt.suptitle(title)
    plt.tight_layout()
    return fig, ax


def plot_beam(field: np.ndarray, title="Laguerre-Gaussian beam", cmap=["inferno", "inferno", "RdBu"],
               dx: int | float | None = None, dy: int | float | None = None):
    """
    Display intensity, phase, real and imaginary parts.

    Parameters
    ----------
    field : np.ndarray
                Complex electric-field amplitude.
    title : str
        Plot title (default: "Laguerre-Gaussian beam").
    cmap : list of str
        Matplotlib colormap (default: inferno (intensity), inferno (phase), RdBu (complex field)).
    dx : int or float
        X pixel size in meters. If not provided, the axis is in pixels. Musy be positive.
    dy : int or float
        Y pixel size in meters. If not provided, the axis is in pixels. Must be positive.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.

    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of Matplotlib axes containing the generated subplots.
    """

    # ---------- Type checking ----------
    if not isinstance(field, np.ndarray):
        raise TypeError(
            f"field must be an np.ndarray, got {type(field).__name__}."
        )

    if not isinstance(title, (str, type(None))):
        raise TypeError(
            f"title must be a string, got {type(title).__name__}."
        )

    if not isinstance(cmap, (list, type(None))):
        raise TypeError(
            f"cmap must be a list of strings, got {type(cmap).__name__}."
        )

    if not isinstance(dx, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dx must be a real number, got {type(dx).__name__}."
        )

    if not isinstance(dy, (int, float, np.integer, np.floating, type(None))):
        raise TypeError(
            f"dy must be a real number, got {type(dy).__name__}."
        )

    # ---------- Value checking ----------
    if not np.all(np.isfinite(field)):
        raise ValueError("field must be finite.")

    if dx is not None and not np.isfinite(dx):
        raise ValueError("dx must be finite.")

    if dy is not None and not np.isfinite(dy):
        raise ValueError("dy must be finite.")

    if dx is not None and dx <= 0:
        raise ValueError("dx must be positive.")

    if dy is not None and dy <= 0:
        raise ValueError("dy must be positive.")

    # ---------- Plot ----------

    intensity = np.abs(field)**2
    phase = np.angle(field)

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(10, 9)
    )

    if dx is not None and dy is not None:
        ny, nx = phase.shape
        extent = [- nx * dx / 2, nx * dx / 2, - ny * dy / 2, ny * dy / 2]
        im0 = ax[0,0].imshow(intensity, cmap=cmap[0], extent=extent)
        ax[0,0].set_title("Intensity")
        ax[0,0].set_xlabel("X")
        ax[0,0].set_ylabel("Y")
        ax[0,0].xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[0,0].yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[0,0].tick_params(axis='x', labelrotation=45)
        fig.colorbar(im0, ax=ax[0,0])

        im1 = ax[0,1].imshow(phase, 
                             cmap=cmap[1],
                             vmin=-np.pi,
                             vmax=np.pi, 
                             extent=extent)
        ax[0,1].set_title("Phase")
        ax[0,1].set_xlabel("X")
        ax[0,1].set_ylabel("Y")
        ax[0,1].xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[0,1].yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[0,1].tick_params(axis='x', labelrotation=45)
        fig.colorbar(im1, ax=ax[0,1])

        im2 = ax[1,0].imshow(np.real(field), cmap=cmap[2], extent=extent)
        ax[1,0].set_title("Real part")
        ax[1,0].set_xlabel("X")
        ax[1,0].set_ylabel("Y")
        ax[1,0].xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[1,0].yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[1,0].tick_params(axis='x', labelrotation=45)
        fig.colorbar(im2, ax=ax[1,0])

        im3 = ax[1,1].imshow(np.imag(field), cmap=cmap[2], extent=extent)
        ax[1,1].set_title("Imaginary part")
        ax[1,1].set_xlabel("X")
        ax[1,1].set_ylabel("Y")
        ax[1,1].xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[1,1].yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax[1,1].tick_params(axis='x', labelrotation=45)
        fig.colorbar(im3, ax=ax[1,1])

    else:
        # Intensity
        im0 = ax[0, 0].imshow(
            intensity,
            cmap=cmap[0]
        )
        ax[0,0].set_title("Intensity")
        fig.colorbar(im0, ax=ax[0,0])
    
        # Phase
        im1 = ax[0, 1].imshow(
            phase,
            cmap=cmap[1],
            vmin=-np.pi,
            vmax=np.pi
        )
        ax[0,1].set_title("Phase")
        fig.colorbar(im1, ax=ax[0,1])
    
        # Real
        im2 = ax[1,0].imshow(
            np.real(field),
            cmap=cmap[2]
        )
        ax[1,0].set_title("Real part")
        fig.colorbar(im2, ax=ax[1,0])
    
        # Imaginary
        im3 = ax[1,1].imshow(
            np.imag(field),
            cmap=cmap[2]
        )
        ax[1,1].set_title("Imaginary part")
        fig.colorbar(im3, ax=ax[1,1])

    fig.suptitle(title)
    plt.tight_layout()
    return fig, ax
