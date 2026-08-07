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
    dx: int or float
        X pixel size in meters. If not provided, the axis is in pixels. Musy be positive.
    dy: int or float
        Y pixel size in meters. If not provided, the axis is in pixels. Must be positive.

    Returns
    -------
    None
    """

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
    plt.show()


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
    dx: int or float
        X pixel size in meters. If not provided, the axis is in pixels. Musy be positive.
    dy: int or float
        Y pixel size in meters. If not provided, the axis is in pixels. Must be positive.

    Returns
    -------
    None
    """

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
    plt.show()


def plot_complex(field: np.ndarray, title="Complex field"):
    """
    Plot real and imaginary components of a complex field.

    Parameters
    ----------
    field : np.ndarray
        Complex electric-field amplitude.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    real = np.real(field)
    imag = np.imag(field)

    im0 = ax[0].imshow(real, cmap="RdBu")
    ax[0].set_title("Real part")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(imag, cmap="RdBu")
    ax[1].set_title("Imaginary part")
    fig.colorbar(im1, ax=ax[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_beam(field: np.ndarray, title="Laguerre-Gaussian beam"):
    """
    Display intensity, phase, real and imaginary parts.

    Parameters
    ----------
    field : np.ndarray
        Complex electric-field amplitude.

    Returns
    -------
    None
    """

    intensity = np.abs(field)**2
    phase = np.angle(field)

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(10, 9)
    )

    # Intensity
    im0 = ax[0, 0].imshow(
        intensity,
        cmap="inferno"
    )
    ax[0,0].set_title("Intensity")
    fig.colorbar(im0, ax=ax[0,0])

    # Phase
    im1 = ax[0, 1].imshow(
        phase,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi
    )
    ax[0,1].set_title("Phase")
    fig.colorbar(im1, ax=ax[0,1])

    # Real
    im2 = ax[1,0].imshow(
        np.real(field),
        cmap="RdBu"
    )
    ax[1,0].set_title("Real part")
    fig.colorbar(im2, ax=ax[1,0])

    # Imaginary
    im3 = ax[1,1].imshow(
        np.imag(field),
        cmap="RdBu"
    )
    ax[1,1].set_title("Imaginary part")
    fig.colorbar(im3, ax=ax[1,1])


    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    