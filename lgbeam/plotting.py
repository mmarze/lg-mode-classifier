import numpy as np
import matplotlib.pyplot as plt


def plot_intensity(field: np.ndarray, title="Intensity", cmap="inferno"):
    """
    Plot the intensity of a complex electric field.

    Parameters
    ----------
    field : np.ndarray
        Complex electric-field amplitude.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    None
    """

    intensity = np.abs(field)**2

    plt.figure(figsize=(6, 5))
    plt.imshow(intensity, cmap=cmap)
    plt.colorbar(label="Intensity $|U|^2$")
    plt.title(title)
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.tight_layout()
    plt.show()


def plot_phase(field: np.ndarray, title="Phase", cmap="twilight"):
    """
    Plot the phase distribution of a complex electric field.

    Parameters
    ----------
    field : np.ndarray
        Complex electric-field amplitude.

    Returns
    -------
    None
    """

    phase = np.angle(field)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        phase,
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi
    )

    plt.colorbar(label="Phase [rad]")
    plt.title(title)
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
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
    