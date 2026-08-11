import numpy as np
import matplotlib.pyplot as plt

from lgbeam.mesh import create_mesh
from lgbeam.optics import Rayleigh_range
from lgbeam.beams import LaguerreGauss
from lgbeam.plotting import plot_intensity


# Create mesh
r, phi = create_mesh(
    L=1e-3, 
    N=512
)

z_r = Rayleigh_range(w0=500e-6, n=1.0, wavelength=532e-9)
for z in np.linspace(0, 2*z_r, 6):

    # Calculate Laguerre-Gauss beam
    beam = LaguerreGauss(
        p=0, 
        l=1, 
        r=r,
        phi=phi,
        z=z,
        w0=500e-6,
        wavelength=532e-9,
        n=1.0
        )

    # Plot beam intensity
    pixel_size = 2 * 1e-3 / 512
    fig1, ax1 = plot_intensity(beam, title=f"Intenisty, $LG_{10}$, z={z:.2f} m", dx=pixel_size, dy=pixel_size)
    plt.show()
    plt.close(fig1)
