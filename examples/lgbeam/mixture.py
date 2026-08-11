import matplotlib.pyplot as plt

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.plotting import plot_intensity


# Create mesh
r, phi = create_mesh(
    L=1e-3, 
    N=512
)

# Calculate Laguerre-Gauss beam
beam1 = LaguerreGauss(
    p=0, 
    l=0, 
    r=r,
    phi=phi,
    z=10e-3,
    w0=500e-6,
    wavelength=532e-9,
    n=1.0
    )

beam2 = LaguerreGauss(
    p=2, 
    l=0, 
    r=r,
    phi=phi,
    z=10e-3,
    w0=500e-6,
    wavelength=532e-9,
    n=1.0
    )

field = 0.2 * beam1 + 0.8 * beam2

# Plot beam intensity
pixel_size = 2 * 1e-3 / 512

fig1, ax1 = plot_intensity(field, title=r"Intenisty, $0.2 \cdot LG_{00} + 0.8 \cdot LG_{02}$", dx=pixel_size, dy=pixel_size)
plt.show()
plt.close(fig1)
