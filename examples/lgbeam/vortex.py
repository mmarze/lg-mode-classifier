import matplotlib.pyplot as plt

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.plotting import plot_intensity, plot_phase


# Create mesh
r, phi = create_mesh(
    L=1e-3, 
    N=512
)

# Calculate Laguerre-Gauss beam
beam = LaguerreGauss(
    p=0, 
    l=1, 
    r=r,
    phi=phi,
    z=10e-3,
    w0=500e-6,
    wavelength=532e-9,
    n=1.0
    )

# Plot beam intensity
pixel_size = 2 * 1e-3 / 512
fig1, ax1 = plot_intensity(beam, title="Intenisty, $LG_{10}$", dx=pixel_size, dy=pixel_size)
plt.show()
plt.close(fig1)

# Plot beam phase
fig2, ax2 = plot_phase(beam, title="Phase, $LG_{10}$",dx=pixel_size, dy=pixel_size)
plt.show()
plt.close(fig2)
