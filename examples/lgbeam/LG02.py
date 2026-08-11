import matplotlib.pyplot as plt
from pathlib import Path

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.plotting import plot_intensity


output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Create mesh
r, phi = create_mesh(
    L=1e-3, 
    N=512
)

# Calculate Laguerre-Gauss beam
beam = LaguerreGauss(
    p=2, 
    l=0, 
    r=r,
    phi=phi,
    z=10e-3,
    w0=500e-6,
    wavelength=532e-9,
    n=1.0
    )

# Plot the beam
pixel_size = 2 * 1e-3 / 512

fig1, ax1 = plot_intensity(beam, title="Intenisty, $LG_{02}$", dx=pixel_size, dy=pixel_size)
# plt.show()
# plt.close(fig1)
plt.savefig(output_dir/'Plot_beam_LG02.png', dpi=600)
