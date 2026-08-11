import matplotlib.pyplot as plt
from pathlib import Path

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.plotting import plot_intensity, plot_complex, plot_beam

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


# Create mesh
r, phi = create_mesh(
    L=1e-3, 
    N=512
)

# Calculate Laguerre-Gauss beam
beam = LaguerreGauss(
    p=0, 
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

fig1, axis1 = plot_intensity(beam, title="Intenisty, $LG_{00}$" , dx=pixel_size, dy=pixel_size)
# plt.show()
# plt.close(fig1)
plt.savefig(output_dir/'Intensity_LG00.png', dpi=600)

fig2, axis2 = plot_complex(beam, title="Complex field, $LG_{00}$" , dx=pixel_size, dy=pixel_size)
# plt.show()
# plt.close(fig2)
plt.savefig(output_dir/'Complex_field_LG00.png', dpi=600)

fig3, ax3 = plot_beam(beam, title="Laguerre-Gaussian beam, $LG_{00}$" , dx=pixel_size, dy=pixel_size)
# plt.show()
# plt.close(fig3)
plt.savefig(output_dir/'Plot_beam_LG00.png', dpi=600)
