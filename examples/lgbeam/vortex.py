from lgbeam.mesh import create_mesh
from lgbeam.beams import LG
from lgbeam.plotting import plot_intensity, plot_phase


# Create mesh
r, phi = create_mesh(
    L=1e-3, 
    N=512
)

# Calculate Laguerre-Gauss beam
beam = LG(
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
plot_intensity(beam, title="Intenisty, $LG_{10}$", dx=pixel_size, dy=pixel_size)

# Plot beam phase
pixel_size = 2 * 1e-3 / 512
plot_phase(beam, title="Phase, $LG_{10}$",dx=pixel_size, dy=pixel_size)
