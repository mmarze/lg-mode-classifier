import numpy as np
import json
import matplotlib.pyplot as plt

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.plotting import plot_intensity
from lgbeam.dataset.config import WAVELENGTH, EXPOSURE, W0_RANGE, L, N


class Sensor:
    def __init__(
            self, 
            width, 
            height, 
            pixel_size, 
            qe, 
            read_noise,
            dark_current,
            full_well,
            adc_bits,
            seed
        ):
        """
        Generic CMOS sensor model.

        Parameters
        ----------
        width : int
            Number of pixels horizontally.

        height : int
            Number of pixels vertically.

        pixel_size : int or float
            Pixel size in meters.

        qe : float
            Quantum efficiency (0-1).

        read_noise : int or float
            Read noise RMS in electrons.

        dark_current : int or float
            Dark current in electrons/pixel/second.

        full_well : int or float
            Maximum electrons per pixel.

        adc_bits : int
            ADC resolution.

        seed : int, optional
            Random seed for reproducibility.
        """

        # ---------- Type checking ----------
        if not isinstance(width, (int, np.integer)):
                raise TypeError(
                    f"width must be an integer, got {type(width).__name__}."
                )

        if not isinstance(height, (int, np.integer)):
            raise TypeError(
                f"height must be an integer, got {type(height).__name__}."
            )

        if not isinstance(pixel_size, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"pixel_size must be a number, got {type(pixel_size).__name__}."
            )

        if not isinstance(qe, (float, np.floating)):
            raise TypeError(
                f"qe must be a float, got {type(qe).__name__}."
            )

        if not isinstance(read_noise, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"read_noise must be a number, got {type(read_noise).__name__}."
            )

        if not isinstance(dark_current, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"read_noise must be a number, got {type(read_noise).__name__}."
            )

        if not isinstance(full_well, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"full_well must be a number, got {type(full_well).__name__}."
            )

        if not isinstance(adc_bits, (int, np.integer)):
            raise TypeError(
                f"adc_bits must be an integer, got {type(adc_bits).__name__}."
            )

        if not isinstance(seed, (int, np.integer, type(None))):
            raise TypeError(
                f"seed must be an integer, got {type(seed).__name__}."
            )

        # ---------- Value checking ----------
        if not np.isfinite(width):
            raise ValueError("Sensor width must be finite.")

        if not np.isfinite(height):
            raise ValueError("Sensor height must be finite.")

        if not np.isfinite(pixel_size):
            raise ValueError("Sensor pixel size must be finite.")

        if not np.isfinite(qe):
            raise ValueError("Sensor quantum efficiency must be finite.")

        if not np.isfinite(read_noise):
            raise ValueError("Sensor read noise must be finite.")

        if not np.isfinite(dark_current):
            raise ValueError("Sensor dark current must be finite.")

        if not np.isfinite(full_well):
            raise ValueError("Sensor full well must be finite.")

        if not np.isfinite(adc_bits):
            raise ValueError("Sensor ADC bits must be finite.")

        if seed is not None and not np.isfinite(seed):
            raise ValueError("Seed must be finite.")

        if width <= 0:
            raise ValueError("Width must be positive")

        if height <= 0:
            raise ValueError("Height must be positive")

        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")

        if not 0 < qe <= 1:
            raise ValueError("QE must be between 0 and 1")

        if read_noise < 0:
             raise ValueError("Read noise cannot be negative")

        if dark_current < 0:
            raise ValueError("Dark current cannot be negative")

        if full_well <= 0:
            raise ValueError("Full well must be positive")

        if adc_bits <= 0:
            raise ValueError("ADC bits must be positive")

        # ---------- Values  ----------
        self.width = width
        self.height = height
        self.pixel_size = pixel_size

        # Sensor physics
        self.qe = qe
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.full_well = full_well
        # Electronics
        self.adc_bits = adc_bits
        # Random generator
        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def capture(self, photons, exposure):

        # Photons to electrons
        electrons = self.rng.poisson(
            photons * self.qe
        ).astype(np.float64)

        # Dark current noise
        dark_electrons = self.rng.poisson(
            self.dark_current * exposure,
            size=photons.shape
        )

        electrons += dark_electrons

        # Full well saturation
        electrons = np.clip(electrons, 0, self.full_well)

        # Read noise 
        electrons += self.rng.normal(0, self.read_noise, size=photons.shape)

        # ADC conversion
        max_dn = (2 ** self.adc_bits) - 1
        conversion_gain = max_dn / self.full_well
        image = electrons * conversion_gain

        # Digital limits
        image = np.clip(image, 0, max_dn)

        # Convert to integer image
        image = np.round(image).astype(np.uint16)

        return image


    def save_config(self, filename):

        config = {
            "width": self.width,
            "height": self.height,
            "pixel_size": self.pixel_size,
            "qe": self.qe,
            "read_noise": self.read_noise,
            "dark_current": self.dark_current,
            "full_well": self.full_well,
            "adc_bits": self.adc_bits,
            "seed": self.seed
        }

        with open(filename, "w") as f:
            json.dump(config, f, indent=4)


if __name__ == '__main__':
    sony_imx541 = Sensor(
                    width=4512, 
                    height=4576, 
                    pixel_size=2.74e-6, 
                    qe=0.68,
                    read_noise=2.3, 
                    dark_current=22, 
                    full_well=9_400, 
                    adc_bits=12,
                    seed=None
    )

    r, phi = create_mesh(L, N)
    photons = LaguerreGauss(p=0, 
                            l=0, 
                            r=r, 
                            phi=phi, 
                            w0=(W0_RANGE[0]+W0_RANGE[1])/2 , 
                            z=0, 
                            wavelength=WAVELENGTH
                            )

    photons = np.abs(photons) / 2.5
    
    image = sony_imx541.capture(photons, EXPOSURE)
    fig, ax = plot_intensity(image)
    plt.show()
