import numpy as np
import json
import matplotlib.pyplot as plt
import os

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
            seed=None
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


    def capture(self, photons: np.ndarray, exposure:int | float) -> np.ndarray:
        """
        Simulate an image capture with the sensor.

        The input photon image is cropped to the physical dimensions of
        the sensor. Photon shot noise, quantum efficiency, dark current,
        read noise, full-well saturation, and ADC quantization are then
        applied sequentially.

        Parameters
        ----------
        photons : numpy.ndarray
            2D array containing the number of incident photons per pixel.
            The input image must be large enough to contain the sensor
            field of view.

        exposure : int or float
            Exposure time in seconds. Must be non-negative.

        Returns
        -------
        numpy.ndarray
            2D image with dimensions ``(height, width)`` containing the
            digitized sensor output. Pixel values are stored as ``uint16``
            and range from 0 to ``2**adc_bits - 1``.

        Notes
        -----
        The simulation includes the following effects:

        1. Photon shot noise is modeled using a Poisson distribution.
        2. Quantum efficiency (QE) converts incident photons into
        photoelectrons.
        3. Dark current is modeled as Poisson-distributed electrons.
        4. The accumulated electrons are clipped at the sensor full-well
        capacity.
        5. Read noise is modeled as zero-mean Gaussian noise with a
        standard deviation equal to ``self.read_noise`` electrons.
        6. Electrons are converted to digital numbers (DN) using the ADC
        conversion gain.
        7. The resulting digital values are clipped to the ADC range and
        rounded to integer values.

        The generated random values are obtained from the sensor's
        ``numpy.random.Generator`` instance, allowing reproducible
        simulations when a seed is provided during initialization.

        The generated dark-current electrons and read-noise realization
        are stored in ``self.dark_electrons`` and
        ``self.real_read_noise``, respectively.
        """
        # ---------- Type checking ----------
        if not isinstance(photons, (np.ndarray)):
                raise TypeError(
                    f"Photons must be an numpy.ndarray, got {type(photons).__name__}."
                )

        if not isinstance(exposure, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"Exposure must be a number, got {type(exposure).__name__}."
            )

        # ---------- Value checking ----------
        if photons.ndim != 2:
            raise ValueError(f"photons must be a 2D array, got {photons.ndim} dimensions.")

        if photons.shape[0] < self.height or photons.shape[1] < self.width:
            raise ValueError("Input photon image must be at least as large as the sensor.")

        if not np.issubdtype(photons.dtype, np.number):
            raise TypeError(f"photons must contain numeric values, got {photons.dtype}.") 

        if not np.all(np.isfinite(photons)):
            raise ValueError("Photon values must be finite.")

        if np.any(photons < 0):
            raise ValueError("Photon values cannot be negative.") 

        if not np.isfinite(exposure):
            raise ValueError("Exposure time must be finite.") 

        if exposure < 0:
            raise ValueError("Exposure time cannot be negative.")

        # ---------- Image capture  ----------

        # Crop image to sensor dimensions
        self.half_x = self.width // 2
        self.half_y = self.height // 2
        image_in_half_x, image_in_half_y = photons.shape
        image_in_half_x //= 2
        image_in_half_y //= 2

        photons = photons[image_in_half_x - self.half_x:image_in_half_x - self.half_x + self.width,
                          image_in_half_y - self.half_y:image_in_half_y - self.half_y + self.height]

        # Photons to electrons
        electrons = self.rng.poisson(
            photons * self.qe
        ).astype(np.float64)

        # Dark current noise
        self.dark_electrons = self.rng.poisson(
            self.dark_current * exposure,
            size=photons.shape
        )

        electrons += self.dark_electrons
        self.total_dark_electrons = self.dark_electrons.sum()

        # Full well saturation
        electrons = np.clip(electrons, 0, self.full_well)

        # Read noise 
        self.real_read_noise = self.rng.normal(0, self.read_noise, size=photons.shape)
        electrons +=  self.real_read_noise
        self.total_real_read_noise = self.real_read_noise.sum()

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
        """
        Save the sensor configuration and capture statistics to a JSON file.

        Parameters
        ----------
        filename : str or os.PathLike
            Path to the output JSON file.

        Notes
        -----
        The saved configuration contains the sensor dimensions, pixel size,
        quantum efficiency, nominal read noise, dark current, full-well
        capacity, ADC resolution, and random seed.

        If ``capture()`` has been called before this method, the JSON file
        also contains statistics from the most recent capture:

        - ``real_read_noise_per_pxl``: mean of the generated read-noise
        realization over all sensor pixels.
        - ``real_dark_current``: total number of dark-current electrons
        generated during the most recent capture.

        The values stored in ``real_read_noise_per_pxl`` and
        ``real_dark_current`` describe a particular random realization and
        therefore may differ between captures, even when the nominal sensor
        parameters remain unchanged.

        Raises
        ------
        TypeError
            If a value in the configuration cannot be serialized to JSON.

        OSError
            If the output file cannot be created or written.
        """
        # ---------- Type checking ---------- 
        if not isinstance(filename, (str, os.PathLike)): 
            raise TypeError( f"filename must be a string or os.PathLike, " f"got {type(filename).__name__}." )

        # ---------- Value checking ---------- 
        if isinstance(filename, str) and not filename.strip(): 
            raise ValueError( "Filename cannot be empty." ) 

        # ---------- Configuration ----------

        config = {
            "width": self.width,
            "height": self.height,
            "pixel_size": self.pixel_size,
            "qe": self.qe,
            "read_noise": self.read_noise,
            "real_read_noise_mean": float(np.mean(self.real_read_noise)),
            "real_read_noise_std": float(np.std(self.real_read_noise)),
            "dark_current": self.dark_current,
            "real_dark_current": int(self.total_dark_electrons),
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
