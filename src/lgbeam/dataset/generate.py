import numpy as np
from pathlib import Path

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.dataset.sensor import Sensor
from lgbeam.dataset.config import (LG_MODES, 
                                   WAVELENGTH, 
                                   SENSOR_SIZE, 
                                   PIXEL_SIZE, 
                                   CENTER_SHIFT, 
                                   EXPOSURE, 
                                   QE,
                                   READ_NOISE,
                                   DARK_CURRENT,
                                   FULL_WELL,
                                   ADC_BITS,
                                   POWER_RANGE, 
                                   W0_RANGE, 
                                   SEED,
                                   L, 
                                   N,
                                   N_IMAGES_PER_CLASS)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_DIR = PROJECT_ROOT / "output" / "images"
METADATA_DIR = PROJECT_ROOT / "output" / "metadata"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_dataset(l=L, 
                     n=N, 
                     lg_modes=LG_MODES, 
                     n_images_per_class=N_IMAGES_PER_CLASS,
                     sensor_size=SENSOR_SIZE,
                     pixel_size=PIXEL_SIZE,
                     qe=QE,
                     read_noise=READ_NOISE,
                     dark_current=DARK_CURRENT,
                     full_well=FULL_WELL,
                     adc_bits=ADC_BITS,
                     seed=SEED,
                     w0_range=W0_RANGE,
                     wavelength=WAVELENGTH,
                     power_range=POWER_RANGE,
                     exposure=EXPOSURE,
                     max_center_shift=CENTER_SHIFT
                     ):
    """
    Generate a synthetic dataset of Laguerre-Gaussian (LG) beam images 
    captured with a simulated Sony IMX541 image sensor. For each LG mode 
    specified in ``lg_modes``, the function generates ``n_images_per_class`` 
    images. Each image is randomized by varying the beam waist, optical power, 
    and beam position. The resulting optical intensity distribution is passed 
    through a simulated camera model that accounts for sensor characteristics 
    such as quantum efficiency, read noise, dark current, full-well capacity, 
    and ADC quantization. The generated images are saved as CSV files, while 
    the corresponding simulated sensor configuration is saved as a metadata 
    TXT file. Default parameters are stored in config.pt file.
    
    Parameters
    ----------
    l : float, optional 
        Physical size of the computational mesh in metres. The mesh spans the 
        transverse beam plane and is used for calculating the Laguerre-Gaussian 
        beam field. 
        
    n : int, optional 
        Number of samples along each dimension of the computational mesh. 
        
    lg_modes : iterable of tuple[int, int], optional 
        Collection of Laguerre-Gaussian modes to generate. Each mode is specified 
        as a tuple ``(l, p)``, where ``l`` is the azimuthal mode index and ``p`` 
        is the radial mode index. 
        
    n_images_per_class : int, optional 
        Number of images generated for each LG mode. sensor_size : tuple[int, int], 
        optional Sensor dimensions in pixels, specified as ``(width, height)``. 
        
    pixel_size : float, optional 
        Physical pixel size of the simulated sensor in metres. 
        
    qe : float, optional 
        Quantum efficiency of the simulated sensor. This parameter determines 
        the fraction of incident photons converted into photoelectrons. 
        
    read_noise : float, optional 
        Standard deviation of the sensor read noise, typically expressed in electrons. 
        
    dark_current : float, optional 
        Dark current of the simulated sensor. 
        
    full_well : int or float, optional 
        Full-well capacity of the sensor in electrons. Pixel values cannot 
        exceed this capacity. 
        
    adc_bits : int, optional 
        Bit depth of the simulated analog-to-digital converter (ADC). 
    
    seed : int, optional 
        Random seed used by the simulated sensor. 
        
    w0_range : tuple[int | float, int | float], optional 
        Minimum and maximum beam waist values in metres. A random beam waist 
        is selected from this range for every generated image. 
        
    wavelength : float, optional 
        Wavelength of the Laguerre-Gaussian beam in metres. 
        
    power_range : tuple[int, int], optional 
        Minimum and maximum scaling factors applied to the normalized beam intensity. 
        A random value from this range is selected for each generated image. 
        
    exposure : float, optional 
        Sensor exposure time used when capturing the simulated image. 
        
    max_center_shift : int, optional 
        Maximum displacement of the beam centre from the nominal sensor centre, 
        measured in pixels. Independent random shifts are applied along the x 
        and y axes. 
        
    Returns
    -------
    None 
        The function does not return the generated images. Instead, it writes 
        the dataset and corresponding metadata to disk. 
        
    Notes
    -----
    The generated beam field is converted to an intensity-like quantity using 
    its absolute value and subsequently normalized to its maximum value. A 
    random power scaling factor is then applied. 
    
    The beam is randomly shifted in the x and y directions before being cropped to the 
    sensor dimensions. The resulting photon distribution is passed to the simulated 
    ``Sensor.capture()`` method. 
    
    The function currently saves files to the following directories: 

    ``output/images/`` 

    Contains one CSV file per generated image. 
    
    ``output/metadata/`` 
    Contains one TXT file per generated image describing the simulated sensor configuration. 

    File names follow the convention:: 
        mode_{l}{p}_image_{index:04d}.csv 
        mode_{l}{p}_image_{index:04d}.txt 
        
        where ``l`` and ``p`` identify the LG mode and ``index`` is the image number within the class. 
        
    Examples
    --------
    Generate a dataset using the default configuration:: 
    
        generate_dataset()
        
    Generate 100 images for each of the specified LG modes::
        
        generate_dataset(
                lg_modes=[(0, 0), (1, 0), (2, 0)], 
                n_images_per_class=100, 
        )
    """
    # ---------- Type checking ----------
    if not isinstance(lg_modes, (list, tuple)):
            raise TypeError(
                f"lg_modes must be a list of tuples of two integers, got {type(lg_modes).__name__}."
            )

    for mode in lg_modes: 
         if not isinstance(mode, (list, tuple)) or len(mode) != 2: 
              raise ValueError(
                    "Each element of lg_modes must be a tuple/list of "
                    "(l, p), e.g. [(0, 0), (1, 0), (2, 0)]"
                )

    if not isinstance(n_images_per_class, (int, np.integer)):
        raise TypeError(
            f"Number of images per class must be an integer, got {type(n_images_per_class).__name__}."
        )

    if not isinstance(w0_range, (list, tuple)) or len(w0_range) != 2:
        raise ValueError("w0_range must contain exactly two values: (min, max)")

    if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in w0_range):
        raise TypeError("w0_range values must be numeric")

    if not isinstance(power_range, (list, tuple)):
        raise TypeError(
            f"Power range must be a list or a tuple of two integers, got {type(power_range).__name__}."
        )

    if not isinstance(power_range, (list, tuple)) or len(power_range) != 2:
        raise ValueError("power_range must contain exactly two values: (min, max)")

    if not all(isinstance(x, (int, np.integer)) for x in power_range):
        raise TypeError("power_range values must be integers")

    if not isinstance(max_center_shift, (int, np.integer)):
            raise TypeError(
                f"max_center_shift must be an integer, got {type(max_center_shift).__name__}."
            )

    # ---------- Value checking ----------
    if n_images_per_class <= 0: 
        raise ValueError("n_images_per_class must be greater than 0")

    w0_min, w0_max = w0_range 

    if w0_min <= 0 or w0_max <= 0: 
        raise ValueError("w0_range values must be greater than 0") 

    if w0_min > w0_max: 
         raise ValueError("w0_range minimum must be <= maximum")

    power_min, power_max = power_range

    if power_min <= 0 or power_max <= 0:
        raise ValueError("power_range values must be greater than 0")

    if power_min > power_max:
        raise ValueError("power_range minimum must be <= maximum")

    if max_center_shift < 0:
        raise ValueError("max_center_shift must be >= 0")

    # ---------- Dataset generation ----------

    rng = np.random.default_rng(seed)

    # Generate mesh for LG beam calculations
    r, phi = create_mesh(l, n)

    for elem in lg_modes:
        for i in range(n_images_per_class):

            sony_imx541 = Sensor(
                                width=sensor_size[0], 
                                height=sensor_size[1], 
                                pixel_size=pixel_size, 
                                qe=qe,
                                read_noise=read_noise, 
                                dark_current=dark_current, 
                                full_well=full_well, 
                                adc_bits=adc_bits,
                                seed=seed
                )

            w0 = rng.integers(w0_range[0] * 1e6, w0_range[1] * 1e6 + 1) * 1e-6
            photons = LaguerreGauss(elem[1], 
                                    elem[0], 
                                    r=r, 
                                    phi=phi, 
                                    z=0, 
                                    w0=w0, 
                                    wavelength=wavelength)

            photons = np.abs(photons)
            # Brightness control
            photons /= np.max(photons)
            mul = rng.integers(low=power_range[0], high=power_range[1] +1)
            photons *= mul
            # XY shift
            x_decenter = rng.integers(0,2*max_center_shift+1) - max_center_shift
            y_decenter = rng.integers(0,2*max_center_shift+1) - max_center_shift
            mask_size = max(sensor_size)

            photons = photons[max_center_shift - x_decenter:max_center_shift + mask_size - x_decenter,
                              max_center_shift - y_decenter:max_center_shift + mask_size - y_decenter]

            # Image capture
            image = sony_imx541.capture(photons, exposure)
            x1, x2 = elem

            # Save image to csv file
            np.savetxt(IMAGE_DIR / f"mode_{x1}{x2}_image_{i:04d}.csv", 
                       image, 
                       delimiter=','
                    )
            # Save metadata to txt file
            sony_imx541.save_config(METADATA_DIR / f"mode_{x1}{x2}_image_{i:04d}.json")


if __name__ == '__main__':
    generate_dataset()
