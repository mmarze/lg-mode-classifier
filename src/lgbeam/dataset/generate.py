import numpy as np
import matplotlib.pyplot as plt

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

    # Set seed
    np.random.seed(SEED)

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

            w0 = np.random.randint(w0_range[0] * 1e6, w0_range[1] * 1e6 + 1) * 1e-6
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
            mul = np.random.randint(low=power_range[0], high=power_range[1] +1)
            photons *= mul
            # XY shift
            x_decenter = np.random.randint(0,2*max_center_shift+1) - max_center_shift
            y_decenter = np.random.randint(0,2*max_center_shift+1) - max_center_shift
            mask_size = max(SENSOR_SIZE)

            photons = photons[CENTER_SHIFT - x_decenter:CENTER_SHIFT + mask_size - x_decenter,
                              CENTER_SHIFT - y_decenter:CENTER_SHIFT + mask_size - y_decenter]

            # Image capture
            image = sony_imx541.capture(photons, exposure)

            print(np.max(image))

            # # Save image to csv file
            # np.savetxt(f"output/images/mode_{elem[0]}{elem[1]}_image_{i:04d}.csv", 
            #            image, 
            #            delimiter=','
            #            )
            # # Save metadata to txt file
            # sony_imx541.save_config(f"output/metadata/mode_{elem[0]}{elem[1]}_image_{i:04d}.txt")


if __name__ == '__main__':
    generate_dataset()