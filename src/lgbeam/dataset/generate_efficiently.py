import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from lgbeam.mesh import create_mesh
from lgbeam.beams import LaguerreGauss
from lgbeam.dataset.sensor import Sensor
from lgbeam.dataset.config import (
    LG_MODES,
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
    N_IMAGES_PER_CLASS,
    DOWNSAMPLING
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output" / "modes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final stored image size: (width, height).
OUTPUT_SIZE = (SENSOR_SIZE[0]//DOWNSAMPLING, SENSOR_SIZE[1]//DOWNSAMPLING)


# ---------------------------------------------------------------------------
# Worker-local state
# ---------------------------------------------------------------------------

_WORKER_R = None
_WORKER_PHI = None
_WORKER_SENSOR_KWARGS = None
_WORKER_W0_RANGE = None
_WORKER_WAVELENGTH = None
_WORKER_POWER_RANGE = None
_WORKER_EXPOSURE = None
_WORKER_MAX_CENTER_SHIFT = None


def _init_worker(
    r,
    phi,
    sensor_kwargs,
    w0_range,
    wavelength,
    power_range,
    exposure,
    max_center_shift,
):
    """Initialize data used by all jobs executed by one worker."""
    global _WORKER_R
    global _WORKER_PHI
    global _WORKER_SENSOR_KWARGS
    global _WORKER_W0_RANGE
    global _WORKER_WAVELENGTH
    global _WORKER_POWER_RANGE
    global _WORKER_EXPOSURE
    global _WORKER_MAX_CENTER_SHIFT

    _WORKER_R = r
    _WORKER_PHI = phi
    _WORKER_SENSOR_KWARGS = sensor_kwargs
    _WORKER_W0_RANGE = w0_range
    _WORKER_WAVELENGTH = wavelength
    _WORKER_POWER_RANGE = power_range
    _WORKER_EXPOSURE = exposure
    _WORKER_MAX_CENTER_SHIFT = max_center_shift


def _resize_uint16(image, output_size):
    """Downsample a uint16 image while keeping uint16 output."""
    image = np.asarray(image, dtype=np.uint16)
    pil_image = Image.fromarray(image, mode="I;16")
    # Image.Resampling.BOX -> an average from all pixels
    resized = pil_image.resize(output_size, resample=Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.uint16)


def _generate_one_image(task):
    """
    Generate one image and downsample it to OUTPUT_SIZE.

    Returns only the image. No metadata is stored in the dataset.
    """
    mode_l, mode_p, seed = task
    rng = np.random.default_rng(seed)

    sensor = Sensor(
        **_WORKER_SENSOR_KWARGS,
        seed=seed,
    )

    w0 = (
        rng.integers(
            _WORKER_W0_RANGE[0] * 1e6,
            _WORKER_W0_RANGE[1] * 1e6 + 1,
        )
        * 1e-6
    )

    photons = LaguerreGauss(
        mode_p,
        mode_l,
        r=_WORKER_R,
        phi=_WORKER_PHI,
        z=0,
        w0=w0,
        wavelength=_WORKER_WAVELENGTH,
    )

    # Keep the optical calculation in float32.
    photons = np.abs(photons).astype(np.float32, copy=False)

    max_photons = np.max(photons)
    if max_photons <= 0 or not np.isfinite(max_photons):
        raise ValueError(
            f"Invalid maximum photon value for mode ({mode_l}, {mode_p}): "
            f"{max_photons}"
        )

    photons /= np.float32(max_photons)

    power = rng.integers(
        low=_WORKER_POWER_RANGE[0],
        high=_WORKER_POWER_RANGE[1] + 1,
    )
    photons *= np.float32(power)

    x_shift = (
        rng.integers(
            0,
            2 * _WORKER_MAX_CENTER_SHIFT + 1,
        )
        - _WORKER_MAX_CENTER_SHIFT
    )

    y_shift = (
        rng.integers(
            0,
            2 * _WORKER_MAX_CENTER_SHIFT + 1,
        )
        - _WORKER_MAX_CENTER_SHIFT
    )

    mask_size = max(
        _WORKER_SENSOR_KWARGS["width"],
        _WORKER_SENSOR_KWARGS["height"],
    )

    photons = photons[
        _WORKER_MAX_CENTER_SHIFT - x_shift:
        _WORKER_MAX_CENTER_SHIFT + mask_size - x_shift,
        _WORKER_MAX_CENTER_SHIFT - y_shift:
        _WORKER_MAX_CENTER_SHIFT + mask_size - y_shift,
    ]

    # Sensor.capture() returns uint16.
    image = sensor.capture(photons, _WORKER_EXPOSURE)
    image = np.asarray(image, dtype=np.uint16)

    # Downsample only after the sensor simulation.
    # Stored shape: (height, width) = OUTPUT_SIZE.
    return _resize_uint16(image, OUTPUT_SIZE)


def _validate_inputs(
    lg_modes,
    n_images_per_class,
    w0_range,
    power_range,
    max_center_shift,
    adc_bits,
    sensor_size,
):
    """Validate parameters used by dataset generation."""
    if not isinstance(lg_modes, (list, tuple)):
        raise TypeError(
            f"lg_modes must be a list or tuple of (l, p) pairs, "
            f"got {type(lg_modes).__name__}."
        )

    for mode in lg_modes:
        if not isinstance(mode, (list, tuple)) or len(mode) != 2:
            raise ValueError(
                "Each element of lg_modes must be a tuple/list of "
                "(l, p), e.g. [(0, 0), (1, 0), (2, 0)]."
            )
        if not all(isinstance(x, (int, np.integer)) for x in mode):
            raise TypeError(f"LG mode indices must be integers, got {mode}.")

    if not isinstance(n_images_per_class, (int, np.integer)):
        raise TypeError("n_images_per_class must be an integer.")
    if n_images_per_class <= 0:
        raise ValueError("n_images_per_class must be greater than 0.")

    if not isinstance(w0_range, (list, tuple)) or len(w0_range) != 2:
        raise ValueError("w0_range must contain exactly two values: (min, max).")
    if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in w0_range):
        raise TypeError("w0_range values must be numeric.")
    if w0_range[0] <= 0 or w0_range[1] <= 0:
        raise ValueError("w0_range values must be greater than 0.")
    if w0_range[0] > w0_range[1]:
        raise ValueError("w0_range minimum must be <= maximum.")

    if not isinstance(power_range, (list, tuple)) or len(power_range) != 2:
        raise ValueError("power_range must contain exactly two values: (min, max).")
    if not all(isinstance(x, (int, np.integer)) for x in power_range):
        raise TypeError("power_range values must be integers.")
    if power_range[0] <= 0 or power_range[1] <= 0:
        raise ValueError("power_range values must be greater than 0.")
    if power_range[0] > power_range[1]:
        raise ValueError("power_range minimum must be <= maximum.")

    if not isinstance(max_center_shift, (int, np.integer)):
        raise TypeError("max_center_shift must be an integer.")
    if max_center_shift < 0:
        raise ValueError("max_center_shift must be >= 0.")

    if not isinstance(adc_bits, (int, np.integer)):
        raise TypeError("adc_bits must be an integer.")
    if adc_bits <= 0:
        raise ValueError("adc_bits must be greater than 0.")
    if adc_bits > 16:
        raise ValueError("uint16 storage supports at most 16 ADC bits.")

    if len(sensor_size) != 2:
        raise ValueError("sensor_size must contain exactly (width, height).")
    if sensor_size[0] <= 0 or sensor_size[1] <= 0:
        raise ValueError("Sensor dimensions must be positive.")


def _mode_filename(mode_l, mode_p):
    return OUTPUT_DIR / f"mode_l{mode_l}_p{mode_p}.h5"


def _generate_mode_file(
    mode,
    n_images_per_class,
    mode_seed,
    r,
    phi,
    sensor_kwargs,
    w0_range,
    wavelength,
    power_range,
    exposure,
    max_center_shift,
    n_workers,
):
    """Generate one HDF5 file containing only images for one LG mode."""
    mode_l, mode_p = mode
    output_path = _mode_filename(mode_l, mode_p)
    output_shape = (OUTPUT_SIZE[1], OUTPUT_SIZE[0]) 

    mode_rng = np.random.default_rng(mode_seed)
    tasks = (
        (
            mode_l,
            mode_p,
            int(mode_rng.integers(0, 2**32 - 1)),
        )
        for _ in range(n_images_per_class)
    )

    with h5py.File(output_path, "w") as h5:
        # The HDF5 file contains exactly one object: the image dataset.
        images = h5.create_dataset(
            "images",
            shape=(n_images_per_class, *output_shape),
            dtype=np.uint16,
            chunks=(1, *output_shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        with ProcessPoolExecutor(
            max_workers=max(1, int(n_workers)),
            initializer=_init_worker,
            initargs=(
                r,
                phi,
                sensor_kwargs,
                w0_range,
                wavelength,
                power_range,
                exposure,
                max_center_shift,
            ),
        ) as executor:
            for index, image in enumerate(
                executor.map(_generate_one_image, tasks, chunksize=1)
            ):
                if image.shape != output_shape:
                    raise ValueError(
                        f"Unexpected image shape {image.shape}; "
                        f"expected {output_shape}."
                    )

                images[index] = image

                if (index + 1) % 100 == 0 or index + 1 == n_images_per_class:
                    print(
                        f"Mode ({mode_l}, {mode_p}): "
                        f"{index + 1}/{n_images_per_class} "
                        f"({100 * (index + 1) / n_images_per_class:.1f}%)"
                    )

    return output_path


def generate_dataset(
    l=L,
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
    max_center_shift=CENTER_SHIFT,
    n_workers=2,
):
    """
    Generate one compressed HDF5 file per LG mode.

    Each file contains only:
        images : (N, SENSOR_SIZE[0]//DOWNSAMPLING, SENSOR_SIZE[1]//DOWNSAMPLING), uint16

    No labels, metadata, attributes, w0, power, shifts or seeds are stored.
    The LG mode is identified solely by the filename.
    """

    _validate_inputs(
        lg_modes=lg_modes,
        n_images_per_class=n_images_per_class,
        w0_range=w0_range,
        power_range=power_range,
        max_center_shift=max_center_shift,
        adc_bits=adc_bits,
        sensor_size=sensor_size,
    )

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    else:
        seed = int(seed)

    master_rng = np.random.default_rng(seed)

    print(f"Master seed: {seed}")
    print(f"Workers per mode: {n_workers}")
    print(f"Images per mode: {n_images_per_class}")
    print(f"Sensor size: {sensor_size[0]} x {sensor_size[1]}")
    print(f"Stored image size: {OUTPUT_SIZE[0]} x {OUTPUT_SIZE[1]}")
    print(f"Output directory: {OUTPUT_DIR}")

    # The full sensor simulation still uses the original mesh/resolution.
    r, phi = create_mesh(l, n)

    sensor_kwargs = {
        "width": int(sensor_size[0]),
        "height": int(sensor_size[1]),
        "pixel_size": pixel_size,
        "qe": qe,
        "read_noise": read_noise,
        "dark_current": dark_current,
        "full_well": full_well,
        "adc_bits": int(adc_bits),
    }

    generated_files = []

    for mode in lg_modes:
        mode_l, mode_p = mode

        mode_seed = int(master_rng.integers(0, 2**32 - 1))

        print()
        print("=" * 70)
        print(f"Generating mode ({mode_l}, {mode_p})")
        print("=" * 70)

        output_path = _generate_mode_file(
            mode=mode,
            n_images_per_class=n_images_per_class,
            mode_seed=mode_seed,
            r=r,
            phi=phi,
            sensor_kwargs=sensor_kwargs,
            w0_range=w0_range,
            wavelength=wavelength,
            power_range=power_range,
            exposure=exposure,
            max_center_shift=max_center_shift,
            n_workers=n_workers,
        )

        generated_files.append(output_path)
        print(f"Saved: {output_path}")

    print()
    print("=" * 70)
    print("Dataset generation completed.")
    print("=" * 70)

    for path in generated_files:
        print(path)


if __name__ == "__main__":
    generate_dataset(n_workers=3)
