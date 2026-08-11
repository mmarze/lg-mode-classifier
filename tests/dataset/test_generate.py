import json
from pathlib import Path

import numpy as np
import pytest

from lgbeam.dataset.generate import generate_dataset


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "output" / "images"
METADATA_DIR = PROJECT_ROOT / "output" / "metadata"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path):
    return np.loadtxt(path, delimiter=",")


# ---------------------------------------------------------------------------
# Basic dataset generation
# ---------------------------------------------------------------------------

def test_generate_dataset_creates_expected_number_of_images(tmp_path, monkeypatch):
    """
    One image should be generated for each requested LG mode.
    """

    monkeypatch.chdir(PROJECT_ROOT)

    # generate_dataset() expects the directories to exist.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    modes = [(0, 0), (1, 0), (2, 0)]

    # Remove files from previous test runs
    for file in IMAGE_DIR.glob("*.csv"):
        file.unlink()

    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=modes,
        n_images_per_class=1,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    image_files = sorted(IMAGE_DIR.glob("*.csv"))
    metadata_files = sorted(METADATA_DIR.glob("*.json"))

    assert len(image_files) == len(modes)
    assert len(metadata_files) == len(modes)


def test_generate_dataset_creates_expected_number_of_images_per_class(
    tmp_path,
    monkeypatch,
):
    """
    The number of generated images should equal:

        number_of_modes * images_per_class
    """

    # Remove files from previous test runs
    for file in IMAGE_DIR.glob("*.csv"):
        file.unlink()

    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    monkeypatch.chdir(PROJECT_ROOT)

    # generate_dataset() expects the directories to exist.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    modes = [(0, 0), (1, 0)]
    images_per_class = 3

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=modes,
        n_images_per_class=images_per_class,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    image_files = sorted(IMAGE_DIR.glob("*.csv"))
    metadata_files = sorted(METADATA_DIR.glob("*.json"))

    expected = len(modes) * images_per_class

    assert len(image_files) == expected
    assert len(metadata_files) == expected


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

def test_generate_dataset_uses_expected_filenames(tmp_path, monkeypatch):
    """
    Generated files should follow the documented naming convention.
    """

    # Remove files from previous test runs
    for file in IMAGE_DIR.glob("*.csv"):
        file.unlink()

    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    monkeypatch.chdir(PROJECT_ROOT)

    # generate_dataset() expects the directories to exist.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=[(1, 2)],
        n_images_per_class=2,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    assert (IMAGE_DIR / "mode_12_image_0000.csv").exists()
    assert (IMAGE_DIR / "mode_12_image_0001.csv").exists()

    assert (METADATA_DIR / "mode_12_image_0000.json").exists()
    assert (METADATA_DIR / "mode_12_image_0001.json").exists()


# ---------------------------------------------------------------------------
# Image properties
# ---------------------------------------------------------------------------

def test_generated_images_have_sensor_dimensions(tmp_path, monkeypatch):
    """
    Every generated image should have dimensions equal to sensor_size.
    """

    # Remove files from previous test runs
    for file in IMAGE_DIR.glob("*.csv"):
        file.unlink()

    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    monkeypatch.chdir(PROJECT_ROOT)

    # generate_dataset() expects the directories to exist.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    sensor_size = (32, 24)

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=[(0, 0)],
        n_images_per_class=2,
        sensor_size=sensor_size,
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    image_files = sorted(IMAGE_DIR.glob("*.csv"))

    assert len(image_files) == 2

    for image_file in image_files:
        image = _read_csv(image_file)

        # Sensor.capture() returns (height, width)
        assert image.shape == (sensor_size[1], sensor_size[0])


def test_generated_images_contain_finite_values(tmp_path, monkeypatch):
    """
    Generated images must not contain NaN or infinite values.
    """

    monkeypatch.chdir(PROJECT_ROOT)

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=[(0, 0)],
        n_images_per_class=2,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    for image_file in (tmp_path / "output" / "images").glob("*.csv"):
        image = _read_csv(image_file)

        assert np.all(np.isfinite(image))


def test_generated_images_are_within_adc_range(tmp_path, monkeypatch):
    """
    Pixel values must be within the ADC range.
    """

    adc_bits = 12
    max_dn = 2**adc_bits - 1

    monkeypatch.chdir(PROJECT_ROOT)

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=[(0, 0)],
        n_images_per_class=2,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=adc_bits,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    for image_file in (tmp_path / "output" / "images").glob("*.csv"):
        image = _read_csv(image_file)

        assert np.all(image >= 0)
        assert np.all(image <= max_dn)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_contains_sensor_configuration(tmp_path, monkeypatch):
    """
    Metadata should contain the parameters used to construct the sensor.
    """

    # Remove files from previous test runs
    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    monkeypatch.chdir(PROJECT_ROOT)

    # generate_dataset() expects the directories to exist.
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    generate_dataset(
        l=0.01,
        n=64,
        lg_modes=[(0, 0)],
        n_images_per_class=1,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    metadata_file = (
        METADATA_DIR
        / "mode_00_image_0000.json"
    )

    assert metadata_file.exists()

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    assert metadata["width"] == 32
    assert metadata["height"] == 32
    assert metadata["pixel_size"] == pytest.approx(2.74e-6)
    assert metadata["qe"] == pytest.approx(0.68)
    assert metadata["read_noise"] == pytest.approx(2.3)
    assert metadata["dark_current"] == pytest.approx(22)
    assert metadata["full_well"] == pytest.approx(9400)
    assert metadata["adc_bits"] == 12
    assert metadata["seed"] == 123


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_invalid_lg_modes_type():
    with pytest.raises(TypeError, match="lg_modes must be a list of tuples of two integer"):
        generate_dataset(lg_modes="invalid")


@pytest.mark.parametrize(
    "lg_modes",
    [
        [(0,)],
        [(0, 0, 1)],
        [0],
        ["00"],
    ],
)
def test_invalid_lg_mode_format(lg_modes):
    with pytest.raises(ValueError, match="Each element of lg_modes"):
        generate_dataset(lg_modes=lg_modes)


def test_invalid_number_of_images_type():
    with pytest.raises(TypeError, match="Number of images per class"):
        generate_dataset(n_images_per_class=1.5)


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_number_of_images_value(value):
    with pytest.raises(ValueError, match="n_images_per_class must be greater"):
        generate_dataset(n_images_per_class=value)


@pytest.mark.parametrize(
    "w0_range",
    [
        (1e-3,),
        (1e-3, 2e-3, 3e-3),
        [1e-3],
    ],
)
def test_invalid_w0_range_length(w0_range):
    with pytest.raises(ValueError, match="w0_range must contain exactly two"):
        generate_dataset(w0_range=w0_range)


def test_invalid_w0_range_type():
    with pytest.raises(TypeError, match="w0_range values must be numeric"):
        generate_dataset(w0_range=("small", "large"))


@pytest.mark.parametrize(
    "w0_range",
    [
        (0, 1e-3),
        (-1e-3, 1e-3),
        (1e-3, 0),
        (2e-3, 1e-3),
    ],
)
def test_invalid_w0_range_values(w0_range):
    with pytest.raises(ValueError):
        generate_dataset(w0_range=w0_range)


def test_invalid_power_range_type():
    with pytest.raises(TypeError, match="Power range must be"):
        generate_dataset(power_range=1)


@pytest.mark.parametrize(
    "power_range",
    [
        (1,),
        (1, 2, 3),
    ],
)
def test_invalid_power_range_length(power_range):
    with pytest.raises(ValueError, match="power_range must contain exactly two"):
        generate_dataset(power_range=power_range)


@pytest.mark.parametrize(
    "power_range",
    [
        (1.5, 10),
        (1, 10.5),
    ],
)
def test_invalid_power_range_values(power_range):
    with pytest.raises(TypeError, match="power_range values must be integers"):
        generate_dataset(power_range=power_range)


@pytest.mark.parametrize(
    "power_range",
    [
        (0, 10),
        (-1, 10),
        (1, 0),
        (10, 1),
    ],
)
def test_invalid_power_range_limits(power_range):
    with pytest.raises(ValueError):
        generate_dataset(power_range=power_range)


@pytest.mark.parametrize("shift", [-1, -10])
def test_negative_center_shift_is_rejected(shift):
    with pytest.raises(ValueError, match="max_center_shift must be"):
        generate_dataset(max_center_shift=shift)


def test_invalid_center_shift_type():
    with pytest.raises(TypeError, match="max_center_shift"):
        generate_dataset(max_center_shift=1.5)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _read_generated_files():
    files = {}

    for directory in (IMAGE_DIR, METADATA_DIR):
        for path in directory.rglob("*"):
            if path.is_file():
                files[path.relative_to(PROJECT_ROOT)] = path.read_bytes()

    return files

def test_generate_dataset_is_reproducible(monkeypatch):
    """
    The same seed should generate identical datasets.
    """

    monkeypatch.chdir(PROJECT_ROOT)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        l=0.01,
        n=64,
        lg_modes=[(0, 0)],
        n_images_per_class=2,
        sensor_size=(32, 32),
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=22,
        full_well=9400,
        adc_bits=12,
        seed=123,
        w0_range=(1e-3, 2e-3),
        wavelength=633e-9,
        power_range=(10, 20),
        exposure=1e-3,
        max_center_shift=0,
    )

    # Remove files from previous test runs
    for file in IMAGE_DIR.glob("*.csv"):
        file.unlink()

    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    # First run
    generate_dataset(**kwargs)

    first_run = _read_generated_files()

    # Remove generated files
    for file in IMAGE_DIR.glob("*.csv"):
        file.unlink()

    for file in METADATA_DIR.glob("*.json"):
        file.unlink()

    # Second run
    generate_dataset(**kwargs)

    second_run = _read_generated_files()

    assert first_run == second_run
    