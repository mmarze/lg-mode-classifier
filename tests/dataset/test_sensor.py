import json

import numpy as np
import pytest

from lgbeam.dataset.sensor import Sensor


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sensor():
    return Sensor(
        width=512,
        height=512,
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=0.1,
        full_well=9400,
        adc_bits=12,
        seed=42,
    )


@pytest.fixture
def sensor_params():
    return {
        "width": 512,
        "height": 512,
        "pixel_size": 2.74e-6,
        "qe": 0.68,
        "read_noise": 2.3,
        "dark_current": 0.1,
        "full_well": 9400,
        "adc_bits": 12,
        "seed": 42,
    }


# ============================================================
# Initialization
# ============================================================

def test_sensor_creation(sensor):
    assert sensor.width == 512
    assert sensor.height == 512
    assert sensor.pixel_size == 2.74e-6
    assert sensor.qe == 0.68
    assert sensor.read_noise == 2.3
    assert sensor.dark_current == 0.1
    assert sensor.full_well == 9400
    assert sensor.adc_bits == 12
    assert sensor.seed == 42


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("width", 0),
        ("height", 0),
        ("pixel_size", 0),
        ("qe", 0.0),
        ("qe", 1.1),
        ("read_noise", -1),
        ("dark_current", -1),
        ("full_well", 0),
        ("adc_bits", 0),
    ],
)
def test_invalid_sensor_parameter_value(
    sensor_params,
    parameter,
    value,
):
    sensor_params[parameter] = value

    with pytest.raises(ValueError):
        Sensor(**sensor_params)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("width", 512.5),
        ("height", "512"),
        ("pixel_size", "2.74e-6"),
        ("qe", "0.68"),
        ("read_noise", "2.3"),
        ("dark_current", "0.1"),
        ("full_well", "9400"),
        ("adc_bits", 12.5),
        ("seed", "42"),
    ],
)
def test_invalid_sensor_parameter_type(
    sensor_params,
    parameter,
    value,
):
    sensor_params[parameter] = value

    with pytest.raises(TypeError):
        Sensor(**sensor_params)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("pixel_size", np.inf),
        ("pixel_size", np.nan),
        ("qe", np.inf),
        ("qe", np.nan),
        ("read_noise", np.inf),
        ("read_noise", np.nan),
        ("dark_current", np.inf),
        ("dark_current", np.nan),
        ("full_well", np.inf),
        ("full_well", np.nan),
    ],
)
def test_non_finite_float_sensor_parameter(
    sensor_params,
    parameter,
    value,
):
    sensor_params[parameter] = value

    with pytest.raises(ValueError):
        Sensor(**sensor_params)


@pytest.mark.parametrize(
    "parameter",
    [
        "width",
        "height",
        "adc_bits",
        "seed",
    ],
)
def test_non_finite_integer_sensor_parameter(
    sensor_params,
    parameter,
):
    # np.inf is a float, so these parameters fail
    # the type validation before np.isfinite() is reached.
    sensor_params[parameter] = np.inf

    with pytest.raises(TypeError):
        Sensor(**sensor_params)


def test_seed_can_be_none(sensor_params):
    sensor_params["seed"] = None

    sensor = Sensor(**sensor_params)

    assert sensor.seed is None


# ============================================================
# Capture - output
# ============================================================

def test_capture_output_shape(sensor):
    photons = np.ones((512, 512)) * 100

    image = sensor.capture(
        photons,
        exposure=0.005,
    )

    assert image.shape == (512, 512)


def test_capture_output_dtype(sensor):
    photons = np.ones((512, 512)) * 100

    image = sensor.capture(
        photons,
        exposure=0.005,
    )

    assert image.dtype == np.uint16


def test_capture_output_range(sensor):
    photons = np.ones((512, 512)) * 10000

    image = sensor.capture(
        photons,
        exposure=0.005,
    )

    max_dn = 2**sensor.adc_bits - 1

    assert np.min(image) >= 0
    assert np.max(image) <= max_dn


# ============================================================
# Capture - input validation
# ============================================================

def test_capture_rejects_non_numpy_array(sensor):
    photons = [[1, 2], [3, 4]]

    with pytest.raises(TypeError):
        sensor.capture(
            photons,
            exposure=0.005,
        )


@pytest.mark.parametrize(
    "photons",
    [
        np.ones(512),
        np.ones((512, 512, 1)),
        np.ones((1, 512, 512)),
    ],
)
def test_capture_requires_2d_array(sensor, photons):
    with pytest.raises(ValueError):
        sensor.capture(
            photons,
            exposure=0.005,
        )


def test_capture_rejects_too_small_image(sensor):
    photons = np.ones((256, 256))

    with pytest.raises(ValueError):
        sensor.capture(
            photons,
            exposure=0.005,
        )


def test_capture_accepts_larger_input_image(sensor):
    photons = np.ones((600, 600)) * 100

    image = sensor.capture(
        photons,
        exposure=0.005,
    )

    assert image.shape == (
        sensor.height,
        sensor.width,
    )


def test_capture_rejects_non_numeric_array(sensor):
    photons = np.full(
        (512, 512),
        "100",
    )

    with pytest.raises(TypeError):
        sensor.capture(
            photons,
            exposure=0.005,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_capture_rejects_non_finite_photons(sensor, value):
    photons = np.ones((512, 512)) * 100
    photons[0, 0] = value

    with pytest.raises(ValueError):
        sensor.capture(
            photons,
            exposure=0.005,
        )


def test_capture_rejects_negative_photons(sensor):
    photons = np.ones((512, 512)) * 100
    photons[0, 0] = -1

    with pytest.raises(ValueError):
        sensor.capture(
            photons,
            exposure=0.005,
        )


@pytest.mark.parametrize(
    "exposure",
    [
        "0.005",
        None,
        [],
        {},
    ],
)
def test_capture_rejects_invalid_exposure_type(sensor, exposure):
    photons = np.ones((512, 512))

    with pytest.raises(TypeError):
        sensor.capture(
            photons,
            exposure,
        )


@pytest.mark.parametrize(
    "exposure",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_capture_rejects_non_finite_exposure(sensor, exposure):
    photons = np.ones((512, 512))

    with pytest.raises(ValueError):
        sensor.capture(
            photons,
            exposure,
        )


def test_capture_rejects_negative_exposure(sensor):
    photons = np.ones((512, 512))

    with pytest.raises(ValueError):
        sensor.capture(
            photons,
            exposure=-0.001,
        )


# ============================================================
# Capture - physical behavior
# ============================================================

def test_zero_noise_sensor_zero_photons():
    sensor = Sensor(
        width=512,
        height=512,
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=0,
        dark_current=0,
        full_well=9400,
        adc_bits=12,
        seed=42,
    )

    photons = np.zeros((512, 512))

    image = sensor.capture(
        photons,
        exposure=0,
    )

    assert np.all(image == 0)


def test_zero_photons_with_dark_current():
    sensor = Sensor(
        width=512,
        height=512,
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=0,
        dark_current=100,
        full_well=9400,
        adc_bits=12,
        seed=42,
    )

    photons = np.zeros((512, 512))

    image = sensor.capture(
        photons,
        exposure=1.0,
    )

    assert np.mean(image) > 0


def test_higher_photon_signal_gives_higher_mean_signal():
    photons_low = np.ones((512, 512)) * 10
    photons_high = np.ones((512, 512)) * 1000

    sensor_low = Sensor(
        width=512,
        height=512,
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=0.1,
        full_well=9400,
        adc_bits=12,
        seed=42,
    )

    sensor_high = Sensor(
        width=512,
        height=512,
        pixel_size=2.74e-6,
        qe=0.68,
        read_noise=2.3,
        dark_current=0.1,
        full_well=9400,
        adc_bits=12,
        seed=42,
    )

    low_image = sensor_low.capture(
        photons_low,
        exposure=0.005,
    )

    high_image = sensor_high.capture(
        photons_high,
        exposure=0.005,
    )

    assert np.mean(high_image) > np.mean(low_image)


def test_high_signal_saturates(sensor):
    photons = np.ones((512, 512)) * 1e8

    image = sensor.capture(
        photons,
        exposure=0.01,
    )

    max_dn = 2**sensor.adc_bits - 1

    assert np.max(image) == max_dn


# ============================================================
# Capture - internal state
# ============================================================

def test_capture_creates_dark_electrons(sensor):
    photons = np.ones((512, 512)) * 100

    sensor.capture(
        photons,
        exposure=0.005,
    )

    assert hasattr(sensor, "dark_electrons")
    assert sensor.dark_electrons.shape == (512, 512)
    assert np.all(sensor.dark_electrons >= 0)


def test_capture_creates_read_noise(sensor):
    photons = np.ones((512, 512)) * 100

    sensor.capture(
        photons,
        exposure=0.005,
    )

    assert hasattr(sensor, "real_read_noise")
    assert sensor.real_read_noise.shape == (512, 512)


def test_capture_stores_dark_electron_total(sensor):
    photons = np.ones((512, 512)) * 100

    sensor.capture(
        photons,
        exposure=0.005,
    )

    assert sensor.total_dark_electrons == np.sum(
        sensor.dark_electrons
    )


def test_capture_stores_read_noise_total(sensor):
    photons = np.ones((512, 512)) * 100

    sensor.capture(
        photons,
        exposure=0.005,
    )

    assert sensor.total_real_read_noise == pytest.approx(
        np.sum(sensor.real_read_noise)
    )


# ============================================================
# Cropping
# ============================================================

def test_capture_uses_central_region():
    sensor = Sensor(
        width=4,
        height=4,
        pixel_size=2.74e-6,
        qe=1.0,
        read_noise=0,
        dark_current=0,
        full_well=100000,
        adc_bits=16,
        seed=42,
    )

    photons = np.zeros((8, 8))

    # Only the central 4x4 region contains signal.
    photons[2:6, 2:6] = 100

    image = sensor.capture(
        photons,
        exposure=0,
    )

    assert np.all(image > 0)


# ============================================================
# Reproducibility
# ============================================================

def test_seed_reproducibility():
    photons = np.ones((20, 20)) * 100

    sensor1 = Sensor(
        20,
        20,
        2.74e-6,
        0.68,
        2.3,
        0.1,
        9400,
        12,
        seed=123,
    )

    sensor2 = Sensor(
        20,
        20,
        2.74e-6,
        0.68,
        2.3,
        0.1,
        9400,
        12,
        seed=123,
    )

    image1 = sensor1.capture(
        photons,
        0.005,
    )

    image2 = sensor2.capture(
        photons,
        0.005,
    )

    assert np.array_equal(
        image1,
        image2,
    )


# ============================================================
# Save configuration
# ============================================================

def test_save_config_creates_valid_json(sensor, tmp_path):
    filename = tmp_path / "sensor.json"

    sensor.capture(
        np.ones((512, 512)) * 100,
        exposure=0.005,
    )

    sensor.save_config(filename)

    assert filename.exists()

    with open(filename) as file:
        data = json.load(file)

    assert isinstance(data, dict)


def test_save_config_contains_all_sensor_parameters(
    sensor,
    tmp_path,
):
    filename = tmp_path / "sensor.json"

    sensor.capture(
        np.ones((512, 512)) * 100,
        exposure=0.005,
    )

    sensor.save_config(filename)

    with open(filename) as file:
        data = json.load(file)

    expected_keys = {
        "width",
        "height",
        "pixel_size",
        "qe",
        "read_noise",
        "real_read_noise_mean",
        "real_read_noise_std",
        "dark_current",
        "real_dark_current",
        "full_well",
        "adc_bits",
        "seed",
    }

    assert set(data.keys()) == expected_keys


def test_save_config_contains_read_noise_statistics(
    sensor,
    tmp_path,
):
    filename = tmp_path / "sensor.json"

    sensor.capture(
        np.ones((512, 512)) * 100,
        exposure=0.005,
    )

    sensor.save_config(filename)

    with open(filename) as file:
        data = json.load(file)

    assert data["real_read_noise_mean"] == pytest.approx(
        np.mean(sensor.real_read_noise)
    )

    assert data["real_read_noise_std"] == pytest.approx(
        np.std(sensor.real_read_noise)
    )


def test_save_config_contains_dark_current_statistics(
    sensor,
    tmp_path,
):
    filename = tmp_path / "sensor.json"

    sensor.capture(
        np.ones((512, 512)) * 100,
        exposure=0.005,
    )

    sensor.save_config(filename)

    with open(filename) as file:
        data = json.load(file)

    assert data["real_dark_current"] == int(
        sensor.total_dark_electrons
    )


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
    ],
)
def test_save_config_rejects_empty_filename(
    sensor,
    filename,
):
    with pytest.raises(ValueError):
        sensor.save_config(filename)


@pytest.mark.parametrize(
    "filename",
    [
        None,
        123,
        3.14,
        [],
        {},
    ],
)
def test_save_config_rejects_invalid_filename_type(
    sensor,
    filename,
):
    with pytest.raises(TypeError):
        sensor.save_config(filename)


def test_save_config_accepts_pathlike(sensor, tmp_path):
    filename = tmp_path / "sensor.json"

    sensor.capture(
        np.ones((512, 512)),
        exposure=0.005,
    )

    sensor.save_config(filename)

    assert filename.exists()
    