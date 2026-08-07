import numpy as np
import json
import pytest

from lgbeam.dataset.sensor import Sensor


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
        seed=42
    )


# ----------------------------
# Initialization tests
# ----------------------------

def test_sensor_creation(sensor):

    assert sensor.width == 512
    assert sensor.height == 512
    assert sensor.qe == 0.68
    assert sensor.adc_bits == 12



def test_invalid_width():

    with pytest.raises(ValueError):

        Sensor(
            width=0,
            height=512,
            pixel_size=2.74e-6,
            qe=0.68,
            read_noise=2.3,
            dark_current=0.1,
            full_well=9400,
            adc_bits=12,
            seed=42
        )



def test_invalid_qe():

    with pytest.raises(ValueError):

        Sensor(
            width=512,
            height=512,
            pixel_size=2.74e-6,
            qe=1.5,
            read_noise=2.3,
            dark_current=0.1,
            full_well=9400,
            adc_bits=12,
            seed=42
        )



def test_negative_read_noise():

    with pytest.raises(ValueError):

        Sensor(
            width=512,
            height=512,
            pixel_size=2.74e-6,
            qe=0.68,
            read_noise=-1,
            dark_current=0.1,
            full_well=9400,
            adc_bits=12,
            seed=42
        )



# ----------------------------
# Capture tests
# ----------------------------

def test_capture_output_shape(sensor):

    photons = np.ones(
        (512,512)
    ) * 100


    image = sensor.capture(
        photons,
        exposure=0.005
    )


    assert image.shape == photons.shape



def test_capture_dtype(sensor):

    photons = np.ones(
        (512,512)
    ) * 100


    image = sensor.capture(
        photons,
        exposure=0.005
    )


    assert image.dtype == np.uint16



def test_capture_range(sensor):

    photons = np.ones(
        (512,512)
    ) * 10000


    image = sensor.capture(
        photons,
        exposure=0.005
    )


    assert np.min(image) >= 0

    assert np.max(image) <= (
        2**sensor.adc_bits - 1
    )



# ----------------------------
# Physical behavior tests
# ----------------------------

def test_zero_photons_gives_dark_image(sensor):

    photons = np.zeros(
        (512,512)
    )


    image = sensor.capture(
        photons,
        exposure=0
    )


    # It should not be exactly zero because
    # read noise exists

    assert np.mean(image) < 10



def test_high_signal_saturates(sensor):

    photons = np.ones(
        (512,512)
    ) * 1e8


    image = sensor.capture(
        photons,
        exposure=0.01
    )


    max_dn = (
        2**sensor.adc_bits - 1
    )


    assert np.max(image) == max_dn



# ----------------------------
# Reproducibility test
# ----------------------------

def test_seed_reproducibility():

    photons = np.ones(
        (20,20)
    ) * 100


    sensor1 = Sensor(
        20,
        20,
        2.74e-6,
        0.68,
        2.3,
        0.1,
        9400,
        12,
        seed=123
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
        seed=123
    )


    image1 = sensor1.capture(
        photons,
        0.005
    )

    image2 = sensor2.capture(
        photons,
        0.005
    )


    assert np.array_equal(
        image1,
        image2
    )



# ----------------------------
# Save config test
# ----------------------------

def test_save_config(sensor, tmp_path):

    filename = tmp_path / "sensor.json"


    sensor.save_config(filename)


    assert filename.exists()


    with open(filename) as f:

        data = json.load(f)


    assert data["width"] == 512
    assert data["adc_bits"] == 12
