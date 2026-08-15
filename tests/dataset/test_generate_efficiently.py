import h5py
import numpy as np
import pytest

import lgbeam.dataset.generate_efficiently as ge


def setup_worker_state():
    ge._WORKER_R = np.zeros((4, 4))
    ge._WORKER_PHI = np.zeros((4, 4))
    ge._WORKER_SENSOR_KWARGS = {
        "width": 2, "height": 2, "pixel_size": 1.0,
        "qe": 1.0, "read_noise": 0.0, "dark_current": 0.0,
        "full_well": 65535, "adc_bits": 16,
    }
    ge._WORKER_W0_RANGE = (1.0, 2.0)
    ge._WORKER_WAVELENGTH = 532e-9
    ge._WORKER_POWER_RANGE = (10, 20)
    ge._WORKER_EXPOSURE = 1.0
    ge._WORKER_MAX_CENTER_SHIFT = 1


# ---------------------------------------------------------------------------
# Test _validate_input()
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_inputs(self):
        ge._validate_inputs([(0, 0), (1, 0)], 10, (1.0, 2.0), (1, 10), 2, 16, (32, 32))

    def test_valid_numpy_integer_inputs(self):
        ge._validate_inputs([(np.int64(0), np.int32(1))], np.int64(10), (np.float64(1), np.float64(2)), (np.int64(1), np.int32(10)), np.int64(2), np.int64(16), (32, 32))

    @pytest.mark.parametrize("value", [None, "[(0, 0)]", {(0, 0)}, 123])
    def test_lg_modes_must_be_list_or_tuple(self, value):
        with pytest.raises(TypeError, match="lg_modes must be a list or tuple"):
            ge._validate_inputs(value, 10, (1.0, 2.0), (1, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("modes", [[(0,)], [(0, 0, 1)], [0], ["0,0"]])
    def test_invalid_mode_structure(self, modes):
        with pytest.raises(ValueError, match="Each element of lg_modes"):
            ge._validate_inputs(modes, 10, (1.0, 2.0), (1, 10), 2, 16, (32, 32))

    def test_non_integer_mode_indices(self):
        with pytest.raises(TypeError, match="LG mode indices must be integers"):
            ge._validate_inputs([(0, 0.5)], 10, (1.0, 2.0), (1, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [0, -1])
    def test_n_images_per_class_must_be_positive(self, value):
        with pytest.raises(ValueError, match="n_images_per_class must be greater than 0"):
            ge._validate_inputs([(0, 0)], value, (1.0, 2.0), (1, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [1.5, "10", None])
    def test_n_images_per_class_must_be_integer(self, value):
        with pytest.raises(TypeError, match="n_images_per_class must be an integer"):
            ge._validate_inputs([(0, 0)], value, (1.0, 2.0), (1, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [(1.0,), (1.0, 2.0, 3.0), "1,2", None])
    def test_invalid_w0_range_structure(self, value):
        with pytest.raises(ValueError, match="w0_range must contain exactly two"):
            ge._validate_inputs([(0, 0)], 10, value, (1, 10), 2, 16, (32, 32))

    def test_invalid_w0_range_type(self):
        with pytest.raises(TypeError, match="w0_range values must be numeric"):
            ge._validate_inputs([(0, 0)], 10, ("a", 2.0), (1, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [(0.0, 2.0), (-1.0, 2.0), (2.0, 1.0)])
    def test_invalid_w0_range_values(self, value):
        with pytest.raises(ValueError):
            ge._validate_inputs([(0, 0)], 10, value, (1, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [(1,), (1, 2, 3), "1,2", None])
    def test_invalid_power_range_structure(self, value):
        with pytest.raises(ValueError, match="power_range must contain exactly two"):
            ge._validate_inputs([(0, 0)], 10, (1.0, 2.0), value, 2, 16, (32, 32))

    def test_power_range_must_contain_integers(self):
        with pytest.raises(TypeError, match="power_range values must be integers"):
            ge._validate_inputs([(0, 0)], 10, (1.0, 2.0), (1.5, 10), 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [(0, 10), (-1, 10), (10, 1)])
    def test_invalid_power_range_values(self, value):
        with pytest.raises(ValueError):
            ge._validate_inputs([(0, 0)], 10, (1.0, 2.0), value, 2, 16, (32, 32))

    @pytest.mark.parametrize("value", [-1, 1.5, "2"])
    def test_invalid_max_center_shift(self, value):
        with pytest.raises((TypeError, ValueError)):
            ge._validate_inputs([(0, 0)], 10, (1.0, 2.0), (1, 10), value, 16, (32, 32))

    @pytest.mark.parametrize("value", [0, -1, 17, 32, 1.5, "16"])
    def test_invalid_adc_bits(self, value):
        with pytest.raises((TypeError, ValueError)):
            ge._validate_inputs([(0, 0)], 10, (1.0, 2.0), (1, 10), 2, value, (32, 32))

    @pytest.mark.parametrize("sensor_size", [None, 32, (32,), (32, 32, 32), (0, 32), (32, 0), (-1, 32), (32, -1)])
    def test_invalid_sensor_size(self, sensor_size):
        with pytest.raises((TypeError, ValueError)):
            ge._validate_inputs([(0, 0)], 10, (1.0, 2.0), (1, 10), 2, 16, sensor_size)


# ---------------------------------------------------------------------------
# Test _resize_uint16()
# ---------------------------------------------------------------------------

class TestResizeUint16:
    def test_output_shape(self):
        result = ge._resize_uint16(np.arange(16, dtype=np.uint16).reshape(4, 4), (2, 2))
        assert result.shape == (2, 2)

    def test_output_dtype_is_uint16(self):
        result = ge._resize_uint16(np.arange(16, dtype=np.uint16).reshape(4, 4), (2, 2))
        assert result.dtype == np.uint16

    def test_identity_resize(self):
        image = np.arange(16, dtype=np.uint16).reshape(4, 4)
        np.testing.assert_array_equal(ge._resize_uint16(image, (4, 4)), image)

    def test_downsampling_constant_image(self):
        result = ge._resize_uint16(np.ones((8, 8), dtype=np.uint16) * 100, (2, 2))
        assert result.shape == (2, 2)
        assert np.all(result == 100)

    def test_accepts_non_uint16_input(self):
        result = ge._resize_uint16(np.ones((4, 4), dtype=np.float32) * 100, (2, 2))
        assert result.dtype == np.uint16


# ---------------------------------------------------------------------------
# Test _init_worker()
# ---------------------------------------------------------------------------

def test_init_worker_sets_global_state():
    r = np.array([[1, 2]])
    phi = np.array([[3, 4]])
    sensor_kwargs = {"width": 10, "height": 20}
    args = (r, phi, sensor_kwargs, (1.0, 2.0), 532e-9, (1, 10), 0.01, 5)
    ge._init_worker(*args)
    assert ge._WORKER_R is r
    assert ge._WORKER_PHI is phi
    assert ge._WORKER_SENSOR_KWARGS is sensor_kwargs
    assert ge._WORKER_W0_RANGE == (1.0, 2.0)
    assert ge._WORKER_WAVELENGTH == 532e-9
    assert ge._WORKER_POWER_RANGE == (1, 10)
    assert ge._WORKER_EXPOSURE == 0.01
    assert ge._WORKER_MAX_CENTER_SHIFT == 5


# ---------------------------------------------------------------------------
# Test random seed
# ---------------------------------------------------------------------------

def test_generate_one_image_passes_seed_to_sensor(monkeypatch):
    setup_worker_state()
    received = {}

    class FakeSensor:
        def __init__(self, **kwargs):
            # Take seed value from kwargs
            received["seed"] = kwargs["seed"]
        def capture(self, photons, exposure):
            return np.ones((2, 2), dtype=np.uint16)

    monkeypatch.setattr(ge, "Sensor", FakeSensor)
    monkeypatch.setattr(ge, "LaguerreGauss", lambda *a, **k: np.ones((4, 4), dtype=np.float32))
    # input: l, p, seed
    result = ge._generate_one_image((0, 0, 123456))
    assert received["seed"] == 123456
    assert result.shape == (ge.OUTPUT_SIZE[1], ge.OUTPUT_SIZE[0])
    assert result.dtype == np.uint16


def test_generate_one_image_same_seed_is_deterministic(monkeypatch):
    setup_worker_state()

    class FakeSensor:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
        def capture(self, photons, exposure):
            return np.full((2, 2), self.seed % 65535, dtype=np.uint16)

    monkeypatch.setattr(ge, "Sensor", FakeSensor)
    monkeypatch.setattr(ge, "LaguerreGauss", lambda *a, **k: np.ones((4, 4), dtype=np.float32))
    np.testing.assert_array_equal(
        ge._generate_one_image((0, 0, 123)),
        ge._generate_one_image((0, 0, 123)),
    )


def test_generate_one_image_different_seed_changes_result(monkeypatch):
    setup_worker_state()

    class FakeSensor:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
        def capture(self, photons, exposure):
            return np.full((2, 2), self.seed % 65535, dtype=np.uint16)

    monkeypatch.setattr(ge, "Sensor", FakeSensor)
    monkeypatch.setattr(ge, "LaguerreGauss", lambda *a, **k: np.ones((4, 4), dtype=np.float32))
    image1 = ge._generate_one_image((0, 0, 123))
    image2 = ge._generate_one_image((0, 0, 456))
    assert not np.array_equal(image1, image2)
   

# ---------------------------------------------------------------------------
# Test invalid photons number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [np.zeros((4, 4), dtype=np.float32), np.full((4, 4), np.nan, dtype=np.float32)])
def test_generate_one_image_rejects_invalid_photons(monkeypatch, value):
    setup_worker_state()
    monkeypatch.setattr(ge, "Sensor", lambda **kwargs: None)
    monkeypatch.setattr(ge, "LaguerreGauss", lambda *a, **k: value)
    with pytest.raises(ValueError, match="Invalid maximum photon value"):
        ge._generate_one_image((0, 0, 123))


# ---------------------------------------------------------------------------
# Test images generation
# ---------------------------------------------------------------------------

def test_each_image_gets_a_seed():
    tasks = list(ge._make_image_tasks((2, 1), 100, 12345))
    seeds = [task[2] for task in tasks]
    assert len(tasks) == 100
    assert all(task[:2] == (2, 1) for task in tasks)
    assert all(isinstance(seed, int) for seed in seeds)
    assert all(0 <= seed < 2**32 - 1 for seed in seeds)
    assert len(set(seeds)) == len(seeds)


def test_image_tasks_are_reproducible():
    assert list(ge._make_image_tasks((2, 1), 100, 12345)) == list(ge._make_image_tasks((2, 1), 100, 12345))


def test_different_mode_seed_changes_image_seeds():
    seeds1 = [t[2] for t in ge._make_image_tasks((2, 1), 100, 12345)]
    seeds2 = [t[2] for t in ge._make_image_tasks((2, 1), 100, 54321)]
    assert seeds1 != seeds2


# ---------------------------------------------------------------------------
# HDF5 file test
# ---------------------------------------------------------------------------

def test_generate_mode_file_creates_expected_hdf5(monkeypatch, tmp_path):
    monkeypatch.setattr(ge, "OUTPUT_DIR", tmp_path)
    # monkeypatch.setattr(ge, "OUTPUT_SIZE", (2, 2))
    setup_worker_state()

    class FakeSensor:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
        def capture(self, photons, exposure):
            return np.full((ge.OUTPUT_SIZE[1], ge.OUTPUT_SIZE[0]), self.seed % 65535, dtype=np.uint16)

    monkeypatch.setattr(ge, "Sensor", FakeSensor)
    monkeypatch.setattr(ge, "LaguerreGauss", lambda *a, **k: np.ones((4, 4), dtype=np.float32))

    result = ge._generate_mode_file(
        mode=(0, 0), n_images_per_class=4, mode_seed=123,
        r=np.zeros((4, 4)), phi=np.zeros((4, 4)),
        sensor_kwargs=ge._WORKER_SENSOR_KWARGS, w0_range=(1.0, 2.0),
        wavelength=532e-9, power_range=(1, 10), exposure=1.0,
        max_center_shift=1, n_workers=1,
    )

    expected_tasks = list(ge._make_image_tasks((0, 0), 4, 123))
    expected_values = [seed % 65535 for _, _, seed in expected_tasks]    
    assert result == tmp_path / "mode_l0_p0.h5"

    with h5py.File(result, "r") as h5:
        assert list(h5.keys()) == ["images"]
        images = h5["images"]
        assert images.shape == (4, ge.OUTPUT_SIZE[1], ge.OUTPUT_SIZE[0])
        assert images.dtype == np.uint16
        assert images.chunks == (1, ge.OUTPUT_SIZE[1], ge.OUTPUT_SIZE[0])
        assert images.compression == "gzip"
        assert images.compression_opts == 4
        assert images.shuffle is True
        data = images[:]

    assert data.shape == (4, ge.OUTPUT_SIZE[1], ge.OUTPUT_SIZE[0])
    assert data.dtype == np.uint16
    assert np.all(data <= np.iinfo(np.uint16).max)


def test_generate_dataset_creates_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ge, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ge, "OUTPUT_SIZE", (2, 2))
    monkeypatch.setattr(ge, "create_mesh", lambda l, n: (np.zeros((2, 2)), np.zeros((2, 2))))
    calls = []

    def fake_generate_mode_file(**kwargs):
        calls.append(kwargs)
        path = tmp_path / f"mode_l{kwargs['mode'][0]}_p{kwargs['mode'][1]}.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("images", shape=(kwargs["n_images_per_class"], 2, 2), dtype=np.uint16)
        return path

    monkeypatch.setattr(ge, "_generate_mode_file", fake_generate_mode_file)
    ge.generate_dataset(
        l=1, n=1, lg_modes=[(0, 0), (1, 0)], n_images_per_class=2,
        sensor_size=(2, 2), pixel_size=1.0, qe=1.0, read_noise=0.0,
        dark_current=0.0, full_well=65535, adc_bits=16, seed=12345,
        w0_range=(1.0, 2.0), wavelength=532e-9, power_range=(1, 10),
        exposure=1.0, max_center_shift=0, n_workers=1,
    )
    assert len(calls) == 2
    assert calls[0]["mode"] == (0, 0)
    assert calls[1]["mode"] == (1, 0)
    assert calls[0]["mode_seed"] != calls[1]["mode_seed"]
    assert (tmp_path / "mode_l0_p0.h5").exists()
    assert (tmp_path / "mode_l1_p0.h5").exists()


def test_generate_dataset_seed_none(monkeypatch, tmp_path):
    monkeypatch.setattr(ge, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ge, "OUTPUT_SIZE", (2, 2))
    monkeypatch.setattr(ge, "create_mesh", lambda l, n: (np.zeros((2, 2)), np.zeros((2, 2))))
    received = {}

    def fake_generate_mode_file(**kwargs):
        received["mode_seed"] = kwargs["mode_seed"]
        path = tmp_path / "mode_l0_p0.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("images", shape=(1, 2, 2), dtype=np.uint16)
        return path

    monkeypatch.setattr(ge, "_generate_mode_file", fake_generate_mode_file)
    ge.generate_dataset(
        lg_modes=[(0, 0)], n_images_per_class=1, sensor_size=(2, 2),
        pixel_size=1.0, qe=1.0, read_noise=0.0, dark_current=0.0,
        full_well=65535, adc_bits=16, seed=None, w0_range=(1.0, 2.0),
        wavelength=532e-9, power_range=(1, 10), exposure=1.0,
        max_center_shift=0, n_workers=1,
    )
    assert isinstance(received["mode_seed"], int)
    assert 0 <= received["mode_seed"] < 2**32 - 1
