import pytest
import numpy as np
import h5py
from pathlib import Path
import torch


from training.data import (
    ADC_MAX,
    get_indices,
    get_class_labels,
    calculate_transformation_params,
    MyDataset
)

# ==========================================================
# Test get_indices
# ==========================================================

@pytest.mark.parametrize(
    "ratios",
    [
        [0, 1],
        np.array,
        1,
        "1",
        (1)
    ],
)
def test_get_indices_ratios_for_datatypes(ratios):
    with pytest.raises(TypeError):
         get_indices(ratios, 100)


@pytest.mark.parametrize(
    "N",
    [
        "1",
        1.0,
        np.array([0]),
        (0, 0),
        (0, 0.)
    ],
)
def test_get_indices_N_for_datatypes(N):
    with pytest.raises(TypeError):
            get_indices((0.8, 0.1), N)


@pytest.mark.parametrize(
    "ratios",
    [
        (0.,  0.),
        (1., 0.1,),
        (-0.1, 0.1),
        (0.1, -0.1),
        (1.1, 0.)
    ],
)
def test_get_indices_ratios_for_values(ratios):
    with pytest.raises(ValueError):
         get_indices(ratios, 100)


@pytest.mark.parametrize(
    "ratios",
    [
        (np.nan, 0.1),
        (0.8, np.nan),
        (np.inf, 0.1),
        (0.8, np.inf),
        (-np.inf, 0.1),
        (0.8, -np.inf),
    ],
)
def test_get_indices_ratios_for_non_finite_values(ratios):
    with pytest.raises(ValueError):
        get_indices(ratios, 100)


@pytest.mark.parametrize(
    "ratios",
    [
        (),
        (0.8,),
        (0.8, 0.1, 0.1),
    ],
)
def test_get_indices_ratios_for_wrong_length(ratios):
    with pytest.raises(ValueError):
        get_indices(ratios, 100)


@pytest.mark.parametrize(
    "N",
    [
        -1,
        0,
    ],
)
def test_get_indices_N_for_values(N):
    with pytest.raises(ValueError):
         get_indices((0.9, 0.1), N)


def test_get_indices_accepts_numpy_integer_N():
    N = np.int64(100)

    train, test, val = get_indices((0.8, 0.1), N)

    assert len(train) == 80
    assert len(test) == 10
    assert len(val) == 10


def test_get_indices_accepts_numpy_float_ratios():
    ratios = (np.float64(0.8), np.float64(0.1))

    train, test, val = get_indices(ratios, 100)

    assert len(train) == 80
    assert len(test) == 10
    assert len(val) == 10


def test_get_indices_for_output_with_val():
    ratios = (0.8, 0.1)
    N = 15

    ind_train, ind_test, ind_val = get_indices(ratios, N)

    assert len(ind_train) == int(N * ratios[0])
    assert len(ind_test) == int(N * ratios[1])
    assert len(ind_val) == N - len(ind_train) - len(ind_test)


def test_get_indices_returns_numpy_arrays():
    train, test, val = get_indices((0.8, 0.1), 100)

    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)
    assert isinstance(val, np.ndarray)


def test_get_indices_for_output_no_val():
    ratios = (0.9, 0.1)
    N = 15

    ind_train, ind_test, ind_val = get_indices(ratios, N)

    assert len(ind_train) == int(N * ratios[0])
    assert len(ind_test) == int(N * ratios[1])
    assert len(ind_val) == N - len(ind_train) - len(ind_test)


def test_get_indices_are_unique_and_cover_entire_dataset():
    N = 100

    train, test, val = get_indices((0.8, 0.1), N)

    all_indices = np.concatenate([train, test, val])

    assert len(all_indices) == N
    assert len(np.unique(all_indices)) == N
    assert np.array_equal(np.sort(all_indices), np.arange(N))


def test_get_indices_with_no_validation():
    train, test, val = get_indices((0.9, 0.1), 100)

    assert len(train) == 90
    assert len(test) == 10
    assert len(val) == 0
    assert isinstance(val, np.ndarray)


# ==========================================================
# Test get_class_labels
# ==========================================================

# ==========================================================
# Test get_class_labels
# ==========================================================

def assert_array_lists_equal(actual, expected):
    assert len(actual) == len(expected)

    for actual_array, expected_array in zip(actual, expected):
        assert np.array_equal(actual_array, expected_array)


# ----------------------------------------------------------
# Valid inputs / outputs
# ----------------------------------------------------------

def test_get_class_labels_output_with_val():
    train = [
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
    ]
    test = [
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
    ]
    val = [
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
    ]

    ctrain, ctest, cval = get_class_labels(train, test, val)

    expected = [
        np.array([0, 0, 0]),
        np.array([1, 1, 1]),
        np.array([2, 2, 2]),
    ]

    assert_array_lists_equal(ctrain, expected)
    assert_array_lists_equal(ctest, expected)
    assert_array_lists_equal(cval, expected)


def test_get_class_labels_output_without_val():
    train = [
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
    ]
    test = [
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
    ]

    ctrain, ctest, cval = get_class_labels(train, test, [])

    expected = [
        np.array([0, 0, 0]),
        np.array([1, 1, 1]),
        np.array([2, 2, 2]),
    ]

    assert_array_lists_equal(ctrain, expected)
    assert_array_lists_equal(ctest, expected)
    assert cval == []


def test_get_class_labels_preserves_number_of_classes():
    train = [
        np.array([10]),
        np.array([20]),
        np.array([30]),
        np.array([40]),
        np.array([50]),
    ]
    test = [
        np.array([60]),
        np.array([70]),
        np.array([80]),
        np.array([90]),
        np.array([100]),
    ]
    val = [
        np.array([110]),
        np.array([120]),
        np.array([130]),
        np.array([140]),
        np.array([150]),
    ]

    ctrain, ctest, cval = get_class_labels(train, test, val)

    expected = [
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([3]),
        np.array([4]),
    ]

    assert_array_lists_equal(ctrain, expected)
    assert_array_lists_equal(ctest, expected)
    assert_array_lists_equal(cval, expected)


def test_get_class_labels_preserves_each_class_length():
    train = [
        np.array([1]),
        np.array([1, 2]),
        np.array([1, 2, 3]),
    ]
    test = [
        np.array([4]),
        np.array([4, 5]),
        np.array([4, 5, 6]),
    ]
    val = [
        np.array([7]),
        np.array([7, 8]),
        np.array([7, 8, 9]),
    ]

    ctrain, ctest, cval = get_class_labels(train, test, val)

    assert [len(x) for x in ctrain] == [1, 2, 3]
    assert [len(x) for x in ctest] == [1, 2, 3]
    assert [len(x) for x in cval] == [1, 2, 3]


@pytest.mark.parametrize(
    "bad_value",
    [
        None,
        (),
        np.array([0, 1]),
        "indices",
        1,
        1.0,
    ],
)
def test_get_class_labels_rejects_non_list_train(bad_value):
    valid = [np.array([0, 1, 2])]

    with pytest.raises(TypeError):
        get_class_labels(bad_value, valid, valid)


@pytest.mark.parametrize(
    "bad_value",
    [
        None,
        (),
        np.array([0, 1]),
        "indices",
        1,
        1.0,
    ],
)
def test_get_class_labels_rejects_non_list_test(bad_value):
    valid = [np.array([0, 1, 2])]

    with pytest.raises(TypeError):
        get_class_labels(valid, bad_value, valid)


@pytest.mark.parametrize(
    "bad_value",
    [
        None,
        (),
        np.array([0, 1]),
        "indices",
        1,
        1.0,
    ],
)
def test_get_class_labels_rejects_non_list_validation(bad_value):
    valid = [np.array([0, 1, 2])]

    with pytest.raises(TypeError):
        get_class_labels(valid, valid, bad_value)


@pytest.mark.parametrize(
    "bad_element",
    [
        [0, 1, 2],
        (0, 1, 2),
        "indices",
        1,
        1.0,
        None,
    ],
)
def test_get_class_labels_rejects_non_numpy_train_elements(bad_element):
    valid = [np.array([0, 1, 2])]

    with pytest.raises(TypeError):
        get_class_labels([bad_element], valid, valid)


@pytest.mark.parametrize(
    "bad_element",
    [
        [0, 1, 2],
        (0, 1, 2),
        "indices",
        1,
        1.0,
        None,
    ],
)
def test_get_class_labels_rejects_non_numpy_test_elements(bad_element):
    valid = [np.array([0, 1, 2])]

    with pytest.raises(TypeError):
        get_class_labels(valid, [bad_element], valid)


@pytest.mark.parametrize(
    "bad_element",
    [
        [0, 1, 2],
        (0, 1, 2),
        "indices",
        1,
        1.0,
        None,
    ],
)
def test_get_class_labels_rejects_non_numpy_validation_elements(bad_element):
    valid = [np.array([0, 1, 2])]

    with pytest.raises(TypeError):
        get_class_labels(valid, valid, [bad_element])


@pytest.mark.parametrize(
    "bad_index",
    [
        1.0,
        np.float64(1.0),
        "1",
        None,
    ],
)
def test_get_class_labels_rejects_non_integer_train_indices(bad_index):
    valid = [np.array([0, 1, 2])]

    bad_train = [np.array([bad_index], dtype=object)]

    with pytest.raises(TypeError):
        get_class_labels(bad_train, valid, valid)


@pytest.mark.parametrize(
    "bad_index",
    [
        1.0,
        np.float64(1.0),
        "1",
        None,
    ],
)
def test_get_class_labels_rejects_non_integer_test_indices(bad_index):
    valid = [np.array([0, 1, 2])]

    bad_test = [np.array([bad_index], dtype=object)]

    with pytest.raises(TypeError):
        get_class_labels(valid, bad_test, valid)


@pytest.mark.parametrize(
    "bad_index",
    [
        1.0,
        np.float64(1.0),
        "1",
        None,
    ],
)
def test_get_class_labels_rejects_non_integer_validation_indices(bad_index):
    valid = [np.array([0, 1, 2])]

    bad_val = [np.array([bad_index], dtype=object)]

    with pytest.raises(TypeError):
        get_class_labels(valid, valid, bad_val)


def test_get_class_labels_accepts_numpy_integer_indices():
    train = [np.array([np.int64(0), np.int32(1)])]
    test = [np.array([np.int64(2), np.int32(3)])]
    val = [np.array([np.int64(4), np.int32(5)])]

    ctrain, ctest, cval = get_class_labels(train, test, val)

    expected = [np.array([0, 0])]

    assert_array_lists_equal(ctrain, expected)
    assert_array_lists_equal(ctest, expected)
    assert_array_lists_equal(cval, expected)


@pytest.mark.parametrize(
    "position",
    [0, 1, 2],
)
def test_get_class_labels_rejects_negative_indices(position):
    train = [np.array([0, 1, 2])]
    test = [np.array([0, 1, 2])]
    val = [np.array([0, 1, 2])]

    datasets = [train, test, val]
    datasets[position][0][0] = -1

    with pytest.raises(ValueError):
        get_class_labels(train, test, val)


def test_get_class_labels_rejects_different_train_test_lengths():
    train = [
        np.array([0, 1]),
        np.array([0, 1]),
    ]
    test = [
        np.array([0, 1]),
    ]

    with pytest.raises(ValueError):
        get_class_labels(train, test, [])


def test_get_class_labels_rejects_different_train_validation_lengths():
    train = [
        np.array([0, 1]),
        np.array([0, 1]),
    ]
    test = [
        np.array([0, 1]),
        np.array([0, 1]),
    ]
    val = [
        np.array([0, 1]),
    ]

    with pytest.raises(ValueError):
        get_class_labels(train, test, val)


def test_get_class_labels_allows_empty_validation_dataset():
    train = [
        np.array([0, 1]),
        np.array([0, 1]),
    ]
    test = [
        np.array([0, 1]),
        np.array([0, 1]),
    ]

    ctrain, ctest, cval = get_class_labels(train, test, [])

    assert_array_lists_equal(
        ctrain,
        [
            np.array([0, 0]),
            np.array([1, 1]),
        ],
    )

    assert_array_lists_equal(
        ctest,
        [
            np.array([0, 0]),
            np.array([1, 1]),
        ],
    )

    assert cval == []


def test_get_class_labels_handles_empty_class_arrays():
    train = [
        np.array([]),
        np.array([0, 1, 2]),
    ]
    test = [
        np.array([]),
        np.array([3, 4, 5]),
    ]
    val = [
        np.array([]),
        np.array([6, 7, 8]),
    ]

    ctrain, ctest, cval = get_class_labels(train, test, val)

    assert_array_lists_equal(
        ctrain,
        [
            np.array([]),
            np.array([1, 1, 1]),
        ],
    )

    assert_array_lists_equal(
        ctest,
        [
            np.array([]),
            np.array([1, 1, 1]),
        ],
    )

    assert_array_lists_equal(
        cval,
        [
            np.array([]),
            np.array([1, 1, 1]),
        ],
    )


def test_get_class_labels_handles_zero_classes():
    ctrain, ctest, cval = get_class_labels([], [], [])

    assert ctrain == []
    assert ctest == []
    assert cval == []

# ==========================================================
# Test calculate_transformation_params
# ==========================================================

def create_h5_file(path, images, dataset_name="images"):
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=images)


def test_calculate_transformation_params_output(tmp_path):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            [[0, 0], [0, 0]],
            [[ADC_MAX, ADC_MAX], [ADC_MAX, ADC_MAX]],
        ],
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    mean, std = calculate_transformation_params(
        [file],
        [np.array([0, 1])],
    )

    assert np.isclose(mean, 0.5)
    assert np.isclose(std, 0.5)


def test_calculate_transformation_params_uses_only_selected_indices(tmp_path):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            np.zeros((2, 2)),
            np.full((2, 2), ADC_MAX),
            np.full((2, 2), ADC_MAX),
        ],
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    mean, std = calculate_transformation_params(
        [file],
        [np.array([0])],
    )

    assert np.isclose(mean, 0.0)
    assert np.isclose(std, 0.0)


def test_calculate_transformation_params_multiple_files(tmp_path):
    file1 = tmp_path / "data1.h5"
    file2 = tmp_path / "data2.h5"

    create_h5_file(
        file1,
        np.array([np.zeros((2, 2))], dtype=np.uint16),
    )

    create_h5_file(
        file2,
        np.array([np.full((2, 2), ADC_MAX)], dtype=np.uint16),
    )

    mean, std = calculate_transformation_params(
        [file1, file2],
        [
            np.array([0]),
            np.array([0]),
        ],
    )

    assert np.isclose(mean, 0.5)
    assert np.isclose(std, 0.5)


def test_calculate_transformation_params_multiple_batches(tmp_path):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            np.zeros((2, 2)),
            np.full((2, 2), ADC_MAX),
            np.zeros((2, 2)),
            np.full((2, 2), ADC_MAX),
        ],
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    mean, std = calculate_transformation_params(
        [file],
        [np.array([0, 1, 2, 3])],
        n_subset=1,
    )

    assert np.isclose(mean, 0.5)
    assert np.isclose(std, 0.5)


def test_calculate_transformation_params_no_images_selected(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError):
        calculate_transformation_params(
            [file],
            [np.array([], dtype=int)],
        )


@pytest.mark.parametrize(
    "files",
    [
        None,
        (),
        "data.h5",
        np.array(["data.h5"]),
    ],
)
def test_calculate_transformation_params_files_must_be_list(files):
    with pytest.raises(TypeError):
        calculate_transformation_params(files, [])


@pytest.mark.parametrize(
    "file",
    [
        123,
        1.0,
        None,
        [],
        {},
    ],
)
def test_calculate_transformation_params_invalid_filepath_type(file):
    with pytest.raises(TypeError):
        calculate_transformation_params(
            [file],
            [np.array([0])],
        )


def test_calculate_transformation_params_empty_filepath():
    with pytest.raises(ValueError):
        calculate_transformation_params(
            [""],
            [np.array([0])],
        )


def test_calculate_transformation_params_accepts_path(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((1, 2, 2), dtype=np.uint16),
    )

    mean, std = calculate_transformation_params(
        [file],
        [np.array([0])],
    )

    assert mean == 0.0
    assert std == 0.0


def test_calculate_transformation_params_files_indices_length_mismatch(
    tmp_path,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((1, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError):
        calculate_transformation_params(
            [file],
            [
                np.array([0]),
                np.array([0]),
            ],
        )


@pytest.mark.parametrize(
    "indices",
    [
        None,
        (),
        "indices",
        np.array([0, 1]),
    ],
)
def test_calculate_transformation_params_indices_must_be_list(indices):
    with pytest.raises(TypeError):
        calculate_transformation_params([], indices)


@pytest.mark.parametrize(
    "bad_indices",
    [
        [[0, 1]],
        [(0, 1)],
        [["0", "1"]],
        [None],
    ],
    ids=[
        "list",
        "tuple",
        "list_of_strings",
        "none",
    ],
)
def test_calculate_transformation_params_indices_elements_must_be_arrays(
    tmp_path,
    bad_indices,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(TypeError):
        calculate_transformation_params(
            [file],
            bad_indices,
        )


@pytest.mark.parametrize(
    "index",
    [
        1.0,
        "1",
        None,
        1.5,
    ],
)
def test_calculate_transformation_params_indices_must_be_integers(
    index,
    tmp_path,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    indices = [
        np.array([index], dtype=object)
    ]

    with pytest.raises(TypeError):
        calculate_transformation_params(
            [file],
            indices,
        )


def test_calculate_transformation_params_negative_index(
    tmp_path,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError):
        calculate_transformation_params(
            [file],
            [np.array([-1])],
        )


def test_calculate_transformation_params_duplicate_indices(
    tmp_path,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((3, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError):
        calculate_transformation_params(
            [file],
            [np.array([0, 0, 1])],
        )


def test_calculate_transformation_params_index_out_of_bounds(
    tmp_path,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((3, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(IndexError):
        calculate_transformation_params(
            [file],
            [np.array([3])],
        )


def test_calculate_transformation_params_missing_images_dataset(
    tmp_path,
):
    file = tmp_path / "data.h5"

    with h5py.File(file, "w") as f:
        f.create_dataset(
            "something_else",
            data=np.zeros((2, 2, 2)),
        )

    with pytest.raises(KeyError):
        calculate_transformation_params(
            [file],
            [np.array([0])],
        )


@pytest.mark.parametrize(
    "n_subset",
    [
        1.0,
        "256",
        None,
        1.5,
    ],
)
def test_calculate_transformation_params_n_subset_type(n_subset):
    with pytest.raises(TypeError):
        calculate_transformation_params(
            [],
            [],
            n_subset=n_subset,
        )


@pytest.mark.parametrize(
    "n_subset",
    [
        0,
        -1,
        -100,
    ],
)
def test_calculate_transformation_params_n_subset_value(n_subset):
    with pytest.raises(ValueError):
        calculate_transformation_params(
            [],
            [],
            n_subset=n_subset,
        )

# ==========================================================
# Test MyDataset
# ==========================================================

@pytest.mark.parametrize(
    "files",
    [
        None,
        (),
        "data.h5",
        np.array(["data.h5"]),
    ],
)
def test_mydataset_files_must_be_list(files):
    with pytest.raises(TypeError):
        MyDataset(files, [])


@pytest.mark.parametrize(
    "indices",
    [
        None,
        (),
        "indices",
        np.array([0, 1]),
    ],
)
def test_mydataset_indices_must_be_list(indices):
    with pytest.raises(TypeError):
        MyDataset([], indices)


@pytest.mark.parametrize(
    "file",
    [
        123,
        1.0,
        None,
        [],
        {},
    ],
)
def test_mydataset_file_type(file):
    with pytest.raises(TypeError):
        MyDataset([file], [np.array([0])])


def test_mydataset_file_must_exist(tmp_path):
    file = tmp_path / "does_not_exist.h5"

    with pytest.raises(FileNotFoundError):
        MyDataset(
            [file],
            [np.array([0])],
        )


def test_mydataset_empty_filepath():
    with pytest.raises(ValueError):
        MyDataset(
            [""],
            [np.array([0])],
        )


@pytest.mark.parametrize(
        "bad_indices",
    [
        [[0, 1]],
        [(0, 1)],
        [["0", "1"]],
        ["indices"],
        [None],
    ],
    ids=[
        "list",
        "tuple",
        "list_of_strings",
        "string",
        "none",
    ],
)
def test_mydataset_indices_elements_must_be_numpy_arrays(
    tmp_path,
    bad_indices,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(TypeError):
        MyDataset([file], bad_indices)


@pytest.mark.parametrize(
    "bad_index",
    [
        1.0,
        "1",
        None,
        1.5,
    ],
)
def test_mydataset_indices_must_be_integers(
    tmp_path,
    bad_index,
):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    indices = [
        np.array([bad_index], dtype=object)
    ]

    with pytest.raises(TypeError):
        MyDataset([file], indices)


def test_mydataset_indices_cannot_be_negative(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError):
        MyDataset(
            [file],
            [np.array([-1])],
        )


def test_mydataset_files_and_indices_same_length(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError):
        MyDataset(
            [file],
            [
                np.array([0]),
                np.array([1]),
            ],
        )







def test_my_dataset_initializes_with_string_paths(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((3, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [str(file)],
        [np.array([0, 1, 2])],
    )

    assert dataset.files == [file]
    assert dataset.transform is None


def test_my_dataset_initializes_with_path_objects(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([0, 1])],
    )

    assert dataset.files == [Path(file)]


def test_my_dataset_flattens_samples(tmp_path):
    file1 = tmp_path / "data1.h5"
    file2 = tmp_path / "data2.h5"

    create_h5_file(
        file1,
        np.zeros((3, 2, 2), dtype=np.uint16),
    )

    create_h5_file(
        file2,
        np.zeros((3, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file1, file2],
        [
            np.array([0, 2]),
            np.array([1, 2]),
        ],
    )

    assert dataset.samples == [
        (0, 0),
        (0, 2),
        (1, 1),
        (1, 2),
    ]


def test_my_dataset_converts_image_indices_to_int(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    indices = [
        np.array([np.int64(0), np.int64(1)])
    ]

    dataset = MyDataset([file], indices)

    assert dataset.samples == [
        (0, 0),
        (0, 1),
    ]

    assert all(
        isinstance(image_id, int)
        for _, image_id in dataset.samples
    )


def test_my_dataset_initializes_h5_handles_as_none(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([0, 1])],
    )

    assert dataset._h5_files == [None]
    assert dataset._datasets == [None]


def test_my_dataset_len_with_multiple_files(tmp_path):
    file1 = tmp_path / "data1.h5"
    file2 = tmp_path / "data2.h5"

    create_h5_file(
        file1,
        np.zeros((3, 2, 2), dtype=np.uint16),
    )

    create_h5_file(
        file2,
        np.zeros((5, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file1, file2],
        [
            np.array([0, 1]),
            np.array([0, 1, 2]),
        ],
    )

    assert len(dataset) == 5


def test_my_dataset_len_with_no_samples(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([], dtype=int)],
    )

    assert len(dataset) == 0


def test_my_dataset_get_dataset_opens_file_lazily(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    # File should not be opened during initialization.
    assert dataset._h5_files[0] is None
    assert dataset._datasets[0] is None

    dset = dataset._get_dataset(0)

    assert dataset._h5_files[0] is not None
    assert dataset._datasets[0] is not None
    assert dset is dataset._datasets[0]


def test_my_dataset_get_dataset_reuses_opened_dataset(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    dset1 = dataset._get_dataset(0)
    dset2 = dataset._get_dataset(0)

    assert dset1 is dset2


def test_my_dataset_get_dataset_missing_images_raises_key_error(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
        dataset_name="wrong_name",
    )

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    with pytest.raises(KeyError):
        dataset._get_dataset(0)


def test_my_dataset_getitem_returns_tensor(tmp_path):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            [[0, 1], [2, 3]],
        ],
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    result = dataset[0]

    assert isinstance(result, torch.Tensor)


def test_my_dataset_getitem_returns_float32(tmp_path):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            [[0, 1], [2, 3]],
        ],
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    result = dataset[0]

    assert result.dtype == torch.float32


def test_my_dataset_getitem_adds_channel_dimension(tmp_path):
    file = tmp_path / "data.h5"

    images = np.zeros(
        (1, 10, 20),
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    result = dataset[0]

    assert result.shape == (1, 10, 20)


def test_my_dataset_getitem_normalizes_by_adc_max(tmp_path):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            [
                [0, ADC_MAX],
                [ADC_MAX // 2, 1],
            ]
        ],
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    result = dataset[0]

    expected = torch.tensor(
        [
            [
                [0.0, 1.0],
                [
                    (ADC_MAX // 2) / ADC_MAX,
                    1 / ADC_MAX,
                ],
            ]
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(result, expected)


def test_my_dataset_getitem_returns_correct_sample_from_multiple_samples(
    tmp_path,
):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            np.zeros((2, 2), dtype=np.uint16),
            np.full((2, 2), ADC_MAX, dtype=np.uint16),
        ]
    )

    create_h5_file(file, images)

    dataset = MyDataset(
        [file],
        [np.array([0, 1])],
    )

    first = dataset[0]
    second = dataset[1]

    assert torch.allclose(
        first,
        torch.zeros((1, 2, 2)),
    )

    assert torch.allclose(
        second,
        torch.ones((1, 2, 2)),
    )


def test_my_dataset_getitem_works_with_multiple_files(tmp_path):
    file1 = tmp_path / "data1.h5"
    file2 = tmp_path / "data2.h5"

    create_h5_file(
        file1,
        np.zeros((1, 2, 2), dtype=np.uint16),
    )

    create_h5_file(
        file2,
        np.full(
            (1, 2, 2),
            ADC_MAX,
            dtype=np.uint16,
        ),
    )

    dataset = MyDataset(
        [file1, file2],
        [
            np.array([0]),
            np.array([0]),
        ],
    )

    first = dataset[0]
    second = dataset[1]

    assert torch.allclose(
        first,
        torch.zeros((1, 2, 2)),
    )

    assert torch.allclose(
        second,
        torch.ones((1, 2, 2)),
    )


def test_my_dataset_getitem_applies_transform(tmp_path):
    file = tmp_path / "data.h5"

    images = np.ones(
        (1, 2, 2),
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    def transform(image):
        return image * 2

    dataset = MyDataset(
        [file],
        [np.array([0])],
        transform=transform,
    )

    result = dataset[0]

    expected = torch.full(
        (1, 2, 2),
        2 / ADC_MAX,
        dtype=torch.float32,
    )

    assert torch.allclose(result, expected)


def test_my_dataset_transform_is_called_after_normalization(tmp_path):
    file = tmp_path / "data.h5"

    images = np.full(
        (1, 2, 2),
        ADC_MAX,
        dtype=np.uint16,
    )

    create_h5_file(file, images)

    received = {}

    def transform(image):
        received["value"] = image.clone()
        return image

    dataset = MyDataset(
        [file],
        [np.array([0])],
        transform=transform,
    )

    dataset[0]

    assert torch.allclose(
        received["value"],
        torch.ones((1, 2, 2)),
    )


def test_my_dataset_getitem_invalid_sample_index_raises(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((1, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([0])],
    )

    with pytest.raises(IndexError):
        dataset[1]


def test_my_dataset_getitem_negative_sample_index_follows_python_sequence_semantics(
    tmp_path,
):
    file = tmp_path / "data.h5"

    images = np.array(
        [
            np.zeros((2, 2), dtype=np.uint16),
            np.full((2, 2), ADC_MAX, dtype=np.uint16),
        ]
    )

    create_h5_file(file, images)

    dataset = MyDataset(
        [file],
        [np.array([0, 1])],
    )

    result = dataset[-1]

    assert torch.allclose(
        result,
        torch.ones((1, 2, 2)),
    )


def test_my_dataset_getitem_image_index_out_of_bounds_raises(tmp_path):
    file = tmp_path / "data.h5"

    create_h5_file(
        file,
        np.zeros((2, 2, 2), dtype=np.uint16),
    )

    dataset = MyDataset(
        [file],
        [np.array([5])],
    )

    with pytest.raises(IndexError):
        dataset[0]
