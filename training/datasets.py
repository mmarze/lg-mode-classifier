import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import h5py
import numpy as np
from pathlib import Path


SEED = None
ADC_BITS = 12
ADC_MAX = 2**ADC_BITS - 1


# ==========================================================
# Generate indices
# ==========================================================

def get_indices(ratios: tuple, N=10000):
    """
    Determine indices for train, test, and validation datasets.

    Parameters
    ----------
    ratios : tuple of floats
        The tuple with split ratio: (percentage_train, percentage_test).
        The percentage for validation dataset is caluclated based on _percentage_train_
        and _percentage_test_.
    N : int
        The number of elements in the dataset.
    
    Returns
    -------
    indices_train: np.ndarray
        Indices for train dataset.
    indices_test: np.ndarray
        Indices for test dataset.
    indices_validate:  np.ndarray
        Indices for validation dataset.

    Raises
    ------
    TypeError
        If ratios is not a tuple of floats.
        If N is not an integer.
    ValueError
        If percentage_train <= 0 or percentage_test <= 0.
        If percentage_train > 1.
        If percentage_test >= 1.
        If percentage_train + percentage_test > 1.
        If N <= 0.
    """
    # ---------- Type checking ----------
    if not isinstance(ratios, tuple):
        raise TypeError(
            f"Split ratios must be a tuple, got {type(ratios).__name__}."
        )

    for elem in ratios:
        if not isinstance(elem, (float, np.floating)):
            raise TypeError(
                f"Split ratio must be a float, got {type(elem).__name__}."
            )

    if not isinstance(N, (int, np.integer)):
        raise TypeError(
            f"Number of elements in dataset N must be an integer {type(N).__name__}."
        )

    # ---------- Value checking ----------
    if not np.all(np.isfinite(ratios)):
        raise ValueError("Split ratio values must be finite.")

    if len(ratios) != 2:
        raise ValueError(
            "ratios must contain exactly two values: "
            "(train_ratio, test_ratio)."
        )

    if not np.isfinite(N):
        raise ValueError("Number of elemnets in dataset must be finite.")

    train, test = ratios

    if train <= 0:
        raise ValueError("Split ratio for train dataset must be greater than 0.")

    if train > 1:
        raise ValueError("Split ratio for train dataset must be less than or equal to 1.")

    if test <= 0: 
        raise ValueError("Split ratio for test dataset must be greater than 0.")

    if test >= 1:
        raise ValueError("Split ratio for test dataset must be less than 1.")

    if test + train > 1:
        raise ValueError("The sum of split ratios must be less than or equal to 1.")

    if N <= 0:
         raise ValueError("Number of elements in dataset must be greater than 0.")
            
    # ---------- Get indices ----------

    N_test = int(test * N)
    N_train = int(train * N)

    rng = np.random.default_rng(seed=SEED)
    indices = rng.permutation(N)

    indices_test = indices[:N_test]
    indices_train = indices[N_test:N_test + N_train]
    indices_validate = indices[N_test + N_train:]

    return (indices_train, indices_test, indices_validate)


# ==========================================================
# Generate labels
# ==========================================================

def get_class_labels(indices_train: list, indices_test: list, indices_validate: list):
    """
    Determine class labels for train, test, and validation datasets.

    Parameters
    ----------
    indices_train : list
        A list np.ndarrays of indices of images belonging to the training dataset.
     indices_test : list
        A list np.ndarrays of indices of images belonging to the testing dataset.
     indices_validate : list
        A list np.ndarrays of indices of images belonging to the validating dataset.

    Returns
    -------
    indices_train: list
        A list np.ndarrays of class indicators for training dataset.
    indices_test: list
        A list np.ndarrays of class indicators for testing dataset.
    indices_validate:  list
        A list np.ndarrays of class indicators for validating dataset.

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If len(indices_train) != len(indices_test).
    """
    # ---------- Type and values checking ----------
    if not isinstance(indices_train, list):
        raise TypeError(
            f"indices_train must be a list, got {type(indices_train).__name__}."
            )

    for list_indices in indices_train:
        if not isinstance(list_indices, np.ndarray):
            raise TypeError(
                f"Training dataset: each element of indices must be an np.ndarray, "
                f"got {type(list_indices).__name__}."
                )
        
        for index in list_indices:
            if not isinstance(index, (int, np.integer)):
                raise TypeError(
                    f"Training dataset: each image index must be an integer, "
                    f"got {type(index).__name__}."
                )

            if index < 0:
                raise ValueError(
                    f"Training dataset: image index cannot be negative, got {index}."
                )

    if not isinstance(indices_test, list):
        raise TypeError(
            f"indices_test must be a list, got {type(indices_test).__name__}."
            )

    for list_indices in indices_test:
        if not isinstance(list_indices, np.ndarray):
            raise TypeError(
                f"Testing dataset: each element of indices must be an np.ndarray, "
                f"got {type(list_indices).__name__}."
                )
        
        for index in list_indices:
            if not isinstance(index, (int, np.integer)):
                raise TypeError(
                    f"Testing dataset: each image index must be an integer, "
                    f"got {type(index).__name__}."
                )

            if index < 0:
                raise ValueError(
                    f"Testing dataset: image index cannot be negative, got {index}."
                )

    if not isinstance(indices_validate, list):
        raise TypeError(
            f"indices_validate must be a list, got {type(indices_validate).__name__}."
            )


    for list_indices in indices_validate:
        if not isinstance(list_indices, np.ndarray):
            raise TypeError(
                f"Validation dataset: each element of indices must be an np.ndarray, "
                f"got {type(list_indices).__name__}."
                )
        
        for index in list_indices:
            if not isinstance(index, (int, np.integer)):
                raise TypeError(
                    f"Validation dataset: each image index must be an integer, "
                    f"got {type(index).__name__}."
                )

            if index < 0:
                raise ValueError(
                    f"Validation dataset: image index cannot be negative, got {index}."
                )    

    len_train = len(indices_train)
    len_test = len(indices_test)
    len_validate = len(indices_validate)

    if len_train != len_test:
        raise ValueError(
            f"The length of training dataset and testing dataset must be equal."
        )

    if len_validate != 0:
        if len_train != len_validate:
           raise ValueError(
                f"The length of training dataset and testing dataset must be equal."
            ) 
            
    # ---------- Get classes ----------

    len2_train = [len(elem) for elem in indices_train]
    len2_test = [len(elem) for elem in indices_test]
    len2_val = [len(elem) for elem in indices_validate]

    classes_train, classes_test, classes_validate = [], [], []

    for i, elem in enumerate(len2_train):
        classes_train.append(i * np.ones(elem))

    for i, elem in enumerate(len2_test):
        classes_test.append(i * np.ones(elem))

    if len2_val != 0:
        for i, elem in enumerate(len2_val):
            classes_validate.append(i * np.ones(elem))

    return (classes_train, classes_test, classes_validate)


# ==========================================================
# Normalization
# ==========================================================

def calculate_transformation_params(files, indices, n_subset=256):
    """
    Calculates the mean value and the standard deviation for data normalziation before model training.

    Parameters
    ----------
    files : list of filepaths
        A list with paths to the files with the data.
    indices: list of integers
        A list np.ndarrays of indices of images belonging to the dataset.
    n_subset: int
        Number of elements read to the memory in one batch. Default: 256.
    
    Returns
    -------
    mean: float
        The mean value of the dataset.
    std: float
        The standard deviation for test dataset.

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If input values are invalid or no images are selected.
    """
        # ---------- Type and values checking ----------
    if not isinstance(files, list):
        raise TypeError(
            f"files must be a list, got {type(files).__name__}."
        )

    for elem in files:
        if not isinstance(elem, (str, Path)):
            raise TypeError(
                f"Each filepath must be a string or pathlib.Path, "
                f"got {type(elem).__name__}."
            )

        if isinstance(elem, str) and not elem:
            raise ValueError("Filepath cannot be empty.")

    if not isinstance(indices, list):
            raise TypeError(f"indices must be a list, got {type(indices).__name__}.")

    if len(files) != len(indices):
        raise ValueError(f"files and indices must have the same length, "
                            f"got {len(files)} and {len(indices)}.")
    
    for list_indices in indices:
        if not isinstance(list_indices, np.ndarray):
            raise TypeError(
                f"Each element of indices must be an np.ndarray, "
                f"got {type(list_indices).__name__}."
                )
        
        for index in list_indices:
            if not isinstance(index, (int, np.integer)):
                raise TypeError(
                    f"Each image index must be an integer, "
                    f"got {type(index).__name__}."
                )

            if index < 0:
                raise ValueError(
                    f"Image index cannot be negative, got {index}."
                )

    if not isinstance(n_subset, (int, np.integer)):
        raise TypeError(
            f"n_subset must be an integer {type(n_subset).__name__}."
        )

    if n_subset <= 0:
         raise ValueError("n_subset must be greater than 0.")
            
    # ---------- Caluclate statistics ----------

    total_sum = 0.0
    total_squared_sum = 0.0
    total_count = 0

    for file, l_indices in zip(files, indices):

        if len(l_indices) == 0:
            continue

        # Check for unique indices.
        if len(l_indices) != len(np.unique(l_indices)):
            raise ValueError(
                f"Indices must be unique!"
            )

        with h5py.File(file, "r") as f:

            if "images" not in f:
                raise KeyError(f"Dataset 'images' not found in {file}.")

            dset = f["images"]

            # Check that all requested indices exist.
            max_index = max(l_indices)

            if max_index >= dset.shape[0]:
                raise IndexError(
                    f"Image index {max_index} is out of bounds for "
                    f"{file} with {dset.shape[0]} images."
                )

            # Process selected images in batches.
            for start in range(0, len(l_indices), n_subset):
                batch_indices = sorted(l_indices[start:start + n_subset])

                batch = dset[batch_indices]

                # Convert to float64 and normalize to [0, 1].
                batch = batch.astype(np.float32) / ADC_MAX

                total_sum += np.sum(batch, dtype=np.float64)
                total_squared_sum += np.sum(
                    batch ** 2,
                    dtype=np.float64,
                )
                total_count += batch.size

    if total_count == 0:
        raise ValueError("No images were selected.")

    # ---------- Calculate mean and standard deviation ----------

    mean = total_sum / total_count

    variance = total_squared_sum / total_count - mean ** 2
    # Protect against a tiny negative value caused by floating-point rounding.
    variance = max(variance, 0.0)

    std = np.sqrt(variance)

    return mean, std


# ==========================================================
# class MyDataset
# ==========================================================

class MyDataset(Dataset):
    """
    PyTorch Dataset for images stored in HDF5 files.

    The dataset can combine images from multiple HDF5 files. Each image
    is identified by a pair consisting of a file index and an image index
    within that file. HDF5 files are opened lazily when an image from a
    given file is requested.

    Parameters
    ----------
    files : list of str or pathlib.Path
        List of paths to HDF5 files. Each file must contain an ``"images"``
        dataset.

    indices : list of np.ndarray
        Lists of image indices corresponding to each file in ``files``.
        ``indices[i]`` contains the indices of images to use from
        ``files[i]``.

    transform : callable, optional
        Optional transformation applied to each image after it has been
        converted to a PyTorch tensor and normalized to the range [0, 1].
        The transformation should accept a tensor with shape
        ``(1, H, W)`` and return a transformed tensor.

    Attributes
    ----------
    files : list of pathlib.Path
        Paths to the HDF5 files.

    samples : list of tuple
        Flattened list of ``(file_id, image_id)`` pairs identifying all
        samples in the dataset.

    transform : callable or None
        Transformation applied to each image.

    Notes
    -----
    Images are loaded on demand in ``__getitem__`` rather than being
    loaded into memory during initialization. This allows the dataset
    to handle large collections of images without requiring enough RAM
    to store the complete dataset.

    HDF5 file handles are opened lazily. This is also suitable for use
    with PyTorch's ``DataLoader`` when multiple workers are used, since
    each worker can open its own HDF5 file handles.

    Images are assumed to contain unsigned integer values produced by
    an ADC with the maximum value defined by ``ADC_MAX``. Before any
    optional transformation is applied, images are converted to
    floating-point tensors and scaled by ``ADC_MAX`` to approximately
    the range [0, 1].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple ``(image, label)`` where ``image`` has shape
        ``(1, H, W)`` and dtype ``torch.float32``, and ``label``
        is a scalar tensor with dtype ``torch.long`` representing
        the file/class index.

    Raises
    ------
    TypeError
        If ``files`` or ``indices`` is not a list, if an element of
        ``files`` is not a string or ``pathlib.Path``, if an element of
        ``indices`` is not a NumPy array, or if an image index is not an
        integer.

    ValueError
        If ``files`` and ``indices`` have different lengths, if a filepath
        is empty, or if an image index is negative.

    FileNotFoundError
        If any path in ``files`` does not exist.

    KeyError
        If an HDF5 file does not contain an ``"images"`` dataset.
    """

    def __init__(self, files, indices, transform=None):

        # ---------- Type checking ----------

        if not isinstance(files, list):
            raise TypeError(
                f"files must be a list, got {type(files).__name__}."
            )

        if not isinstance(indices, list):
            raise TypeError(
                f"indices must be a list, got {type(indices).__name__}."
            )

        if len(files) != len(indices):
            raise ValueError(
                f"files and indices must have the same length, "
                f"got {len(files)} and {len(indices)}."
            )

        for file in files:
            if not isinstance(file, (str, Path)):
                raise TypeError(
                    f"Each file must be a string or pathlib.Path, "
                    f"got {type(file).__name__}."
                )

            if isinstance(file, str) and not file:
                raise ValueError("Filepath cannot be empty.")

            if not Path(file).exists():
                raise FileNotFoundError(
                    f"HDF5 file does not exist: {file}"
                )

        for file_indices in indices:
            if not isinstance(file_indices, np.ndarray):
                raise TypeError(
                    f"Each element of indices must be an np.ndarray, "
                    f"got {type(file_indices).__name__}."
                )

            for image_id in file_indices:
                if not isinstance(image_id, (int, np.integer)):
                    raise TypeError(
                        f"Each image index must be an integer, "
                        f"got {type(image_id).__name__}."
                    )

                if image_id < 0:
                    raise ValueError(
                        f"Image index cannot be negative, got {image_id}."
                    )

        # ---------- Store data ----------

        self.files = [Path(file) for file in files]
        self.indices = indices
        self.transform = transform

        # Flatten (file, image_index) pairs into one list.
        self.samples = [
            (file_id, int(image_id))
            for file_id, file_indices in enumerate(indices)
            for image_id in file_indices
        ]

        if not self.samples:
            raise ValueError("No images selected.")

        self._datasets = {}
        self._h5_files = {}


    def __len__(self):
        return len(self.samples)


    def _get_dataset(self, file_id):

        if file_id not in self._datasets:
            h5_file = h5py.File(self.files[file_id], "r")

            if "images" not in h5_file:
                h5_file.close()
                raise KeyError(
                    f"Dataset 'images' not found in {self.files[file_id]}."
                )

            self._h5_files[file_id] = h5_file
            self._datasets[file_id] = h5_file["images"]

        return self._datasets[file_id]


    def __getitem__(self, idx):
        file_id, image_id = self.samples[idx]

        dset = self._get_dataset(file_id)
        image = dset[image_id]

        image = torch.from_numpy(image).float()
        image = image / ADC_MAX

        # Add channel dimension:
        # (H, W) -> (1, H, W)
        image = image.unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            file_id,
            dtype=torch.long,
        )

        return image, label.to(dtype=torch.long)

    def close(self):
        if self._h5_files is not None:
            for file in self._h5_files.values():
                file.close()

        self._h5_files = {}
        self._datasets = {}

# ==========================================================
# Worker_init_fn
# ==========================================================

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        return

    dataset = worker_info.dataset

    dataset._datasets = {}
    dataset._h5_files = {}
