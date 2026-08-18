import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from training.evaluation import (
    evaluate_model,
    calculate_metrics,
    confusion_matrix,
    _validate_classification_inputs,
)


# ============================================================================
# Test models
# ============================================================================

class DummyModel(torch.nn.Module):
    """
    Small deterministic model used for testing.
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # A parameter is required because evaluate_model()
        # checks the model's device using model.parameters().
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)

        # Return deterministic logits.
        logits = torch.zeros(
            batch_size,
            self.num_classes,
            device=x.device,
        )

        logits[:, 0] = 3.0
        logits[:, 1] = 1.0
        logits[:, 2] = 0.0

        return logits


class NaNModel(torch.nn.Module):
    """
    Model returning NaN values.
    """

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.full(
            (x.size(0), 3),
            float("nan"),
            device=x.device,
        )


class InfModel(torch.nn.Module):
    """
    Model returning infinite values.
    """

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.full(
            (x.size(0), 3),
            float("inf"),
            device=x.device,
        )


class GradCheckModel(torch.nn.Module):
    """
    Model used to verify that gradients are disabled.
    """

    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(4, 3)
        self.grad_enabled = None

    def forward(self, x):
        self.grad_enabled = torch.is_grad_enabled()

        x = x.view(x.size(0), -1)

        return self.linear(x)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@pytest.fixture
def images():
    return torch.randn(10, 1, 2, 2)


@pytest.fixture
def labels():
    return torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        dtype=torch.long,
    )


@pytest.fixture
def loader(images, labels):
    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


@pytest.fixture
def model():
    return DummyModel(num_classes=3)


@pytest.fixture
def criterion():
    return torch.nn.CrossEntropyLoss()


# ============================================================================
# evaluate_model() - normal behaviour
# ============================================================================

def test_evaluate_model_returns_correct_types(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    loss, labels, predictions, probabilities = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert isinstance(loss, float)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(probabilities, torch.Tensor)


def test_evaluate_model_returns_correct_shapes(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    loss, labels, predictions, probabilities = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert labels.shape == (10,)
    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 3)


def test_evaluate_model_returns_cpu_tensors(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    _, labels, predictions, probabilities = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert labels.device.type == "cpu"
    assert predictions.device.type == "cpu"
    assert probabilities.device.type == "cpu"


def test_evaluate_model_predictions_are_valid(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    _, _, predictions, _ = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert torch.all(predictions >= 0)
    assert torch.all(predictions < 3)


def test_evaluate_model_probabilities_are_valid(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    _, _, _, probabilities = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(10),
        atol=1e-6,
    )


def test_evaluate_model_predictions_are_argmax(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    _, _, predictions, _ = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    # DummyModel always produces:
    # [3.0, 1.0, 0.0]
    # Therefore class 0 should always be predicted.
    expected = torch.zeros(
        10,
        dtype=torch.long,
    )

    assert torch.equal(predictions, expected)


def test_evaluate_model_calculates_loss(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    loss, labels, _, _ = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    # The model always returns the same logits,
    # so the expected loss can be calculated directly.
    logits = torch.tensor(
        [3.0, 1.0, 0.0]
    )

    repeated_logits = logits.repeat(
        labels.size(0),
        1,
    )

    expected_loss = criterion(
        repeated_logits,
        labels,
    ).item()

    assert loss == pytest.approx(
        expected_loss,
        rel=1e-6,
    )


def test_evaluate_model_handles_multiple_batches(
    model,
    images,
    labels,
    criterion,
    device,
):
    model.to(device)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=3,
        shuffle=False,
    )

    _, labels_out, predictions, probabilities = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert labels_out.shape == (10,)
    assert predictions.shape == (10,)
    assert probabilities.shape == (10, 3)


# ============================================================================
# evaluate_model() - model state / gradients
# ============================================================================

def test_evaluate_model_switches_model_to_eval_mode(
    model,
    loader,
    criterion,
    device,
):
    model.to(device)

    model.train()

    assert model.training is True

    evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert model.training is False


def test_evaluate_model_disables_gradients(
    loader,
    criterion,
    device,
):
    model = GradCheckModel().to(device)

    evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    assert model.grad_enabled is False


# ============================================================================
# evaluate_model() - invalid arguments
# ============================================================================

def test_evaluate_model_rejects_invalid_model(
    loader,
    criterion,
    device,
):
    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        evaluate_model(
            "not a model",
            loader,
            criterion,
            device,
        )


def test_evaluate_model_rejects_invalid_loader(
    model,
    criterion,
    device,
):
    model.to(device)

    with pytest.raises(
        TypeError,
        match="loader must be an instance of torch.utils.data.DataLoader",
    ):
        evaluate_model(
            model,
            [],
            criterion,
            device,
        )


def test_evaluate_model_rejects_invalid_criterion(
    model,
    loader,
    device,
):
    model.to(device)

    with pytest.raises(
        TypeError,
        match="criterion must be an instance of torch.nn.Module",
    ):
        evaluate_model(
            model,
            loader,
            "not a criterion",
            device,
        )


@pytest.mark.parametrize(
    "invalid_device",
    [
        "cpu",
        "cuda",
        None,
        0,
    ],
)
def test_evaluate_model_rejects_invalid_device(
    model,
    loader,
    criterion,
    invalid_device,
):
    with pytest.raises(
        TypeError,
        match="device must be an instance of torch.device",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            invalid_device,
        )


def test_evaluate_model_rejects_model_on_wrong_device(
    model,
    loader,
    criterion,
):
    # The model is on CPU.
    # We tell the function to use CUDA.
    # The function should fail before trying to move the images.
    with pytest.raises(
        ValueError,
        match="Model is on cpu, but device is cuda",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            torch.device("cuda"),
        )


# ============================================================================
# evaluate_model() - invalid batches
# ============================================================================

class NonTensorImageDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return "not a tensor", torch.tensor(0)


class NonTensorLabelDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.randn(1, 2, 2), "not a tensor"


def test_evaluate_model_rejects_non_tensor_images(
    model,
    criterion,
    device,
):
    model.to(device)

    loader = DataLoader(
        NonTensorImageDataset(),
        batch_size=1,
    )

    with pytest.raises(
        TypeError,
        match="images must be a torch.Tensor",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


def test_evaluate_model_rejects_non_tensor_labels(
    model,
    criterion,
    device,
):
    model.to(device)

    loader = DataLoader(
        NonTensorLabelDataset(),
        batch_size=1,
    )

    with pytest.raises(
        TypeError,
        match="labels must be a torch.Tensor",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


def test_evaluate_model_rejects_wrong_image_dimensions(
    model,
    criterion,
    device,
):
    model.to(device)

    # [N, H, W] instead of [N, C, H, W]
    images = torch.randn(10, 2, 2)
    labels = torch.zeros(
        10,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    with pytest.raises(
        ValueError,
        match="Expected images to have 4 dimensions",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


def test_evaluate_model_rejects_wrong_label_dimensions(
    model,
    criterion,
    device,
):
    model.to(device)

    images = torch.randn(10, 1, 2, 2)

    # [N, 1] instead of [N]
    labels = torch.zeros(
        10,
        1,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    with pytest.raises(
        ValueError,
        match="Expected labels to have shape",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


def test_evaluate_model_rejects_non_integer_labels(
    model,
    criterion,
    device,
):
    model.to(device)

    images = torch.randn(10, 1, 2, 2)

    labels = torch.randn(10)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    with pytest.raises(
        TypeError,
        match="Labels must have an integer dtype",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


# ============================================================================
# evaluate_model() - invalid model output
# ============================================================================

def test_evaluate_model_rejects_nan_outputs(
    loader,
    criterion,
    device,
):
    model = NaNModel().to(device)

    with pytest.raises(
        ValueError,
        match="Probabilities for each sample",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


def test_evaluate_model_rejects_infinite_outputs(
    loader,
    criterion,
    device,
):
    model = InfModel().to(device)

    with pytest.raises(
        ValueError,
        match="Probabilities for each sample",
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


# ============================================================================
# Empty DataLoader
# ============================================================================

def test_evaluate_model_rejects_empty_loader(
    model,
    criterion,
    device,
):
    model.to(device)

    images = torch.empty(
        (0, 1, 2, 2)
    )

    labels = torch.empty(
        (0,),
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    with pytest.raises(
        ZeroDivisionError,
    ):
        evaluate_model(
            model,
            loader,
            criterion,
            device,
        )


# ============================================================================
# _validate_classification_inputs()
# ============================================================================

@pytest.fixture
def classification_data():
    target = torch.tensor(
        [0, 1, 2, 0, 1],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [0, 1, 1, 2, 1],
        dtype=torch.long,
    )

    return target, predictions


def test_validate_classification_inputs_accepts_valid_data(
    classification_data,
):
    target, predictions = classification_data

    result = _validate_classification_inputs(
        target,
        predictions,
        num_classes=3,
    )

    assert result is None


def test_validate_rejects_non_tensor_target(
    classification_data,
):
    _, predictions = classification_data

    with pytest.raises(
        TypeError,
        match="target must be a torch.Tensor",
    ):
        _validate_classification_inputs(
            [0, 1, 2],
            predictions,
            3,
        )


def test_validate_rejects_non_tensor_predictions(
    classification_data,
):
    target, _ = classification_data

    with pytest.raises(
        TypeError,
        match="predictions must be a torch.Tensor",
    ):
        _validate_classification_inputs(
            target,
            [0, 1, 2],
            3,
        )


@pytest.mark.parametrize(
    "invalid_num_classes",
    [
        1,
        0,
        -1,
    ],
)
def test_validate_rejects_too_few_classes(
    classification_data,
    invalid_num_classes,
):
    target, predictions = classification_data

    with pytest.raises(
        ValueError,
        match="num_classes must be at least 2",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            invalid_num_classes,
        )


def test_validate_rejects_non_integer_num_classes(
    classification_data,
):
    target, predictions = classification_data

    with pytest.raises(
        TypeError,
        match="num_classes must be an integer",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3.0,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (5, 1),
        (1, 5),
        (5, 1, 1),
        (),
    ],
)
def test_validate_rejects_wrong_target_shape(
    classification_data,
    shape,
):
    _, predictions = classification_data

    target = torch.zeros(
        shape,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="target must have shape",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


def test_validate_rejects_wrong_prediction_shape(
    classification_data,
):
    target, _ = classification_data

    predictions = torch.zeros(
        (5, 1),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="predictions must have shape",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


def test_validate_rejects_different_number_of_samples():
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.bool,
    ],
)
def test_validate_rejects_non_integer_target(
    classification_data,
    dtype,
):
    _, predictions = classification_data

    target = torch.tensor(
        [0, 1, 2, 0, 1],
        dtype=dtype,
    )

    with pytest.raises(
        TypeError,
        match="target must have an integer dtype",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.bool,
    ],
)
def test_validate_rejects_non_integer_predictions(
    classification_data,
    dtype,
):
    target, _ = classification_data

    predictions = torch.tensor(
        [0, 1, 2, 0, 1],
        dtype=dtype,
    )

    with pytest.raises(
        TypeError,
        match="predictions must have an integer dtype",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


@pytest.mark.parametrize(
    "target",
    [
        torch.tensor([-1, 0, 1]),
        torch.tensor([0, 1, 3]),
    ],
)
def test_validate_rejects_invalid_target_classes(target):
    predictions = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="target contains class indices outside",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


@pytest.mark.parametrize(
    "predictions",
    [
        torch.tensor([-1, 0, 1]),
        torch.tensor([0, 1, 3]),
    ],
)
def test_validate_rejects_invalid_prediction_classes(predictions):
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="predictions contain class indices outside",
    ):
        _validate_classification_inputs(
            target,
            predictions,
            3,
        )


# ============================================================================
# calculate_metrics()
# ============================================================================

def test_calculate_metrics_returns_expected_keys():
    target = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    probabilities = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])

    metrics = calculate_metrics(
        predictions,
        probabilities,
        target,
        num_classes=3,
    )

    assert isinstance(metrics, dict)

    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
    }


def test_calculate_metrics_perfect_predictions():
    target = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    predictions = target.clone()

    probabilities = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])

    metrics = calculate_metrics(
        predictions,
        probabilities,
        target,
        num_classes=3,
    )

    assert metrics["accuracy"].item() == pytest.approx(1.0)
    assert metrics["precision"].item() == pytest.approx(1.0)
    assert metrics["recall"].item() == pytest.approx(1.0)
    assert metrics["f1"].item() == pytest.approx(1.0)
    assert metrics["auroc"].item() == pytest.approx(1.0)


def test_calculate_metrics_rejects_nan_probabilities():
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    predictions = target.clone()

    probabilities = torch.tensor([
        [0.9, 0.1, 0.0],
        [float("nan"), 0.5, 0.5],
        [0.0, 0.2, 0.8],
    ])

    with pytest.raises(
        ValueError,
        match="probabilities contain NaN or infinite values",
    ):
        calculate_metrics(
            predictions,
            probabilities,
            target,
            3,
        )


def test_calculate_metrics_rejects_infinite_probabilities():
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    predictions = target.clone()

    probabilities = torch.tensor([
        [0.9, 0.1, 0.0],
        [float("inf"), 0.0, 0.0],
        [0.0, 0.2, 0.8],
    ])

    with pytest.raises(
        ValueError,
        match="probabilities contain NaN or infinite values",
    ):
        calculate_metrics(
            predictions,
            probabilities,
            target,
            3,
        )


def test_calculate_metrics_rejects_probability_below_zero():
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    predictions = target.clone()

    probabilities = torch.tensor([
        [0.9, 0.1, 0.0],
        [-0.1, 0.5, 0.6],
        [0.0, 0.2, 0.8],
    ])

    with pytest.raises(
        ValueError,
        match="probabilities must be between 0 and 1",
    ):
        calculate_metrics(
            predictions,
            probabilities,
            target,
            3,
        )


def test_calculate_metrics_rejects_probability_above_one():
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    predictions = target.clone()

    probabilities = torch.tensor([
        [0.9, 0.1, 0.0],
        [0.1, 1.1, -0.2],
        [0.0, 0.2, 0.8],
    ])

    with pytest.raises(
        ValueError,
        match="probabilities must be between 0 and 1",
    ):
        calculate_metrics(
            predictions,
            probabilities,
            target,
            3,
        )


# ============================================================================
# confusion_matrix()
# ============================================================================

def test_confusion_matrix_returns_expected_matrix():
    target = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [0, 1, 0, 1, 1, 2, 2, 0, 2],
        dtype=torch.long,
    )

    matrix = confusion_matrix(
        target,
        predictions,
        num_classes=3,
    )

    expected = torch.tensor([
        [2, 1, 0],
        [0, 2, 1],
        [1, 0, 2],
    ])

    assert torch.equal(matrix, expected)


def test_confusion_matrix_has_correct_shape():
    target = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [0, 1, 2, 1, 2, 0],
        dtype=torch.long,
    )

    matrix = confusion_matrix(
        target,
        predictions,
        num_classes=3,
    )

    assert matrix.shape == (3, 3)


def test_confusion_matrix_rejects_invalid_inputs():
    target = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [0, 1, 3],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="predictions contain class indices outside",
    ):
        confusion_matrix(
            target,
            predictions,
            num_classes=3,
        )
