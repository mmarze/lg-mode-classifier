import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.predict import get_predictions


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class SimpleModel(torch.nn.Module):
    """Small deterministic model for testing."""

    def __init__(self, in_features=4, num_classes=3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Flatten everything except batch dimension
        x = x.view(x.size(0), -1)
        return self.linear(x)


class FixedOutputModel(torch.nn.Module):
    """Model returning predefined logits."""

    def __init__(self, outputs):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))
        self.outputs = outputs

    def forward(self, x):
        return self.outputs[:x.size(0)].to(x.device)


class NaNModel(torch.nn.Module):
    """Model returning NaN values."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.full(
            (x.size(0), 3),
            float("nan"),
            device=x.device,
        )


class GradientCheckingModel(torch.nn.Module):
    """Model that verifies gradients are disabled."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.grad_enabled_during_forward = None

    def forward(self, x):
        self.grad_enabled_during_forward = torch.is_grad_enabled()

        x = x.view(x.size(0), -1)

        return self.linear(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def images():
    return torch.randn(10, 1, 2, 2)


@pytest.fixture
def labels():
    return torch.randint(0, 3, (10,))


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
    return SimpleModel()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_get_predictions_returns_class_predictions(
    model,
    loader,
    device,
):
    model.to(device)

    result = get_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10,)
    assert result.dtype == torch.int64

    assert torch.all(result >= 0)
    assert torch.all(result < 3)


def test_get_predictions_returns_probabilities(
    model,
    loader,
    device,
):
    model.to(device)

    result = get_predictions(
        model=model,
        loader=loader,
        device=device,
        prob=True,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 3)

    # Probabilities should be in [0, 1]
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

    # Probabilities for each sample should sum to 1
    assert torch.allclose(
        result.sum(dim=1),
        torch.ones(10),
        atol=1e-6,
    )


def test_get_predictions_handles_multiple_batches(
    model,
    images,
    labels,
    device,
):
    model.to(device)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=3,
        shuffle=False,
    )

    result = get_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    # 10 samples should result in 10 predictions,
    # regardless of the batch size.
    assert result.shape == (10,)


def test_get_predictions_ignores_labels(
    model,
    images,
    device,
):
    # Labels don't have to contain meaningful values: get_predictions()
    # deliberately ignores them.
    labels = torch.full((10,), 999)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    model.to(device)

    result = get_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    assert result.shape == (10,)


# ---------------------------------------------------------------------------
# Correctness of predictions
# ---------------------------------------------------------------------------

def test_get_predictions_returns_argmax_of_logits(device):
    images = torch.randn(4, 1, 2, 2)
    labels = torch.zeros(4, dtype=torch.long)

    logits = torch.tensor([
        [10.0, 1.0, 0.0],
        [0.0, 20.0, 1.0],
        [1.0, 2.0, 30.0],
        [5.0, 10.0, 1.0],
    ])

    model = FixedOutputModel(logits).to(device)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    result = get_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    expected = torch.tensor([0, 1, 2, 1])

    assert torch.equal(result, expected)


def test_get_predictions_returns_correct_softmax_probabilities(device):
    images = torch.randn(2, 1, 2, 2)
    labels = torch.zeros(2, dtype=torch.long)

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
    ])

    model = FixedOutputModel(logits).to(device)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    result = get_predictions(
        model=model,
        loader=loader,
        device=device,
        prob=True,
    )

    expected = torch.softmax(logits, dim=1)

    assert torch.allclose(
        result,
        expected,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Model state / gradients
# ---------------------------------------------------------------------------

def test_model_is_switched_to_evaluation_mode(
    model,
    loader,
    device,
):
    model.to(device)

    # Explicitly put model in training mode
    model.train()

    assert model.training is True

    get_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    assert model.training is False


def test_gradients_are_disabled(
    loader,
    device,
):
    model = GradientCheckingModel().to(device)

    get_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    assert model.grad_enabled_during_forward is False


# ---------------------------------------------------------------------------
# Invalid function arguments
# ---------------------------------------------------------------------------

def test_invalid_model_type(loader, device):
    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        get_predictions(
            model="not a model",
            loader=loader,
            device=device,
        )


def test_invalid_loader_type(model, device):
    model.to(device)

    with pytest.raises(
        TypeError,
        match="loader must be an instance of torch.utils.data.DataLoader",
    ):
        get_predictions(
            model=model,
            loader=[],
            device=device,
        )


def test_invalid_device_type(model, loader):
    with pytest.raises(
        TypeError,
        match="device must be an instance of torch.device",
    ):
        get_predictions(
            model=model,
            loader=loader,
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_prob",
    [
        0,
        1,
        None,
        "True",
        "False",
        [],
        {},
    ],
)
def test_invalid_prob_type(
    model,
    loader,
    device,
    invalid_prob,
):
    model.to(device)

    with pytest.raises(
        TypeError,
        match="prob must be a boolean",
    ):
        get_predictions(
            model=model,
            loader=loader,
            device=device,
            prob=invalid_prob,
        )


# ---------------------------------------------------------------------------
# Device validation
# ---------------------------------------------------------------------------

def test_model_and_device_must_match(
    model,
    loader,
):
    # Model is on CPU, but function is explicitly told to use CUDA.
    # This test does not require CUDA to actually be available because
    # the function should fail before trying to move the images.
    with pytest.raises(
        ValueError,
        match="Model is on cpu, but device is cuda",
    ):
        get_predictions(
            model=model,
            loader=loader,
            device=torch.device("cuda"),
        )


# ---------------------------------------------------------------------------
# Empty DataLoader
# ---------------------------------------------------------------------------

def test_empty_dataloader_raises_error(
    model,
    device,
):
    model.to(device)

    images = torch.empty((0, 1, 2, 2))
    labels = torch.empty((0,), dtype=torch.long)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    with pytest.raises(
        ValueError,
        match="DataLoader is empty",
    ):
        get_predictions(
            model=model,
            loader=loader,
            device=device,
        )


# ---------------------------------------------------------------------------
# Invalid input shape
# ---------------------------------------------------------------------------

def test_images_must_have_four_dimensions(
    model,
    device,
):
    model.to(device)

    # [N, H, W] instead of [N, C, H, W]
    images = torch.randn(10, 2, 2)
    labels = torch.zeros(10, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    with pytest.raises(
        ValueError,
        match="Expected images to have 4 dimensions",
    ):
        get_predictions(
            model=model,
            loader=loader,
            device=device,
        )


# ---------------------------------------------------------------------------
# Invalid model output
# ---------------------------------------------------------------------------

def test_nan_model_output_raises_error(
    loader,
    device,
):
    model = NaNModel().to(device)

    with pytest.raises(
        ValueError,
        match="Model output contains NaN or infinite values",
    ):
        get_predictions(
            model=model,
            loader=loader,
            device=device,
        )
