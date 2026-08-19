import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.train import train_one_epoch, train_model
import training.train as train_module


# ==========================================================
# Fixtures / helpers
# ==========================================================

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def mps_device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return torch.device("mps")


@pytest.fixture
def model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )


@pytest.fixture
def criterion():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def loader():
    images = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [1.0, 1.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ]
    )

    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    dataset = TensorDataset(images, labels)

    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def empty_loader():
    dataset = TensorDataset(
        torch.empty((0, 1, 2, 2)),
        torch.empty((0,), dtype=torch.long),
    )

    return DataLoader(dataset, batch_size=2)


# ==========================================================
# Test train_one_epoch
# ==========================================================


def test_train_one_epoch_returns_float(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    result = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
    )

    assert isinstance(result, float)


def test_train_one_epoch_returns_finite_loss(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    result = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
    )

    assert torch.isfinite(torch.tensor(result))


def test_train_one_epoch_updates_model_parameters(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
    )

    for name, parameter in model.named_parameters():
        assert not torch.equal(before[name], parameter)


# ----------------------------------------------------------
# train_one_epoch - argument types
# ----------------------------------------------------------

@pytest.mark.parametrize(
    "bad_model",
    [
        None,
        "model",
        1,
        1.0,
    ],
)
def test_train_one_epoch_model_type(
    bad_model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_one_epoch(
            bad_model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "loader",
    [
        None,
        [],
        (),
        "loader",
    ],
)
def test_train_one_epoch_loader_type(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "criterion",
    [
        None,
        "criterion",
        1,
        1.0,
    ],
)
def test_train_one_epoch_criterion_type(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "optimizer",
    [
        None,
        "optimizer",
        1,
        1.0,
    ],
)
def test_train_one_epoch_optimizer_type(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "device",
    [
        None,
        "cpu",
        "cuda",
        1,
    ],
)
def test_train_one_epoch_device_type(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


# ----------------------------------------------------------
# train_one_epoch - values / model
# ----------------------------------------------------------


def test_train_one_epoch_rejects_empty_loader(
    model,
    empty_loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            empty_loader,
            criterion,
            optimizer,
            device,
        )


def test_train_one_epoch_rejects_parameterless_model(
    loader,
    criterion,
    optimizer,
    device,
):
    model = torch.nn.Identity()

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


def test_train_one_epoch_rejects_model_on_wrong_device(
    loader,
    criterion,
    optimizer,
):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )

    wrong_device = torch.device("cuda")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = model.to("cpu")

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            wrong_device,
        )


def test_train_one_epoch_mps(
    model,
    loader,
    criterion,
    mps_device,
):
    model = model.to(mps_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        mps_device,
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


# ----------------------------------------------------------
# train_one_epoch - batch validation
# ----------------------------------------------------------


class NonTensorDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return "image", torch.tensor(0, dtype=torch.long)


def test_train_one_epoch_rejects_non_tensor_images(
    model,
    criterion,
    optimizer,
    device,
):
    loader = DataLoader(NonTensorDataset(), batch_size=1)

    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


class NonTensorLabelDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.zeros((1, 2, 2)), "label"


def test_train_one_epoch_rejects_non_tensor_labels(
    model,
    criterion,
    optimizer,
    device,
):
    loader = DataLoader(
        NonTensorLabelDataset(),
        batch_size=1,
    )

    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((2, 2, 2)),
        torch.zeros((2, 1, 2, 2, 1)),
        torch.zeros((2,)),
    ],
)
def test_train_one_epoch_rejects_invalid_image_shape(
    images,
    criterion,
    optimizer,
    device,
):
    labels = torch.tensor([0, 1], dtype=torch.long)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[[0]], [[1]]], dtype=torch.long),
        torch.tensor([[[[0]]], [[[1]]]], dtype=torch.long),
    ],
)
def test_train_one_epoch_rejects_invalid_label_shape(
    labels,
    criterion,
    optimizer,
    device,
):
    images = torch.zeros((2, 1, 2, 2))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0.0, 1.0]),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int16),
        torch.tensor([0, 1], dtype=torch.int8),
    ],
)
def test_train_one_epoch_rejects_non_long_labels(
    labels,
    criterion,
    optimizer,
    device,
):
    images = torch.zeros((2, 1, 2, 2))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )

    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


# ----------------------------------------------------------
# train_one_epoch - model output
# ----------------------------------------------------------


class WrongShapeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.zeros((x.size(0),))


class WrongOutputTypeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return [0, 1]


def test_train_one_epoch_rejects_non_tensor_model_output(
    loader,
    criterion,
    optimizer,
    device,
):
    model = WrongOutputTypeModel()

    with pytest.raises(TypeError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


def test_train_one_epoch_rejects_invalid_model_output_shape(
    loader,
    criterion,
    optimizer,
    device,
):
    model = WrongShapeModel()

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


def test_train_one_epoch_rejects_non_finite_loss(
    loader,
    optimizer,
    device,
):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )

    class NaNLoss(torch.nn.Module):
        def forward(self, outputs, labels):
            return torch.tensor(float("nan"))

    criterion = NaNLoss()

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
        )


# ==========================================================
# Test train_model
# ==========================================================


def test_train_model_returns_model_and_history(
    model,
    loader,
    criterion,
    optimizer,
    device,
    monkeypatch,
):
    def fake_evaluate_model(
        model,
        loader,
        criterion,
        device,
    ):
        return 0.5, None, None, None

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        fake_evaluate_model,
    )

    monkeypatch.setattr(
    train_module,
    "calculate_metrics",
    lambda pred, prob, lab, num_classes: {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
    },
)

    result_model, history = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        num_epochs=2,
        num_classes=2,
        min_delta=0.0,
    )

    assert isinstance(result_model, torch.nn.Module)

    assert isinstance(history, dict)

    assert "train_loss" in history
    assert "validation_loss" in history

    assert len(history["train_loss"]) == 2
    assert len(history["validation_loss"]) == 2


def test_train_model_mps(
    model,
    loader,
    criterion,
    mps_device,
    monkeypatch,
):
    model = model.to(mps_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda model, loader, criterion, device: (
            0.5,
            None,
            None,
            None,
        ),
    )

    monkeypatch.setattr(
        train_module,
        "calculate_metrics",
        lambda pred, prob, lab, num_classes: {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
        },
    )

    result_model, history = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        mps_device,
        num_epochs=1,
        num_classes=2,
        min_delta=0.0,
    )

    assert next(result_model.parameters()).device.type == "mps"
    assert len(history["train_loss"]) == 1
    assert len(history["validation_loss"]) == 1


# ----------------------------------------------------------
# train_model - argument types
# ----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        "model",
        1,
        1.0,
    ],
)
def test_train_model_model_type(
    value,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            value,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        (),
        "loader",
    ],
)
def test_train_model_train_loader_type(
    value,
    model,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            value,
            value if isinstance(value, DataLoader) else DataLoader(
                TensorDataset(
                    torch.zeros((1, 1, 2, 2)),
                    torch.tensor([0]),
                )
            ),
            criterion,
            optimizer,
            device,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        (),
        "loader",
    ],
)
def test_train_model_validation_loader_type(
    value,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            value,
            criterion,
            optimizer,
            device,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "criterion",
        1,
        1.0,
    ],
)
def test_train_model_criterion_type(
    value,
    model,
    loader,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            value,
            optimizer,
            device,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "optimizer",
        1,
        1.0,
    ],
)
def test_train_model_optimizer_type(
    value,
    model,
    loader,
    criterion,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            value,
            device,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "cpu",
        "cuda",
        1,
    ],
)
def test_train_model_device_type(
    value,
    model,
    loader,
    criterion,
    optimizer,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            value,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.0,
        "1",
        [],
        (),
        True,
        False,
    ],
)
def test_train_model_num_epochs_type(
    value,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            value,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.0,
        "5",
        [],
        (),
        True,
        False,
    ],
)
def test_train_model_patience_type(
    value,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            patience=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.0,
        "2",
        [],
        (),
        True,
        False,
    ],
)
def test_train_model_num_classes_type(
    value,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            value,
            0.0,
        )


@pytest.mark.parametrize(
    "num_classes",
    [
        0,
        -1,
        -10,
    ],
)
def test_train_model_rejects_invalid_num_classes(
    num_classes,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            num_classes,
            0.0,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "0.1",
        [],
        (),
        True,
        False,
    ],
)
def test_train_model_min_delta_type(
    value,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(TypeError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            2,
            value,
        )     


@pytest.mark.parametrize(
    "min_delta",
    [
        -0.0001,
        -1.0,
        -10.0,
    ],
)
def test_train_model_rejects_negative_min_delta(
    min_delta,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            2,
            min_delta,
        )


# ----------------------------------------------------------
# train_model - values
# ----------------------------------------------------------


@pytest.mark.parametrize(
    "num_epochs",
    [
        0,
        -1,
        -10,
    ],
)
def test_train_model_rejects_invalid_num_epochs(
    num_epochs,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            num_epochs,
             num_classes=2,
            min_delta=0.1,
            patience=1,
        )


@pytest.mark.parametrize(
    "patience",
    [
        -1,
        -10,
    ],
)
def test_train_model_rejects_negative_patience(
    patience,
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            num_classes=2,
            min_delta=0.1,
            patience=patience,
        )


def test_train_model_rejects_empty_train_loader(
    model,
    empty_loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_model(
            model,
            empty_loader,
            empty_loader,
            criterion,
            optimizer,
            device,
            1,
            num_classes=2,
            min_delta=0.1,
            patience=1,
        )


def test_train_model_rejects_empty_validation_loader(
    model,
    loader,
    empty_loader,
    criterion,
    optimizer,
    device,
):
    with pytest.raises(ValueError):
        train_model(
            model,
            loader,
            empty_loader,
            criterion,
            optimizer,
            device,
            1,
            num_classes=2,
            min_delta=0.1,
            patience=1,
        )


def test_train_model_rejects_parameterless_model(
    loader,
    criterion,
    optimizer,
    device,
):
    model = torch.nn.Identity()

    with pytest.raises(ValueError):
        train_model(
            model,
            loader,
            loader,
            criterion,
            optimizer,
            device,
            1,
            num_classes=2,
            min_delta=0.1,
            patience=1,
        )


# ----------------------------------------------------------
# train_model - training behavior
# ----------------------------------------------------------


def test_train_model_history_has_one_entry_per_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    monkeypatch,
):
    def fake_evaluate_model(
        model,
        loader,
        criterion,
        device,
    ):
        return 1.0, None, None, None

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        fake_evaluate_model,
    )
    monkeypatch.setattr(
    train_module,
    "calculate_metrics",
    lambda pred, prob, lab, num_classes: {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
    },
    )

    _, history = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        num_epochs=3,
        num_classes=2,
        min_delta=0.1,
        patience=10,
    )

    assert len(history["train_loss"]) == 3
    assert len(history["validation_loss"]) == 3


def test_train_model_stops_after_patience(
    model,
    loader,
    criterion,
    optimizer,
    device,
    monkeypatch,
):
    def fake_evaluate_model(
        model,
        loader,
        criterion,
        device,
    ):
        return 1.0, None, None, None

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        fake_evaluate_model,
    )

    monkeypatch.setattr(
        train_module,
        "calculate_metrics",
        lambda pred, prob, lab, num_classes: {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
        },
    )

    _, history = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        num_epochs=10,
        num_classes=2,
        min_delta=0.1,
        patience=2,
    )

    # First epoch establishes the best value.
    # Next two epochs do not improve.
    assert len(history["validation_loss"]) == 3


def test_train_model_respects_min_delta(
    model,
    loader,
    criterion,
    optimizer,
    device,
    monkeypatch,
):
    validation_losses = iter([1.0, 0.95, 0.90])

    def fake_evaluate_model(
        model,
        loader,
        criterion,
        device,
    ):
        return next(validation_losses), None, None, None

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        fake_evaluate_model,
    )

    monkeypatch.setattr(
        train_module,
        "calculate_metrics",
        lambda pred, prob, lab, num_classes: {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
        },
    )

    _, history = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        num_epochs=10,
        num_classes=2,
        min_delta=0.1,
        patience=1,
    )

    assert len(history["validation_loss"]) == 2


def test_train_model_does_not_stop_when_validation_improves(
    model,
    loader,
    criterion,
    optimizer,
    device,
    monkeypatch,
):
    validation_losses = iter([1.0, 0.8, 0.6])

    def fake_evaluate_model(
        model,
        loader,
        criterion,
        device,
    ):
        return next(validation_losses), None, None, None

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        fake_evaluate_model,
    )

    monkeypatch.setattr(
        train_module,
        "calculate_metrics",
        lambda pred, prob, lab, num_classes: {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
        },
    )

    _, history = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        num_epochs=3,
        patience=1,
        num_classes=2,
        min_delta=0.0,
    )

    assert len(history["validation_loss"]) == 3


def test_train_model_restores_best_model(
    model,
    loader,
    criterion,
    optimizer,
    device,
    monkeypatch,
):
    validation_losses = iter([0.5, 1.0])

    def fake_evaluate_model(
        model,
        loader,
        criterion,
        device,
    ):
        return next(validation_losses), None, None, None

    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        fake_evaluate_model,
    )

    monkeypatch.setattr(
        train_module,
        "calculate_metrics",
        lambda pred, prob, lab, num_classes: {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
        },
    )

    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result_model, _ = train_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        device,
        num_epochs=2,
        patience=10,
        num_classes=2,
        min_delta=0.0,
    )
    
    for name, parameter in result_model.named_parameters():
        assert torch.isfinite(parameter).all()
        assert parameter.shape == initial_state[name].shape
