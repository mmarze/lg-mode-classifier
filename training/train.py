import copy
import time

import torch
from torch.utils.data import DataLoader

from training.evaluation import evaluate_model, calculate_metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to train.
    loader : DataLoader
        DataLoader providing training images and labels.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which the model and data are located.

    Returns
    -------
    float
        Average training loss for the epoch.

    Raises
    ------
    TypeError
        If ``model``, ``loader``, ``criterion``, ``optimizer``, or
        ``device`` has an invalid type. Also raised if images or labels
        provided by the loader are not PyTorch tensors, if labels do not
        have dtype ``torch.long``, or if the model output is not a
        PyTorch tensor.

    ValueError
        If ``loader`` is empty, the model contains no parameters, the
        model is not located on ``device``, images do not have shape
        ``(batch, channels, height, width)``, labels do not have shape
        ``(batch,)``, the number of images and labels differs, the model
        output does not have shape ``(batch, classes)``, the number of
        model outputs and labels differs, or the calculated loss is not
        finite.
    """
    # -------- Check types --------

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    if not isinstance(loader, DataLoader):
        raise TypeError(
            "loader must be an instance of torch.utils.data.DataLoader"
        )

    if not isinstance(criterion, torch.nn.Module):
        raise TypeError(
            "criterion must be an instance of torch.nn.Module"
        )

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer must be an instance of torch.optim.Optimizer"
        )

    if not isinstance(device, torch.device):
        raise TypeError(
            "device must be an instance of torch.device"
        )

    # -------- Check values --------

    if len(loader) == 0:
        raise ValueError("loader must contain at least one batch")

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("model must contain at least one parameter")

    if model_device.type != device.type:
        raise ValueError(
            f"Model is on {model_device}, but device is {device}"
        )

    # -------- Train one epoch --------

    model.train()

    total_loss = 0.0

    for images, labels in loader:

        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor")

        if images.ndim != 4:
            raise ValueError(
                "Expected images to have shape "
                "(batch, channels, height, width), "
                f"got {tuple(images.shape)}"
            )

        if labels.ndim != 1:
            raise ValueError(
                "Expected labels to have shape (batch,), "
                f"got {tuple(labels.shape)}"
            )

        if images.size(0) != labels.size(0):
            raise ValueError(
                "Number of images and labels must be equal"
            )

        if labels.dtype != torch.long:
            raise TypeError(
                f"Labels must have dtype torch.long, got {labels.dtype}"
            )

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                "Model output must be a torch.Tensor"
            )

        if outputs.ndim != 2:
            raise ValueError(
                "Expected model output to have shape "
                "(batch, classes), "
                f"got {tuple(outputs.shape)}"
            )

        if outputs.size(0) != labels.size(0):
            raise ValueError(
                "Number of model outputs and labels must be equal"
            )

        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            raise ValueError(
                f"Loss must be finite, got {loss.item()}"
            )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    num_classes: int,
    min_delta: float,
    patience: int = 5,
) -> tuple[torch.nn.Module, dict]:
    """
    Train a model with early stopping based on validation loss.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to train.
    train_loader : DataLoader
        DataLoader providing training data.
    validation_loader : DataLoader
        DataLoader providing validation data.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which the model and data are located.
    num_epochs : int
        Maximum number of training epochs.
    num_classes: int
        Number of classes.
    min_delta : float
        Minimum improvement to continue learning process.
    patience : int
        Number of consecutive epochs without improvement
        allowed before stopping training. Default: 5.

    Returns
    -------
    model : torch.nn.Module
        The model restored to its best validation performance.
    history : dict
        Training history.

    Raises
    ------
    TypeError
        If ``model``, ``train_loader``, ``validation_loader``,
        ``criterion``, ``optimizer``, or ``device`` has an invalid type.
        If ``min_delta`` is not a real number. Also raised if ``num_epochs``, 
        ``patience``, or ``num_classes`` is not an integer.

    ValueError
        If ``train_loader`` or ``validation_loader`` is empty, the model
        contains no parameters, the model is not located on ``device``,
        ``num_epochs``, ``patience`` or ``num_classes`` is not positive, 
        or ``min_delta`` is negative.
    """

    # -------- Check types --------

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            "train_loader must be an instance of torch.utils.data.DataLoader"
        )

    if not isinstance(validation_loader, DataLoader):
        raise TypeError(
            "validation_loader must be an instance of "
            "torch.utils.data.DataLoader"
        )

    if not isinstance(criterion, torch.nn.Module):
        raise TypeError(
            "criterion must be an instance of torch.nn.Module"
        )

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer must be an instance of torch.optim.Optimizer"
        )

    if not isinstance(device, torch.device):
        raise TypeError(
            "device must be an instance of torch.device"
        )

    if isinstance(num_epochs, bool) or not isinstance(num_epochs, int):
        raise TypeError(
            "num_epochs must be an integer, "
            f"got {type(num_epochs).__name__}"
        )

    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(
            f"num_classes must be an integer, got {type(num_classes).__name__}"
        )

    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
        raise TypeError(
            f"min_delta must be a number, got {type(min_delta).__name__}"
        )

    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError(
            "patience must be an integer, "
            f"got {type(patience).__name__}"
        )

    # -------- Check values --------

    if num_epochs <= 0:
        raise ValueError(
            f"num_epochs must be greater than 0, got {num_epochs}"
        )

    if num_classes <= 0:
        raise ValueError(
            f"num_classes must be greater than 0, got {num_classes}"
        )

    if patience <= 0:
        raise ValueError(
            f"patience must be positive, got {patience}"
        )

    if len(train_loader) == 0:
        raise ValueError(
            "train_loader must contain at least one batch"
        )

    if len(validation_loader) == 0:
        raise ValueError(
            "validation_loader must contain at least one batch"
        )

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        raise ValueError(
            "model must contain at least one parameter"
        )

    if min_delta < 0:
        raise ValueError(
            f"min_delta must be non-negative, got {min_delta}"
        )

    if model_device.type != device.type:
        raise ValueError(
            f"Model is on {model_device}, but device is {device}"
        )

    # -------- Train model --------

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, lab, pred, prob = evaluate_model(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        metrics = calculate_metrics(
            pred,
            prob,
            lab,
            num_classes = num_classes,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"time: {epoch_time//60:.0f} min {epoch_time%60:.1f} s"
        )

        print(
            "Metrics: "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1-score: {metrics['f1']:.4f} | "
            f"AUROC: {metrics['auroc']:.4f}"
        )

        # Check whether validation loss improved
        if best_val_loss - val_loss > min_delta:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save a copy of the best model parameters
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:

            print(
                f"Early stopping after {epoch + 1} epochs. "
                f"Best validation loss: {best_val_loss:.4f}"
            )

            break

    # Restore best model
    model.load_state_dict(best_model_state)

    return model, history
