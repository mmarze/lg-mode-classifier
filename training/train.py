import copy

import torch
from torch.utils.data import DataLoader

from training.evaluation import evaluate_model


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

    if model_device != device:
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
        Also raised if ``num_epochs`` or ``patience`` is not an integer.

    ValueError
        If ``train_loader`` or ``validation_loader`` is empty, the model
        contains no parameters, the model is not located on ``device``,
        ``num_epochs`` is not positive, or ``patience`` is negative.
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

    if patience < 0:
        raise ValueError(
            f"patience must be non-negative, got {patience}"
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

    if model_device != device:
        raise ValueError(
            f"Model is on {model_device}, but device is {device}"
        )

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    # -------- Train model --------

    best_val_loss = float("inf")
    best_model_state = None

    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, _, _, _ = evaluate_model(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )

        # Check whether validation loss improved
        if val_loss < best_val_loss:

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
