import torch
from torch.utils.data import DataLoader


def get_predictions(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        prob=False,
) ->torch.Tensor:
    """
    Generate predictions or class probabilities for a dataset.

    Runs the model in evaluation mode without computing gradients and
    returns either predicted class indices or class probabilities for
    all samples.

    Args:
        model (torch.nn.Module): PyTorch model to evaluate.
        loader (DataLoader): DataLoader providing input images and target labels.
        device (torch.device): Device on which the model and input tensors are located.
        prob (bool): If True, return class probabilities. If False, return
            predicted class indices.

    Returns:
        A tensor.Tensor containing either:
            - Predicted class indices with shape (num_samples,), or
            - Class probabilities with shape (num_samples, num_classes).
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be an instance of torch.utils.data.DataLoader")

    if not isinstance(device, torch.device):
        raise TypeError("device must be an instance of torch.device")

    if not isinstance(prob, bool):
        raise TypeError("prob must be a boolean")

    model_device = next(model.parameters()).device

    if model_device != device:
        raise ValueError(
            f"Model is on {model_device}, but device is {device}"
        )

    result = []

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            if not isinstance(images, torch.Tensor):
                raise TypeError("images must be a torch.Tensor")

            if images.ndim != 4:
                raise ValueError(
                    f"Expected images to have 4 dimensions "
                    f"[batch, channels, height, width], got {images.shape}"
                )

            images = images.to(device)

            outputs = model(images)

            if not torch.isfinite(outputs).all():
                raise ValueError("Model output contains NaN or infinite values")

            if prob:
                result.append(
                    torch.softmax(outputs, dim=1).cpu()
                )
            else:
                result.append(
                    outputs.argmax(dim=1).cpu()
                )

    if not result:
        raise ValueError("DataLoader is empty")

    return torch.cat(result)
