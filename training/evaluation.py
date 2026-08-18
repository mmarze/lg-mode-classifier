import torch
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy, Precision, Recall, 
    F1Score, AUROC, ConfusionMatrix,
    )
import numpy as np


def evaluate_model(
        model: torch.nn.Module, 
        loader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate a trained model on a dataset.

    Runs the model in evaluation mode without computing gradients and
    collects the loss, predicted classes, target labels, and class
    probabilities for all samples.

    Args:
        model: PyTorch model to evaluate.
        loader: DataLoader providing input images and target labels.
        criterion: Loss function used to calculate the model loss.
        device: Device on which the model and input tensors are located.

    Returns:
        tuple:
            loss (float): Average loss across all batches.
            all_labels (torch.Tensor): Ground-truth labels for all samples.
            all_predictions (torch.Tensor): Predicted class indices for all samples.
            all_probabilities (torch.Tensor): Predicted class probabilities
                for all samples, with shape (num_samples, num_classes).
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be an instance of torch.utils.data.DataLoader")

    if not isinstance(criterion, torch.nn.Module):
        raise TypeError("criterion must be an instance of torch.nn.Module")

    if not isinstance(device, torch.device):
        raise TypeError("device must be an instance of torch.device")

    model_device = next(model.parameters()).device

    if model_device != device:
        raise ValueError(
            f"Model is on {model_device}, but device is {device}"
        )

    # Switch model to evaluation mode
    model.eval()

    total_loss = 0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    # With no backpropagation
    with torch.no_grad():

        for images, labels in loader:

            if not isinstance(images, torch.Tensor):
                raise TypeError("images must be a torch.Tensor")

            if not isinstance(labels, torch.Tensor):
                raise TypeError("labels must be a torch.Tensor")

            if images.ndim != 4:
                raise ValueError(
                    f"Expected images to have 4 dimensions "
                    f"[batch, channels, height, width], got {images.shape}"
                )

            if labels.ndim != 1:
                raise ValueError(
                    f"Expected labels to have shape [batch], got {labels.shape}"
                )

            if images.size(0) != labels.size(0):
                raise ValueError(
                    "Number of images and labels must be equal"
                )

            if labels.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError(
                    f"Labels must have an integer dtype, got {labels.dtype}"
                )

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            probability_sums = probabilities.sum(dim=1)

            if not torch.allclose(
                probability_sums,
                torch.ones_like(probability_sums),
                atol=1e-5,
            ):
                raise ValueError(
                    "Probabilities for each sample must sum to approximately 1"
                )
            
            predictions = outputs.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())

    loss = total_loss / len(loader)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities)

    return loss, all_labels, all_predictions, all_probabilities


def _validate_classification_inputs(
    target: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> None:
    """Validate inputs shared by classification metrics."""

    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")

    if not isinstance(predictions, torch.Tensor):
        raise TypeError("predictions must be a torch.Tensor")

    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    if target.ndim != 1:
        raise ValueError(
            f"target must have shape [N], got {target.shape}"
        )

    if predictions.ndim != 1:
        raise ValueError(
            f"predictions must have shape [N], got {predictions.shape}"
        )

    if target.size(0) != predictions.size(0):
        raise ValueError(
            "target and predictions must contain the same number of samples"
        )

    if target.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(
            f"target must have an integer dtype, got {target.dtype}"
        )

    if predictions.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            f"predictions must have an integer dtype, got {predictions.dtype}"
        )

    if torch.any(target < 0) or torch.any(target >= num_classes):
        raise ValueError(
            "target contains class indices outside the valid range"
        )

    if torch.any(predictions < 0) or torch.any(predictions >= num_classes):
        raise ValueError(
            "predictions contain class indices outside the valid range"
        )
    

def calculate_metrics(
        predictions: torch.Tensor, 
        probabilities: torch.Tensor, 
        target: torch.Tensor, 
        num_classes: int,
    ) -> dict[str, torch.Tensor]:
    """
    Calculate classification metrics for a multiclass model.

    Computes macro-averaged precision, recall, F1 score, and AUROC,
    as well as overall classification accuracy.

    Args:
        predictions (torch.Tensor): Predicted class indices for each sample.
        probabilities (torch.Tensor): Predicted class probabilities for each
            sample, with shape (num_samples, num_classes).
        target (torch.Tensor): Ground-truth class indices for each sample.
        num_classes (int): Number of classes in the classification task.

    Returns:
        dict:
            Dictionary containing the following metrics:
            - "accuracy": Overall classification accuracy.
            - "precision": Macro-averaged precision.
            - "recall": Macro-averaged recall.
            - "f1": Macro-averaged F1 score.
            - "auroc": Macro-averaged multiclass AUROC.
    """

    _validate_classification_inputs(
        target,
        predictions,
        num_classes,
    )

    if not torch.isfinite(probabilities).all():
        raise ValueError(
            "probabilities contain NaN or infinite values"
        )

    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError(
            "probabilities must be between 0 and 1"
        )

    
    accuracy = Accuracy(
        task="multiclass", 
        num_classes=num_classes
    )
    accuracy_val = accuracy(predictions, target)

    precision = Precision(
        task='multiclass', 
        num_classes=num_classes,  
        average="macro"
    )
    precision_val = precision(predictions, target)

    recall = Recall(
        task='multiclass', 
        num_classes=num_classes, 
        average="macro"
    )
    recall_val = recall(predictions, target)

    f1 = F1Score(
        task="multiclass", 
        num_classes=num_classes, 
        average="macro"
    )
    f1_val = f1(predictions, target)

    auroc = AUROC(
        task="multiclass", 
        num_classes=num_classes, 
        average="macro"
        )
    auroc_val = auroc(probabilities, target)

    return {
        "accuracy": accuracy_val, 
        "precision": precision_val, 
        "recall": recall_val, 
        "f1": f1_val, 
        "auroc": auroc_val
    }


def confusion_matrix(
        target: torch.Tensor, 
        predictions: torch.Tensor, 
        num_classes: int,
        ) -> torch.Tensor:
    """
    Calculate the confusion matrix for a multiclass classification task.

    Args:
        target (torch.Tensor): Ground-truth class indices for each sample.
        predictions (torch.Tensor): Predicted class indices for each sample.
        num_classes (int): Number of classes in the classification task.

    Returns:
        torch.Tensor: Confusion matrix where rows correspond to target
            classes and columns correspond to predicted classes.
    """

    _validate_classification_inputs(
            target,
            predictions,
            num_classes,
        )

    confmat = ConfusionMatrix(
        task="multiclass",
        num_classes=num_classes,
    )

    return confmat(predictions, target)
