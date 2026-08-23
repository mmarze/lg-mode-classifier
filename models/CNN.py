import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    """
    Convolutional neural network for image classification.

    The network consists of four convolutional blocks, each containing
    a convolutional layer, batch normalization, ReLU activation, and
    max pooling. The feature maps are then reduced using adaptive
    average pooling and passed to a fully connected classification layer.

    The model expects single-channel (grayscale) images and produces
    one output logit for each class.

    Args:
        n_classes (int): Number of target classes. Must be an integer
            greater than or equal to 2.

    Raises:
        TypeError: If ``n_classes`` is not an integer.
        ValueError: If ``n_classes`` is less than 2.

    Input:
        Tensor of shape ``(batch_size, 1, height, width)``.

    Output:
        Tensor of shape ``(batch_size, n_classes)`` containing the
        classification logits.

    Example:
        >>> model = CNN(n_classes=6)
        >>> x = torch.randn(8, 1, 509, 512)
        >>> output = model(x)
        >>> output.shape
        torch.Size([8, 6])
    """

    def __init__(self, n_classes):
        super().__init__()

        if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):

            raise TypeError(
                "Number of classes must be an integer."
            )

        if n_classes < 2:
            raise ValueError(
                "Number of classes must be greater than or equal to 2."
                )

        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == '__main__':
    model = CNN(n_classes=6)
    x = torch.randn(8, 1, 509, 512)
    output = model(x)
    print(output.shape)
