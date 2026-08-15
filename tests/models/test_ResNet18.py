import torch
import torch.nn as nn
from models.ResNet18 import ResNet18, BasicBlock


def test_basic_block_same_dimensions():
    block = BasicBlock(64, 64, stride=1)

    x = torch.randn(4, 64, 32, 32)
    y = block(x)

    assert y.shape == (4, 64, 32, 32)


def test_basic_block_downsampling():
    block = BasicBlock(64, 128, stride=2)

    x = torch.randn(4, 64, 32, 32)
    y = block(x)

    assert y.shape == (4, 128, 16, 16)


def test_resnet18():
    model = ResNet18(num_classes=6)

    x = torch.randn(4, 1, 509, 512)
    y = model(x)

    assert y.shape == (4, 6)


def test_resnet18_backward():
    model = ResNet18(num_classes=6)

    x = torch.randn(4, 1, 509, 512)
    labels = torch.randint(0, 6, (4,))

    criterion = nn.CrossEntropyLoss()

    outputs = model(x)
    loss = criterion(outputs, labels)

    loss.backward()

    assert loss.item() > 0
