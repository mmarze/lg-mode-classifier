import numpy as np
import pytest
import torch
import torch.nn as nn

from models.CNN import CNN


class TestCNNInitialization:

    @pytest.mark.parametrize(
            "n_classes", 
            [2, 3, 6, 10, 100]
    )
    def test_valid_number_of_classes(self, n_classes):
        model = CNN(n_classes)

        assert model.n_classes == n_classes

    @pytest.mark.parametrize(
        "n_classes",
        [0, 1, -1, -10]
    )
    def test_number_of_classes_less_than_two(self, n_classes):
        with pytest.raises(
            ValueError,
            match="Number of classes must be greater than or equal to 2."
        ):
            CNN(n_classes)

    @pytest.mark.parametrize(
        "n_classes",
        [1.5, 6.0, "6", None, [6], True]
    )
    def test_number_of_classes_must_be_integer(self, n_classes):
        with pytest.raises(
            TypeError,
            match="Number of classes must be an integer."
        ):
            CNN(n_classes)

    @pytest.mark.parametrize(
        "n_classes",
        [np.int32(6), np.int64(6)]
    )
    def test_numpy_integer_is_valid(self, n_classes):
        model = CNN(n_classes)

        assert model.n_classes == n_classes


class TestCNNArchitecture:

    def test_first_convolution(self):
        model = CNN(6)

        first_conv = model.features[0]

        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 16
        assert first_conv.kernel_size == (3, 3)
        assert first_conv.padding == (1, 1)

    def test_last_convolution(self):
        model = CNN(6)

        last_conv = model.features[11]

        assert isinstance(last_conv, nn.Conv2d)
        assert last_conv.in_channels == 128
        assert last_conv.out_channels == 256
        assert last_conv.kernel_size == (3, 3)

    def test_classifier_output_features(self):
        n_classes = 6
        model = CNN(n_classes)

        linear = model.classifier[2]

        assert isinstance(linear, nn.Linear)
        assert linear.in_features == 256
        assert linear.out_features == n_classes


class TestCNNForward:

    def test_output_shape(self):
        model = CNN(n_classes=6)

        x = torch.randn(4, 1, 509, 512)

        output = model(x)

        assert output.shape == (4, 6)

    @pytest.mark.parametrize(
            "n_classes", 
            [2, 3, 6, 10]
    )
    def test_output_shape_depends_on_number_of_classes(self, n_classes):
        model = CNN(n_classes)

        x = torch.randn(4, 1, 509, 512)

        output = model(x)

        assert output.shape == (4, n_classes)

    def test_batch_size_is_preserved(self):
        model = CNN(n_classes=6)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 1, 509, 512)

            output = model(x)

            assert output.shape[0] == batch_size

    def test_forward_returns_tensor(self):
        model = CNN(n_classes=6)

        x = torch.randn(2, 1, 509, 512)

        output = model(x)

        assert isinstance(output, torch.Tensor)

    def test_output_contains_finite_values(self):
        model = CNN(n_classes=6)

        x = torch.randn(2, 1, 509, 512)

        output = model(x)

        assert torch.isfinite(output).all()

    def test_feature_map_shape(self):
        model = CNN(n_classes=6)

        x = torch.randn(2, 1, 509, 512)

        features = model.features(x)

        assert features.shape == (2, 256, 15, 16)
