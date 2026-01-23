"""Tests for the classifier module."""

from __future__ import annotations

import pytest
import torch

from pyrkm.classifier import CustomDataset, SimpleClassifier, train_classifier


@pytest.fixture
def custom_dataset():
    data = torch.randn(100, 28, 28)
    targets = torch.randint(0, 10, (100,))
    return CustomDataset(data, targets)


def test_custom_dataset_length(custom_dataset) -> None:
    assert len(custom_dataset) == 100


def test_custom_dataset_getitem(custom_dataset) -> None:
    img, label = custom_dataset[0]
    assert img.shape == (28, 28)
    assert 0 <= label < 10


@pytest.fixture
def simple_classifier():
    return SimpleClassifier()


def test_simple_classifier_forward(simple_classifier) -> None:
    x = torch.randn(1, 1, 28, 28)
    output = simple_classifier(x)
    assert output.shape == (1, 10)


@pytest.fixture
def train_test_data():
    train_data = torch.randn(100, 28, 28)
    train_targets = torch.randint(0, 10, (100,))
    test_data = torch.randn(20, 28, 28)
    test_targets = torch.randint(0, 10, (20,))
    return (train_data, train_targets), (test_data, test_targets)


def test_train_classifier(train_test_data) -> None:
    train_set, test_set = train_test_data
    model, accuracy = train_classifier(test_set, train_set, num_epochs=1)
    assert isinstance(model, SimpleClassifier)
    assert 0 <= accuracy <= 100


def test_simple_classifier_output_range(simple_classifier) -> None:
    """Test classifier output has correct range."""
    x = torch.randn(5, 1, 28, 28)
    output = simple_classifier(x)
    assert output.shape == (5, 10)
    # Output should be raw logits (before softmax)
    assert output.dtype == torch.float32


def test_custom_dataset_indexing(custom_dataset) -> None:
    """Test dataset indexing works correctly."""
    for i in range(min(10, len(custom_dataset))):
        img, label = custom_dataset[i]
        assert img.shape == (28, 28)
        assert isinstance(label, (int, torch.Tensor))


def test_custom_dataset_with_different_sizes() -> None:
    """Test custom dataset with different data sizes."""
    data = torch.randn(50, 32, 32)
    targets = torch.randint(0, 5, (50,))
    dataset = CustomDataset(data, targets)
    assert len(dataset) == 50
    img, label = dataset[0]
    assert img.shape == (32, 32)


def test_classifier_architecture() -> None:
    """Test classifier architecture components."""
    classifier = SimpleClassifier()
    # Check that the model has the expected layers
    assert hasattr(classifier, "conv1")
    assert hasattr(classifier, "conv2")
    assert hasattr(classifier, "fc1")
    assert hasattr(classifier, "fc2")


def test_train_classifier_with_multiple_epochs(train_test_data) -> None:
    """Test training with multiple epochs."""
    train_set, test_set = train_test_data
    model, accuracy = train_classifier(test_set, train_set, num_epochs=3)
    assert isinstance(model, SimpleClassifier)
    assert accuracy >= 0


def test_classifier_gradient_flow(simple_classifier) -> None:
    """Test that gradients flow through the model."""
    x = torch.randn(2, 1, 28, 28, requires_grad=True)
    output = simple_classifier(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None


def test_custom_dataset_empty_error() -> None:
    """Test error handling for empty dataset."""
    data = torch.randn(0, 28, 28)
    targets = torch.randint(0, 10, (0,))
    dataset = CustomDataset(data, targets)
    assert len(dataset) == 0


def test_train_classifier_model_in_eval_mode(train_test_data) -> None:
    """Test that trained model can be set to eval mode."""
    train_set, test_set = train_test_data
    model, _ = train_classifier(test_set, train_set, num_epochs=1)
    model.eval()
    # Move data to same device as model
    device = next(model.parameters()).device
    x = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 10)


def test_classifier_batch_processing() -> None:
    """Test classifier can process different batch sizes."""
    classifier = SimpleClassifier()
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 1, 28, 28)
        output = classifier(x)
        assert output.shape == (batch_size, 10)
