from __future__ import annotations

import pytest
import torch

from pyrkm.classifier import CustomDataset, SimpleClassifier, train_classifier


@pytest.fixture
def custom_dataset():
    data = torch.randn(100, 28, 28)
    targets = torch.randint(0, 10, (100, ))
    return CustomDataset(data, targets)


def test_custom_dataset_length(custom_dataset):
    assert len(custom_dataset) == 100


def test_custom_dataset_getitem(custom_dataset):
    img, label = custom_dataset[0]
    assert img.shape == (28, 28)
    assert 0 <= label < 10


@pytest.fixture
def simple_classifier():
    return SimpleClassifier()


def test_simple_classifier_forward(simple_classifier):
    x = torch.randn(1, 1, 28, 28)
    output = simple_classifier(x)
    assert output.shape == (1, 10)


@pytest.fixture
def train_test_data():
    train_data = torch.randn(100, 28, 28)
    train_targets = torch.randint(0, 10, (100, ))
    test_data = torch.randn(20, 28, 28)
    test_targets = torch.randint(0, 10, (20, ))
    return (train_data, train_targets), (test_data, test_targets)


def test_train_classifier(train_test_data):
    train_set, test_set = train_test_data
    model, accuracy = train_classifier(test_set, train_set, num_epochs=1)
    assert isinstance(model, SimpleClassifier)
    assert 0 <= accuracy <= 100
