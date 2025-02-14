from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from pyrkm.utils import (
    Compute_FID,
    Compute_S,
    ComputeAATS,
    Covariance_error,
    PowerSpectrum_MSE,
    Third_moment_error,
    binarize_image,
    generate_S_matrix,
    generate_synthetic_data,
    getbasebias,
    my_entropy,
    unpickle,
)


def test_getbasebias():
    data = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    result = getbasebias(data)
    assert torch.is_tensor(result)


@pytest.mark.skip(reason='Test is currently broken')
def test_Covariance_error():
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    Nv = 5
    result = Covariance_error(data1, data2, Nv)
    assert torch.is_tensor(result)


def test_Third_moment_error():
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    Nv = 5
    result = Third_moment_error(data1, data2, Nv)
    assert torch.is_tensor(result)


def test_PowerSpectrum_MSE():
    data1 = torch.randn(10, 10)
    data2 = torch.randn(10, 10)
    result = PowerSpectrum_MSE(data1, data2)
    assert torch.is_tensor(result)


def test_ComputeAATS():
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    result = ComputeAATS(data1, data2)
    assert len(result) == 2


@pytest.mark.skip(reason='Test is currently broken')
def test_Compute_FID():
    data1 = torch.randn(10, 3, 28, 28)
    data2 = torch.randn(10, 3, 28, 28)
    result = Compute_FID(data1, data2)
    assert isinstance(result, float)


def test_Compute_S():
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    result = Compute_S(data1, data2)
    assert isinstance(result, float)


def test_generate_S_matrix():
    shape = (10, 10)
    target = 0.5
    result = generate_S_matrix(shape, target)
    assert result.shape == shape


def test_generate_synthetic_data():
    target_entropy = 0.5
    data_size = (10, 28, 28)
    result = generate_synthetic_data(target_entropy, data_size)
    assert result.shape == data_size


def test_my_entropy():
    data = np.random.rand(10, 28, 28)
    result = my_entropy(data)
    assert len(result) == 2


def test_binarize_image():
    image = np.random.randint(0, 255, (28, 28))
    result = binarize_image(image)
    assert result.shape == image.shape


def test_unpickle():
    tmp_path = 'tests/'

    sample_data = {'key': 'value'}
    sample_file = tmp_path + 'sample.pkl'
    with open(sample_file, 'wb') as f:
        pickle.dump(sample_data, f)

    result = unpickle(sample_file)
    assert result == sample_data
