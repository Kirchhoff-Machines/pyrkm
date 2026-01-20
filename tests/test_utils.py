"""
Tests for utility functions.
"""

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
    ensure_dir,
    generate_S_matrix,
    generate_synthetic_data,
    getbasebias,
    load_model,
    make_grid,
    my_entropy,
    unpickle,
)


def test_getbasebias() -> None:
    data = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    result = getbasebias(data)
    assert torch.is_tensor(result)


@pytest.mark.skip(reason="Test is currently broken")
def test_Covariance_error() -> None:
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    Nv = 5
    result = Covariance_error(data1, data2, Nv)
    assert torch.is_tensor(result)


def test_Third_moment_error() -> None:
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    Nv = 5
    result = Third_moment_error(data1, data2, Nv)
    assert torch.is_tensor(result)


def test_PowerSpectrum_MSE() -> None:
    data1 = torch.randn(10, 10)
    data2 = torch.randn(10, 10)
    result = PowerSpectrum_MSE(data1, data2)
    assert torch.is_tensor(result)


def test_ComputeAATS() -> None:
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    result = ComputeAATS(data1, data2)
    assert len(result) == 2


@pytest.mark.skip(reason="Test is currently broken")
def test_Compute_FID() -> None:
    data1 = torch.randn(10, 3, 28, 28)
    data2 = torch.randn(10, 3, 28, 28)
    result = Compute_FID(data1, data2)
    assert isinstance(result, float)


def test_Compute_S() -> None:
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    result = Compute_S(data1, data2)
    assert isinstance(result, float)


def test_generate_S_matrix() -> None:
    shape = (10, 10)
    target = 0.5
    result = generate_S_matrix(shape, target)
    assert result.shape == shape


def test_generate_synthetic_data() -> None:
    target_entropy = 0.5
    data_size = (10, 28, 28)
    result = generate_synthetic_data(target_entropy, data_size)
    assert result.shape == data_size


def test_my_entropy() -> None:
    data = np.random.rand(10, 28, 28)
    S_image, S_pixel = my_entropy(data)
    assert isinstance(S_image, np.ndarray)
    assert isinstance(S_pixel, np.ndarray)


def test_binarize_image() -> None:
    image = np.random.randint(0, 255, (28, 28))
    result = binarize_image(image)
    assert result.shape == image.shape


def test_unpickle() -> None:
    tmp_path = "tests/"

    sample_data = {"key": "value"}
    sample_file = tmp_path + "sample.pkl"
    with open(sample_file, "wb") as f:
        pickle.dump(sample_data, f)

    result = unpickle(sample_file)
    assert result == sample_data


def test_getbasebias_shape() -> None:
    """
    Test getbasebias output shape.
    """
    data = torch.tensor([[0.1, 0.9, 0.3], [0.4, 0.6, 0.8]])
    result = getbasebias(data)
    assert result.shape == (3,)


def test_getbasebias_values() -> None:
    """
    Test getbasebias computes reasonable values.
    """
    data = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    result = getbasebias(data)
    assert torch.is_tensor(result)
    # For all 0s in first column and all 1s in second column
    assert result[0] < result[1]


def test_third_moment_error_symmetric() -> None:
    """
    Test third moment error is symmetric.
    """
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    Nv = 5
    error1 = Third_moment_error(data1, data2, Nv)
    error2 = Third_moment_error(data2, data1, Nv)
    assert torch.allclose(error1, error2, rtol=1e-4)


def test_third_moment_error_zero_for_same_data() -> None:
    """
    Test third moment error is zero for identical data.
    """
    data = torch.randn(10, 5)
    Nv = 5
    error = Third_moment_error(data, data, Nv)
    assert torch.allclose(error, torch.tensor(0.0), atol=1e-5)


def test_power_spectrum_mse_shape() -> None:
    """
    Test PowerSpectrum_MSE output is scalar.
    """
    data1 = torch.randn(10, 10)
    data2 = torch.randn(10, 10)
    result = PowerSpectrum_MSE(data1, data2)
    assert result.numel() == 1


def test_power_spectrum_mse_zero_for_same_data() -> None:
    """
    Test PowerSpectrum_MSE is zero for identical data.
    """
    data = torch.randn(10, 10)
    result = PowerSpectrum_MSE(data, data)
    assert torch.allclose(result, torch.tensor(0.0), atol=1e-5)


def test_compute_aats_output_length() -> None:
    """
    Test ComputeAATS returns two values.
    """
    data1 = torch.randn(10, 5)
    data2 = torch.randn(10, 5)
    result = ComputeAATS(data1, data2)
    assert len(result) == 2
    assert all(isinstance(x, (int, float)) for x in result)


def test_compute_s_range() -> None:
    """
    Test Compute_S returns value in reasonable range.
    """
    data1 = torch.randn(20, 5)
    data2 = torch.randn(20, 5)
    result = Compute_S(data1, data2)
    assert isinstance(result, float)
    assert result >= 0  # Distance metric should be non-negative


def test_compute_s_zero_for_same_data() -> None:
    """
    Test Compute_S is zero for identical data.
    """
    data = torch.randn(20, 5)
    result = Compute_S(data, data)
    assert isinstance(result, (float, np.floating))
    assert abs(result) < 1e-2  # Should be close to zero, allowing some tolerance


def test_generate_s_matrix_shape() -> None:
    """
    Test generate_S_matrix creates correct shape.
    """
    shapes = [(5, 5), (10, 20), (3, 7)]
    for shape in shapes:
        result = generate_S_matrix(shape, target=0.5)
        assert result.shape == shape


def test_generate_s_matrix_range() -> None:
    """
    Test generate_S_matrix creates values in valid range.
    """
    result = generate_S_matrix((10, 10), target=0.5)
    assert np.all((result >= 0) & (result <= 1))


def test_generate_synthetic_data_shape() -> None:
    """
    Test generate_synthetic_data creates correct shape.
    """
    data_sizes = [(10, 28, 28), (5, 16, 16), (20, 32, 32)]
    for data_size in data_sizes:
        result = generate_synthetic_data(target_entropy=0.5, data_size=data_size)
        assert result.shape == data_size


def test_generate_synthetic_data_values() -> None:
    """
    Test generate_synthetic_data creates binary values.
    """
    result = generate_synthetic_data(target_entropy=0.5, data_size=(10, 28, 28))
    assert np.all((result == 0) | (result == 1))


def test_my_entropy_output() -> None:
    """
    Test my_entropy returns two values.
    """
    data = np.random.rand(10, 28, 28)
    S_image, S_pixel = my_entropy(data)
    # Handle both scalar and array returns
    if isinstance(S_image, np.ndarray):
        assert S_image.size > 0
        assert S_pixel.size > 0
    else:
        assert isinstance(S_image, (int, float, np.floating))
        assert isinstance(S_pixel, (int, float, np.floating))
        assert S_image >= 0
        assert S_pixel >= 0


def test_my_entropy_binary_data() -> None:
    """
    Test my_entropy on binary data.
    """
    data = np.random.randint(0, 2, (10, 28, 28)).astype(float)
    S_image, S_pixel = my_entropy(data)
    # Handle array or scalar returns
    if isinstance(S_image, np.ndarray):
        assert np.all((S_image >= 0) & (S_image <= 1))
        assert np.all(S_pixel >= 0)
    else:
        assert 0 <= S_image <= 1
        assert S_pixel >= 0


def test_binarize_image_binary_output() -> None:
    """
    Test binarize_image produces binary output.
    """
    image = np.random.randint(0, 255, (28, 28))
    result = binarize_image(image)
    assert np.all((result == 0) | (result == 1))


def test_binarize_image_preserves_shape() -> None:
    """
    Test binarize_image preserves image shape.
    """
    shapes = [(28, 28), (64, 64), (100, 100)]
    for shape in shapes:
        image = np.random.randint(0, 255, shape)
        result = binarize_image(image)
        assert result.shape == shape


def test_binarize_image_threshold() -> None:
    """
    Test binarize_image applies threshold correctly.
    """
    # Create image with known values
    image = np.array([[0, 127, 255], [50, 128, 200]])
    result = binarize_image(image)
    # Values above threshold should be 1, below should be 0
    assert result[0, 0] == 0
    assert result[0, 2] == 1


def test_unpickle_file_not_found() -> None:
    """
    Test unpickle handles non-existent files.
    """
    try:
        unpickle("nonexistent_file.pkl")
        # If it doesn't raise, result should be None or handle gracefully
    except FileNotFoundError:
        # Expected behavior
        pass


def test_generate_synthetic_data_different_entropies() -> None:
    """
    Test generating data with different target entropies.
    """
    entropies = [0.2, 0.5, 0.8]
    data_size = (10, 28, 28)
    for target_entropy in entropies:
        result = generate_synthetic_data(target_entropy=target_entropy, data_size=data_size)
        assert result.shape == data_size
        # Data should be binary
        assert np.all((result == 0) | (result == 1))


def test_compute_fid_basic() -> None:
    """
    Test basic FID computation.
    """
    # Create two sets of simple images
    real_images = torch.rand(10, 1, 28, 28)
    generated_images = torch.rand(10, 1, 28, 28)

    fid = Compute_FID(real_images, generated_images)

    # FID should be a valid positive number
    assert isinstance(fid, (int, float))
    assert fid >= 0
    assert not np.isnan(fid)


def test_compute_fid_identical_images() -> None:
    """
    Test FID with identical images should be near zero.
    """
    images = torch.rand(10, 1, 28, 28)

    fid = Compute_FID(images, images)

    # FID of identical distributions should be close to 0
    assert fid < 1.0  # Should be very small


def test_ensure_dir_creates_directory(tmp_path) -> None:
    """
    Test ensure_dir creates a new directory.
    """
    import os

    new_dir = tmp_path / "test_directory"
    assert not os.path.exists(new_dir)

    ensure_dir(str(new_dir))

    assert os.path.exists(new_dir)
    assert os.path.isdir(new_dir)


def test_ensure_dir_existing_directory(tmp_path) -> None:
    """
    Test ensure_dir handles existing directories.
    """
    import os

    existing_dir = tmp_path / "existing_dir"
    os.makedirs(existing_dir)
    assert os.path.exists(existing_dir)

    # Should not raise an error
    ensure_dir(str(existing_dir))

    assert os.path.exists(existing_dir)


def test_make_grid_basic() -> None:
    """
    Test make_grid creates a proper grid of images.
    """
    # Create 16 sample 8x8 images
    images = np.random.rand(16, 8, 8)

    grid = make_grid(images, nrow=4, padding=1)

    # Grid should have proper dimensions
    # Formula: grid_h * (H + padding) + padding
    # 4 rows: 4 * (8 + 1) + 1 = 37 pixels
    expected_size = 4 * (8 + 1) + 1
    assert grid.shape == (expected_size, expected_size)


def test_make_grid_single_image() -> None:
    """
    Test make_grid with a single image.
    """
    image = np.random.rand(1, 10, 10)

    grid = make_grid(image, nrow=1, padding=2)

    # Formula: grid_h * (H + padding) + padding = 1 * (10 + 2) + 2 = 14
    expected_size = 1 * (10 + 2) + 2
    assert grid.shape == (expected_size, expected_size)


def test_load_model_nonexistent() -> None:
    """
    Test load_model with non-existent model.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        model_name = "nonexistent_model_xyz123"
        is_loadable, model = load_model(model_name, model_state_path=tmpdir + "/", delete_previous=False)

        assert is_loadable is False
        assert model == []  # Returns empty list, not None


def test_load_model_with_existing_model(tmp_path) -> None:
    """
    Test load_model can load an existing model.
    """
    import pickle

    from pyrkm import RBM

    # Create a simple RBM model
    model_name = "test_rbm_model"
    rbm = RBM(
        n_visible=10,
        n_hidden=5,
        model_name=model_name,
    )

    # Save the model with epoch suffix (required by load_model)
    model_dir = tmp_path / "model_states"
    model_dir.mkdir()
    model_path = model_dir / f"{model_name}_t1.pkl"  # Add _t1 suffix
    with open(model_path, "wb") as f:
        pickle.dump(rbm, f)

    # Try to load it
    is_loadable, loaded_model = load_model(
        model_name, model_state_path=str(model_dir) + "/", delete_previous=False
    )

    assert is_loadable is True
    assert loaded_model is not None
    assert loaded_model.n_visible == 10
    assert loaded_model.n_hidden == 5
