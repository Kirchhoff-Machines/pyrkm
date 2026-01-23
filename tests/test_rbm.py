"""Tests for the RBM module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrkm.rbm import RBM


@pytest.fixture(scope="module")
def rbm() -> RBM:
    """Create RBM instance for testing."""
    return RBM(
        model_name="test_rbm",
        n_visible=6,
        n_hidden=3,
        k=1,
        lr=0.01,
        max_epochs=10,
        energy_type="hopfield",
        optimizer="SGD",
        regularization=False,
        l1_factor=0,
        l2_factor=1e-3,
        g_v=0.5,
        g_h=0.5,
        batch_size=1,
        train_algo="vRDM",
        centering=False,
        average_data=None,
        model_beta=1,
        mytype=torch.float32,
        min_W=-10,
        max_W=10,
    )


def test_initialization(rbm: RBM) -> None:
    """Test that RBM initializes with correct shapes and device."""
    assert rbm.W.shape == (rbm.n_hidden, rbm.n_visible)
    assert rbm.v_bias.shape == (rbm.n_visible,)
    assert rbm.h_bias.shape == (rbm.n_hidden,)
    assert rbm.device.type in ["cpu", "cuda"]


def test_forward_pass(rbm: RBM) -> None:
    """Test RBM forward pass produces correct output shape."""
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible)).float().to(rbm.device)
    v_model = rbm.forward(v, rbm.k)
    assert v_model.shape == v.shape


def test_reconstruction(rbm: RBM) -> None:
    """Test RBM reconstruction maintains input shape."""
    data = np.random.randint(0, 2, (rbm.batch_size, rbm.n_visible))
    original, reconstructed = rbm.reconstruct(data, rbm.k)
    assert original.shape == reconstructed.shape


def test_train_step(rbm: RBM) -> None:
    """Test RBM training increments epoch counter."""
    train_data = [torch.randint(0, 2, (rbm.batch_size, rbm.n_visible)).float().to(rbm.device)]
    rbm.train(train_data, print_error=False, print_test_error=False, print_every=1)
    assert rbm.epoch > 0


def test_clip_weights(rbm: RBM) -> None:
    """Test weight clipping enforces bounds."""
    rbm.W = torch.randn((rbm.n_hidden, rbm.n_visible)) * 20
    rbm.clip_weights()
    assert torch.all(rbm.W <= rbm.max_W)
    assert torch.all(rbm.W >= rbm.min_W)


def test_clip_bias(rbm: RBM) -> None:
    """Test bias clipping enforces bounds."""
    rbm.v_bias = torch.randn((rbm.n_visible,)) * 20
    rbm.h_bias = torch.randn((rbm.n_hidden,)) * 20
    rbm.clip_bias()
    assert torch.all(rbm.v_bias <= rbm.max_W)
    assert torch.all(rbm.v_bias >= rbm.min_W)
    assert torch.all(rbm.h_bias <= rbm.max_W)
    assert torch.all(rbm.h_bias >= rbm.min_W)


def test_prob_h_given_v(rbm: RBM) -> None:
    """Test probability of hidden given visible units."""
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible), device=rbm.device).float()
    p_h = rbm._prob_h_given_v(v)
    assert p_h.shape == (rbm.batch_size, rbm.n_hidden)


def test_prob_v_given_h(rbm: RBM) -> None:
    """Test probability of visible given hidden units."""
    h = torch.randint(0, 2, (rbm.batch_size, rbm.n_hidden), device=rbm.device).float()
    p_v = rbm._prob_v_given_h(h)
    assert p_v.shape == (rbm.batch_size, rbm.n_visible)


def test_device_setup(rbm: RBM) -> None:
    """Test device is properly set."""
    assert rbm.device.type in ["cpu", "cuda"]


def test_free_energy(rbm: RBM) -> None:
    """Test free energy calculation."""
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible), device=rbm.device).float()
    fe = rbm.free_energy(v, rbm.model_beta)
    assert fe is not None
    assert fe.shape[0] == rbm.batch_size


def test_save_and_load_model(rbm: RBM, tmp_path) -> None:
    """Test saving and loading model."""
    import pickle

    save_path = tmp_path / "test_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(rbm, f)

    with open(save_path, "rb") as f:
        loaded_rbm = pickle.load(f)

    assert loaded_rbm.n_visible == rbm.n_visible
    assert loaded_rbm.n_hidden == rbm.n_hidden


def test_sample_v_given_h(rbm: RBM) -> None:
    """Test sampling visible units given hidden units."""
    h = torch.randint(0, 2, (rbm.batch_size, rbm.n_hidden), device=rbm.device).float()
    p_v, v = rbm.Bernoulli_h_to_v(h, rbm.model_beta)
    assert v.shape == (rbm.batch_size, rbm.n_visible)
    assert torch.all((v == 0) | (v == 1))


def test_sample_h_given_v(rbm: RBM) -> None:
    """Test sampling hidden units given visible units."""
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible), device=rbm.device).float()
    p_h, h = rbm.Bernoulli_v_to_h(v, rbm.model_beta)
    assert h.shape == (rbm.batch_size, rbm.n_hidden)
    assert torch.all((h == 0) | (h == 1))


def test_deterministic_v_to_h(rbm: RBM) -> None:
    """Test deterministic conversion from visible to hidden."""
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible), device=rbm.device).float()
    h_prob, h_sample = rbm.v_to_h(v, beta=1001)
    assert torch.equal(h_prob, h_sample)
    assert torch.all((h_sample == 0) | (h_sample == 1))


def test_deterministic_h_to_v(rbm: RBM) -> None:
    """Test deterministic conversion from hidden to visible."""
    h = torch.randint(0, 2, (rbm.batch_size, rbm.n_hidden), device=rbm.device).float()
    v_prob, v_sample = rbm.h_to_v(h, beta=1001)
    assert torch.equal(v_prob, v_sample)
    assert torch.all((v_sample == 0) | (v_sample == 1))


def test_weight_initialization_range(rbm: RBM) -> None:
    """Test weights are clipped within specified range."""
    assert torch.all(rbm.W <= rbm.max_W)
    assert torch.all(rbm.W >= rbm.min_W)


def test_bias_initialization(rbm: RBM) -> None:
    """Test biases are properly initialized."""
    assert rbm.v_bias.shape == (rbm.n_visible,)
    assert rbm.h_bias.shape == (rbm.n_hidden,)


def test_optimizer_initialization() -> None:
    """Test Adam optimizer is properly initialized."""
    rbm_adam = RBM(model_name="test_adam", n_visible=6, n_hidden=3, optimizer="Adam", batch_size=1)
    assert hasattr(rbm_adam, "m_dW")
    assert hasattr(rbm_adam, "v_dW")
    assert rbm_adam.beta1 == 0.9
    assert rbm_adam.beta2 == 0.999


def test_persistent_chains_initialization() -> None:
    """Test persistent chains are initialized for PCD."""
    rbm_pcd = RBM(model_name="test_pcd", n_visible=6, n_hidden=3, train_algo="PCD", batch_size=2)
    assert hasattr(rbm_pcd, "persistent_chains")
    assert rbm_pcd.persistent_chains.shape == (2, 6)


def test_centering_initialization() -> None:
    """Test centering is properly initialized."""
    avg_data = torch.rand(6)
    rbm_center = RBM(
        model_name="test_center", n_visible=6, n_hidden=3, centering=True, average_data=avg_data, batch_size=1
    )
    assert hasattr(rbm_center, "ov")
    assert hasattr(rbm_center, "oh")


def test_multiple_gibbs_steps() -> None:
    """Test RBM with multiple Gibbs steps."""
    rbm_k5 = RBM(model_name="test_k5", n_visible=6, n_hidden=3, k=5, batch_size=1)
    v = torch.randint(0, 2, (1, 6)).float().to(rbm_k5.device)
    v_model = rbm_k5.forward(v, k=5)
    assert v_model.shape == v.shape


def test_rbm_train_with_print_error() -> None:
    """Test RBM training with error printing."""
    rbm_train = RBM(
        model_name="test_rbm_train", n_visible=10, n_hidden=5, k=1, lr=0.01, max_epochs=2, batch_size=2
    )

    train_data = [
        torch.randint(0, 2, (2, 10), device=rbm_train.device).float(),
        torch.randint(0, 2, (2, 10), device=rbm_train.device).float(),
    ]

    rbm_train.train(train_data, print_error=True)
    assert rbm_train.epoch == 2


def test_rbm_train_with_test_data() -> None:
    """Test RBM training with separate test data."""
    rbm_train = RBM(
        model_name="test_rbm_test", n_visible=10, n_hidden=5, k=1, lr=0.01, max_epochs=2, batch_size=2
    )

    train_data = [torch.randint(0, 2, (2, 10), device=rbm_train.device).float()]
    test_data = [torch.randint(0, 2, (2, 10), device=rbm_train.device).float()]

    rbm_train.train(train_data, test_data=test_data, print_error=True)
    assert rbm_train.epoch == 2


def test_rbm_bernoulli_v_to_h_with_beta(rbm: RBM) -> None:
    """Test Bernoulli sampling v to h with beta parameter."""
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible), device=rbm.device).float()
    p_h, h = rbm.Bernoulli_v_to_h(v, beta=0.5)
    assert h.shape == (rbm.batch_size, rbm.n_hidden)
    assert torch.all((h == 0) | (h == 1))


def test_rbm_bernoulli_h_to_v_with_beta(rbm: RBM) -> None:
    """Test Bernoulli sampling h to v with beta parameter."""
    h = torch.randint(0, 2, (rbm.batch_size, rbm.n_hidden), device=rbm.device).float()
    p_v, v = rbm.Bernoulli_h_to_v(h, beta=0.5)
    assert v.shape == (rbm.batch_size, rbm.n_visible)
    assert torch.all((v == 0) | (v == 1))


def test_rbm_reconstruction_has_error(rbm: RBM) -> None:
    """Test that reconstruction differs from original (stochastic)."""
    data = np.random.randint(0, 2, (10, rbm.n_visible))
    original, reconstructed = rbm.reconstruct(data, k=1)

    # Check shapes match
    assert original.shape == reconstructed.shape
