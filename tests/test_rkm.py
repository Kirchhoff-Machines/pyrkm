"""Tests for the RKM module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrkm.rkm import RKM


@pytest.fixture(scope="module")
def rkm() -> RKM:
    """Create RKM instance for testing."""
    return RKM(
        model_name="test_rkm",
        n_visible=10,
        n_hidden=5,
        k=1,
        lr=0.01,
        max_epochs=100,
        energy_type="RKM",
        optimizer="SGD",
        batch_size=2,
        train_algo="vRDM",
        mytype=torch.float32,
    )


def test_initialization(rkm: RKM) -> None:
    """Test RKM model creates without errors."""
    assert rkm is not None
    assert rkm.n_visible == 10
    assert rkm.n_hidden == 5
    assert rkm.W.shape == (rkm.n_hidden, rkm.n_visible)
    assert rkm.v_bias.shape == (rkm.n_visible,)
    assert rkm.h_bias.shape == (rkm.n_hidden,)
    assert rkm.device.type in ["cpu", "cuda"]


def test_rkm_energy_type(rkm: RKM) -> None:
    """Test RKM has correct energy type."""
    assert rkm.energy_type == "RKM"


def test_forward_pass(rkm: RKM) -> None:
    """Test RKM forward pass produces correct output shape."""
    v = torch.randint(0, 2, (rkm.batch_size, rkm.n_visible)).float().to(rkm.device)
    v_model = rkm.forward(v, rkm.k)
    assert v_model.shape == v.shape


def test_reconstruction(rkm: RKM) -> None:
    """Test RKM reconstruction maintains input shape."""
    data = np.random.randint(0, 2, (rkm.batch_size, rkm.n_visible))
    original, reconstructed = rkm.reconstruct(data, rkm.k)
    assert original.shape == reconstructed.shape
    assert original.shape == (rkm.batch_size, rkm.n_visible)


def test_train_step() -> None:
    """Test RKM can train for one batch."""
    rkm_train = RKM(
        model_name="test_rkm_train",
        n_visible=10,
        n_hidden=5,
        k=1,
        lr=0.01,
        max_epochs=1,
        energy_type="RKM",
        batch_size=2,
    )

    train_data = [torch.randint(0, 2, (2, 10), device=rkm_train.device).float()]
    initial_epoch = rkm_train.epoch
    rkm_train.train(train_data, print_error=False)
    assert rkm_train.epoch > initial_epoch


def test_clip_weights(rkm: RKM) -> None:
    """Test weight clipping enforces bounds."""
    rkm.W = torch.randn((rkm.n_hidden, rkm.n_visible), device=rkm.device) * 50
    rkm.clip_weights()
    assert torch.all(rkm.W <= rkm.max_W)
    assert torch.all(rkm.W >= rkm.min_W)


def test_clip_bias(rkm: RKM) -> None:
    """Test bias clipping enforces bounds."""
    rkm.v_bias = torch.randn((rkm.n_visible,), device=rkm.device) * 50
    rkm.h_bias = torch.randn((rkm.n_hidden,), device=rkm.device) * 50
    rkm.clip_bias()
    assert torch.all(rkm.v_bias <= rkm.max_W)
    assert torch.all(rkm.v_bias >= rkm.min_W)
    assert torch.all(rkm.h_bias <= rkm.max_W)
    assert torch.all(rkm.h_bias >= rkm.min_W)


def test_v_to_h(rkm: RKM) -> None:
    """Test visible to hidden conversion."""
    v = torch.randint(0, 2, (rkm.batch_size, rkm.n_visible), device=rkm.device).float()
    p_h, h = rkm.v_to_h(v)
    assert p_h.shape == (rkm.batch_size, rkm.n_hidden)
    assert h.shape == (rkm.batch_size, rkm.n_hidden)


def test_h_to_v(rkm: RKM) -> None:
    """Test hidden to visible conversion."""
    h = torch.randint(0, 2, (rkm.batch_size, rkm.n_hidden), device=rkm.device).float()
    p_v, v = rkm.h_to_v(h)
    assert p_v.shape == (rkm.batch_size, rkm.n_visible)
    assert v.shape == (rkm.batch_size, rkm.n_visible)


def test_free_energy(rkm: RKM) -> None:
    """Test RKM free energy calculation."""
    v = torch.randint(0, 2, (rkm.batch_size, rkm.n_visible), device=rkm.device).float()
    fe = rkm.free_energy(v, beta=1.0)
    assert fe.shape[0] == rkm.batch_size
    assert not torch.any(torch.isnan(fe))


def test_rkm_with_adam_optimizer() -> None:
    """Test RKM initialization with Adam optimizer."""
    rkm_adam = RKM(model_name="test_rkm_adam", n_visible=10, n_hidden=5, optimizer="Adam", batch_size=2)
    assert hasattr(rkm_adam, "m_dW")
    assert hasattr(rkm_adam, "v_dW")


def test_rkm_with_cd() -> None:
    """Test RKM with Contrastive Divergence training."""
    rkm_cd = RKM(model_name="test_rkm_cd", n_visible=10, n_hidden=5, train_algo="CD", batch_size=2)
    assert rkm_cd.train_algo == "CD"


def test_rkm_with_pcd() -> None:
    """Test RKM with Persistent Contrastive Divergence training."""
    rkm_pcd = RKM(model_name="test_rkm_pcd", n_visible=10, n_hidden=5, train_algo="PCD", batch_size=2)
    assert rkm_pcd.train_algo == "PCD"
    assert hasattr(rkm_pcd, "persistent_chains")


def test_rkm_with_hrdm() -> None:
    """Test RKM with hidden-random training algorithm."""
    rkm_hrdm = RKM(model_name="test_rkm_hrdm", n_visible=10, n_hidden=5, train_algo="hRDM", batch_size=2)
    assert rkm_hrdm.train_algo == "hRDM"
