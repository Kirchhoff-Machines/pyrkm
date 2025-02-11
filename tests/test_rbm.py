from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrkm.rbm import RBM


@pytest.fixture(scope='module')
def rbm():
    return RBM(model_name='test_rbm',
               n_visible=6,
               n_hidden=3,
               k=1,
               lr=0.01,
               max_epochs=10,
               energy_type='hopfield',
               optimizer='SGD',
               regularization=False,
               l1_factor=0,
               l2_factor=1e-3,
               g_v=0.5,
               g_h=0.5,
               batch_size=1,
               train_algo='vRDM',
               centering=False,
               average_data=None,
               model_beta=1,
               mytype=torch.float32,
               min_W=-10,
               max_W=10)


def test_initialization(rbm):
    assert rbm.W.shape == (rbm.n_hidden, rbm.n_visible)
    assert rbm.v_bias.shape == (rbm.n_visible, )
    assert rbm.h_bias.shape == (rbm.n_hidden, )
    assert rbm.device.type in ['cpu', 'cuda']


def test_forward_pass(rbm):
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible)).float()
    v_model = rbm.forward(v, rbm.k)
    assert v_model.shape == v.shape


def test_reconstruction(rbm):
    data = np.random.randint(0, 2, (rbm.batch_size, rbm.n_visible))
    original, reconstructed = rbm.reconstruct(data, rbm.k)
    assert original.shape == reconstructed.shape


def test_train_step(rbm):
    train_data = [torch.randint(0, 2, (rbm.batch_size, rbm.n_visible)).float()]
    rbm.train(train_data,
              print_error=False,
              print_test_error=False,
              print_every=1)
    assert rbm.epoch > 0


def test_clip_weights(rbm):
    rbm.W = torch.randn((rbm.n_hidden, rbm.n_visible)) * 20
    rbm.clip_weights()
    assert torch.all(rbm.W <= rbm.max_W)
    assert torch.all(rbm.W >= rbm.min_W)


def test_clip_bias(rbm):
    rbm.v_bias = torch.randn((rbm.n_visible, )) * 20
    rbm.h_bias = torch.randn((rbm.n_hidden, )) * 20
    rbm.clip_bias()
    assert torch.all(rbm.v_bias <= rbm.max_W)
    assert torch.all(rbm.v_bias >= rbm.min_W)
    assert torch.all(rbm.h_bias <= rbm.max_W)
    assert torch.all(rbm.h_bias >= rbm.min_W)


def test_prob_h_given_v(rbm):
    v = torch.randint(0, 2, (rbm.batch_size, rbm.n_visible)).float()
    p_h = rbm._prob_h_given_v(v)
    assert p_h.shape == (rbm.batch_size, rbm.n_hidden)


def test_prob_v_given_h(rbm):
    h = torch.randint(0, 2, (rbm.batch_size, rbm.n_hidden)).float()
    p_v = rbm._prob_v_given_h(h)
    assert p_v.shape == (rbm.batch_size, rbm.n_visible)
