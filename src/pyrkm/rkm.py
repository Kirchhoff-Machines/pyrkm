from __future__ import annotations

import glob
import pickle
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from .rbm import RBM
from .utils import ensure_dir


@dataclass
class RKM(RBM):
    """A class to represent a Restricted Kirchhoff Machine (RKM).
    It is inherited from the RBM class, so look at the RBM class for info about the attributes and methods.
    Also, refer to the paper https://arxiv.org/abs/2509.15842 for more details on the RKM.

    Parameters
    ----------
    energy_type : str, optional
        The type of energy function to use (default is 'RKM').
    offset : float, optional
        Offset parameter for the energy function (default is 0.0).
    sampling : str, optional
        Sampling method to use (default is 'bernoulli').
    distribution : str, optional
        Distribution to use for sampling (default is 'gaussian').
    layer_scaled : bool, optional
        Whether to scale the layer by the number of units (default is True).
    """
    energy_type: str = 'RKM'
    offset: float = 0.0
    sampling: str = 'bernoulli'
    distribution: str = 'gaussian'
    layer_scaled: bool = True

    def __post_init__(self):
        """Initialize the RKM model after the dataclass is created."""
        super().__post_init__()
        if self.average_data is not None:
            self.v_bias = self.average_data.to(self.device).to(self.mytype)
        if self.optimizer == 'Adam':
            self.m_dW = torch.zeros_like(self.W)
            self.m_dv = torch.zeros_like(self.v_bias)
            self.m_dh = torch.zeros_like(self.h_bias)
            self.v_dW = torch.zeros_like(self.W)
            self.v_dv = torch.zeros_like(self.v_bias)
            self.v_dh = torch.zeros_like(self.h_bias)
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
        if self.train_algo == 'PCD':
            self.persistent_chains = torch.where(
                torch.rand(self.batch_size, self.n_visible) > 0.5, 1.0,
                0.0).to(self.device).to(self.mytype)
        if self.centering:
            if self.average_data.shape[0] != self.n_visible:
                print(
                    'Error: you need to provide the average of the data to center the gradient'
                )
                sys.exit()
            self.ov = self.average_data.to(self.device)
            self.oh = torch.full_like(self.h_bias, 0.5)
            self.batch_ov = torch.zeros_like(self.v_bias)
            self.batch_oh = torch.zeros_like(self.h_bias)
            self.slv = 0.01
            self.slh = 0.01
        else:
            self.ov = 0
            self.oh = 0

    def pretrain(self, pretrained_model, model_state_path='model_states/'):
        """Pretrain the model using a pretrained model.

        Parameters
        ----------
        pretrained_model : str
            Name of the pretrained model.
        model_state_path : str, optional
            Path to the model states, by default 'model_states/'.
        """
        # Check if you have model load points
        ensure_dir(model_state_path)
        filename_list = glob.glob(model_state_path +
                                  '{}_t*.pkl'.format(pretrained_model))
        if len(filename_list) > 0:
            all_loadpoints = sorted([
                int(x.split('_t')[-1].split('.pkl')[0]) for x in filename_list
            ])
            last_epoch = all_loadpoints[-1]
            print('** Using as pretraining model {} at epoch {}'.format(
                pretrained_model, last_epoch),
                  flush=True)
            with open(
                    model_state_path +
                    '{}_t{}.pkl'.format(pretrained_model, last_epoch),
                    'rb') as file:
                temp_model = pickle.load(file)
                # *** Import pretrained parameters
                self.W = temp_model.W.to(self.mytype)
                self.h_bias = temp_model.h_bias.to(self.mytype)
                self.v_bias = temp_model.v_bias.to(self.mytype)
        else:
            print('** No load points for {}'.format(pretrained_model),
                  flush=True)

    def v_to_h(self, v, beta=None):
        """Convert visible units to hidden units.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        tuple of torch.Tensor
            Probabilities and sampled hidden units.
        """
        if beta is None:
            beta = self.model_beta
        if self.energy_type == 'RKM':
            effective_h_bias = self.h_bias + 0.5 * self.offset * (
                (torch.abs(self.h_bias) - self.h_bias) / self.g_h +
                (torch.abs(self.W) - self.W).sum(dim=1))
            num = torch.mm(v, self.W_t) + effective_h_bias
            den = torch.abs(
                self.W).sum(dim=1) + torch.abs(self.h_bias) / self.g_h
            h_analog = num / den

            if self.sampling == 'bernoulli':
                if self.layer_scaled:
                    p_h = torch.sigmoid(beta * self.n_visible * h_analog)
                    h = torch.bernoulli(p_h)
                else:
                    p_h = torch.sigmoid(beta * h_analog)
                    h = torch.bernoulli(p_h)
            elif self.sampling == 'multi-threshold':
                if self.distribution == 'gaussian':
                    t = torch.randn_like(h_analog,
                                         dtype=self.mytype,
                                         device=self.device) * 1 / beta
                else:
                    t = (torch.rand_like(
                        h_analog, dtype=self.mytype, device=self.device) * 2 -
                         1) * 1 / beta
                p_h = h_analog
                if self.layer_scaled:
                    h = (p_h > t / self.n_visible).to(v.dtype)
                else:
                    h = (p_h > t).to(v.dtype)
            elif self.sampling == 'single-threshold':
                if self.distribution == 'gaussian':
                    t = torch.randn(
                        1, dtype=self.mytype,
                        device=self.device) * 1 / beta * torch.ones_like(
                            h_analog, dtype=self.mytype, device=self.device)
                else:
                    t = (torch.rand(1, dtype=self.mytype, device=self.device) *
                         2 - 1) * 1 / beta * torch.ones_like(
                             h_analog, dtype=self.mytype, device=self.device)
                p_h = h_analog
                if self.layer_scaled:
                    h = (p_h > t / self.n_visible).to(v.dtype)
                else:
                    h = (p_h > t).to(v.dtype)
            return p_h, h
        else:
            return super().v_to_h(v, beta)

    def h_to_v(self, h, beta=None):
        """Convert hidden units to visible units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        tuple of torch.Tensor
            Probabilities and sampled visible units.
        """
        if beta is None:
            beta = self.model_beta
        if self.energy_type == 'RKM':
            effective_v_bias = self.v_bias + 0.5 * self.offset * (
                (torch.abs(self.v_bias) - self.v_bias) / self.g_v +
                (torch.abs(self.W) - self.W).sum(dim=0))
            num = torch.mm(h, self.W) + effective_v_bias
            den = torch.abs(
                self.W).sum(dim=0) + torch.abs(self.v_bias) / self.g_v
            v_analog = num / den

            if self.sampling == 'bernoulli':
                if self.layer_scaled:
                    p_v = torch.sigmoid(beta * self.n_hidden * v_analog)
                    v = torch.bernoulli(p_v)
                else:
                    p_v = torch.sigmoid(beta * v_analog)
                    v = torch.bernoulli(p_v)
            elif self.sampling == 'multi-threshold':
                if self.distribution == 'gaussian':
                    t = torch.randn_like(v_analog,
                                         dtype=self.mytype,
                                         device=self.device) * 1 / beta
                else:
                    t = (torch.rand_like(
                        v_analog, dtype=self.mytype, device=self.device) * 2 -
                         1) * 1 / beta
                p_v = v_analog
                if self.layer_scaled:
                    v = (p_v > t / self.n_hidden).to(h.dtype)
                else:
                    v = (p_v > t).to(h.dtype)
            elif self.sampling == 'single-threshold':
                if self.distribution == 'gaussian':
                    t = torch.randn(
                        1, dtype=self.mytype,
                        device=self.device) * 1 / beta * torch.ones_like(
                            v_analog, dtype=self.mytype, device=self.device)
                else:
                    t = (torch.rand(1, dtype=self.mytype, device=self.device) *
                         2 - 1) * 1 / beta * torch.ones_like(
                             v_analog, dtype=self.mytype, device=self.device)
                p_v = v_analog
                if self.layer_scaled:
                    v = (p_v > t / self.n_hidden).to(h.dtype)
                else:
                    v = (p_v > t).to(h.dtype)
            return p_v, v
        else:
            return super().h_to_v(h, beta)

    def Deterministic_v_to_h(self, v, beta):
        """Deterministically convert visible units to hidden units.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : float
            Inverse temperature parameter.

        Returns
        -------
        tuple of torch.Tensor
            Deterministic hidden units.
        """
        h = (self.delta_eh(v) > 0).to(v.dtype)
        return h, h

    def Deterministic_h_to_v(self, h, beta):
        """Deterministically convert hidden units to visible units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : float
            Inverse temperature parameter.

        Returns
        -------
        tuple of torch.Tensor
            Deterministic visible units.
        """
        v = (self.delta_ev(h) > 0).to(h.dtype)
        return v, v

    def Bernoulli_v_to_h(self, v, beta):
        """Convert visible units to hidden units using Bernoulli sampling.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : float
            Inverse temperature parameter.

        Returns
        -------
        tuple of torch.Tensor
            Probabilities and sampled hidden units.
        """
        p_h = self._prob_h_given_v(v, beta)
        sample_h = torch.bernoulli(p_h)
        return p_h, sample_h

    def Bernoulli_h_to_v(self, h, beta):
        """Convert hidden units to visible units using Bernoulli sampling.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : float
            Inverse temperature parameter.

        Returns
        -------
        tuple of torch.Tensor
            Probabilities and sampled visible units.
        """
        p_v = self._prob_v_given_h(h, beta)
        sample_v = torch.bernoulli(p_v)
        return p_v, sample_v

    def _free_energy_hopfield(self, v, beta=None):
        """Compute the free energy of the Hopfield network.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        torch.Tensor
            Free energy of the Hopfield network.
        """
        if beta is None:
            beta = self.model_beta
        vbias_term = torch.mv(v, self.v_bias) * beta
        wx_b = torch.mm(v, self.W.t()) + self.h_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b * beta)), axis=1)
        return -hidden_term - vbias_term

    def _energy_hopfield(self, v, h):
        """Compute the energy of the Hopfield network.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        h : torch.Tensor
            Hidden units.

        Returns
        -------
        torch.Tensor
            Energy of the Hopfield network.
        """
        energy = -(torch.mm(v, self.W.t()) * h).sum(1) - torch.mv(
            v, self.v_bias) - torch.mv(h, self.h_bias)
        return energy

    def forward(self, v, k, beta=None):
        """Perform a forward pass through the network.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        k : int
            Number of Gibbs sampling steps.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        torch.Tensor
            Reconstructed visible units.
        """
        if beta is None:
            beta = self.model_beta
        pre_h1, h1 = self.v_to_h(v, beta)
        h_ = h1
        for _ in range(k):
            pre_v_, v_ = self.h_to_v(h_, beta)
            pre_h_, h_ = self.v_to_h(v_, beta)
        return v_

    def train(self,
              train_data,
              test_data=[],
              print_error=False,
              print_test_error=False,
              model_state_path='model_states/',
              print_every=100):
        """Train the model using the given data and parameters.

        Parameters
        ----------
        train_data : torch.Tensor
            Training data.
        test_data : torch.Tensor, optional
            Test data, by default [].
        print_error : bool, optional
            Whether to print training error, by default False.
        print_test_error : bool, optional
            Whether to print test error, by default False.
        model_state_path : str, optional
            Path to save model states, by default 'model_states/'.
        print_every : int, optional
            Frequency of printing progress, by default 100.
        """
        while self.epoch < self.max_epochs:
            self.W_t = self.W.t()

            for _, v_data in enumerate(train_data):

                start_time = time.time()
                # restart the power
                self.power_f = 0
                self.power_b = 0

                # For the positive phase, we use the data and propagate to the hidden nodes
                h_data = self.v_to_h(v_data)[1]
                p_f = self.power_forward(v_data)
                self.power_f += p_f.mean()
                self.energy += p_f.sum()

                # For the negative phase, it depends on the training algorithm
                if self.train_algo == 'PCD':
                    # Update the chain after every batch
                    # self.persistent_chains = self.forward(
                    #     self.persistent_chains, self.k)
                    v_model = self.persistent_chains
                    # print(
                    #     'Warning: No physical measurements are implemented for PCD training algorithm.'
                    # )
                    for _ in range(self.k):
                        h_model = self.v_to_h(v_model)[1]
                        p_f = self.power_forward(v_model)
                        self.power_f += p_f.mean()

                        v_model = self.h_to_v(h_model)[1]
                        p_b = self.power_backward(h_model)
                        self.power_b += p_b.mean()

                        self.energy += p_f.sum() + p_b.sum()

                    self.persistent_chains = v_model
                elif self.train_algo == 'RDM':
                    v_model = torch.randint(high=2,
                                            size=(self.batch_size,
                                                  self.n_visible),
                                            device=self.device,
                                            dtype=self.mytype)
                    v_model = self.forward(v_model, self.k)
                    print(
                        'Warning: No physical measurements are implemented for RDM training algorithm.'
                        + 'Use hRDM or vRDM instead.')
                elif self.train_algo == 'CD':
                    v_model = v_data
                    for _ in range(self.k):
                        h_model = self.v_to_h(v_model, self.model_beta)[1]
                        p_f = self.power_forward(v_model)
                        self.power_f += p_f.mean()

                        v_model = self.h_to_v(h_model, self.model_beta)[1]
                        p_b = self.power_backward(h_model)
                        self.power_b += p_b.mean()

                        self.energy += p_f.sum() + p_b.sum()
                elif self.train_algo == 'vRDM':
                    # visible RDM
                    v_model = torch.randint(high=2,
                                            size=(self.batch_size,
                                                  self.n_visible),
                                            device=self.device,
                                            dtype=self.mytype)
                    for _ in range(self.k):
                        h_model = self.v_to_h(v_model, self.model_beta)[1]
                        p_f = self.power_forward(v_model)
                        self.power_f += p_f.mean()

                        v_model = self.h_to_v(h_model, self.model_beta)[1]
                        p_b = self.power_backward(h_model)
                        self.power_b += p_b.mean()

                        self.energy += p_f.sum() + p_b.sum()
                elif self.train_algo == 'hRDM':
                    # hidden RDM
                    h_model = torch.randint(high=2,
                                            size=(self.batch_size,
                                                  self.n_hidden),
                                            device=self.device,
                                            dtype=self.mytype)
                    v_model = self.h_to_v(h_model, self.model_beta)[1]
                    p_b = self.power_backward(h_model)
                    self.power_b += p_b.mean()

                    self.energy += p_b.sum()

                    for _ in range(self.k - 1):
                        h_model = self.v_to_h(v_model, self.model_beta)[1]
                        p_f = self.power_forward(v_model)
                        self.power_f += p_f.mean()

                        v_model = self.h_to_v(h_model, self.model_beta)[1]
                        p_b = self.power_backward(h_model)
                        self.power_b += p_b.mean()

                        self.energy += p_f.sum() + p_b.sum()

                # Apply centering
                if self.centering:
                    self.batch_ov = v_data.mean(0)
                    self.batch_oh = h_data.mean(0)
                    # update with sliding
                    self.ov = (1 -
                               self.slv) * self.ov + self.slv * self.batch_ov
                    self.oh = (1 -
                               self.slh) * self.oh + self.slh * self.batch_oh

                # Compute gradients
                dEdW_data, dEdv_bias_data, dEdh_bias_data = self.derivatives(
                    v_data, h_data)
                dEdW_model, dEdv_bias_model, dEdh_bias_model = self.derivatives(
                    v_model, h_model)

                # Average over batch
                dEdW_data = torch.mean(dEdW_data, dim=0)
                dEdv_bias_data = torch.mean(dEdv_bias_data, dim=0)
                dEdh_bias_data = torch.mean(dEdh_bias_data, dim=0)
                dEdW_model = torch.mean(dEdW_model, dim=0)
                dEdv_bias_model = torch.mean(dEdv_bias_model, dim=0)
                dEdh_bias_model = torch.mean(dEdh_bias_model, dim=0)

                # Update weights and biases
                if self.optimizer == 'Adam':
                    self.Adam_update(self.epoch + 1, dEdW_data, dEdW_model,
                                     dEdv_bias_data, dEdv_bias_model,
                                     dEdh_bias_data, dEdh_bias_model)
                elif self.optimizer == 'SGD':
                    self.SGD_update(dEdW_data, dEdW_model, dEdv_bias_data,
                                    dEdv_bias_model, dEdh_bias_data,
                                    dEdh_bias_model)

                self.after_step_keepup()

                # compute new relaxation times
                self.relax_t_f, self.relax_t_b = self.relaxation_times()

                self.epoch += 1

                # Store the model state
                if self.epoch in self.t_to_save:
                    ensure_dir(model_state_path)
                    with open(
                            model_state_path +
                            '{}_t{}.pkl'.format(self.model_name, self.epoch),
                            'wb') as file:
                        pickle.dump(self, file)

                if self.epoch % print_every == 0:
                    t = time.time() - start_time
                    if print_error:
                        v_model = self.forward(v_data, 1)
                        rec_error_train = ((v_model -
                                            v_data)**2).mean(1).mean(0)
                        if not print_test_error:
                            print('Epoch: %d , train-err %.5g , time: %f' %
                                  (self.epoch, rec_error_train, t),
                                  flush=True)
                        else:
                            t_model = self.forward(test_data, 1)
                            rec_error_test = ((t_model -
                                               test_data)**2).mean(1).mean(0)
                            print(
                                'Epoch: %d , Test-err %.5g , train-err %.5g , time: %f'
                                % (self.epoch, rec_error_test, rec_error_train,
                                   t),
                                flush=True)
                    else:
                        print('Epoch: %d , time: %f' % (self.epoch, t),
                              flush=True)

        print('*** Training finished', flush=True)

    def after_step_keepup(self):
        """Perform operations after each training step."""
        # self.W_t = self.W.t() # already done in clip_weights
        self.clip_weights()
        self.clip_bias()

    def SGD_update(self, dEdW_data, dEdW_model, dEdv_bias_data,
                   dEdv_bias_model, dEdh_bias_data, dEdh_bias_model):
        """Update the model parameters using SGD.

        Parameters
        ----------
        dEdW_data : torch.Tensor
            Gradient of the weights from data.
        dEdW_model : torch.Tensor
            Gradient of the weights from the model.
        dEdv_bias_data : torch.Tensor
            Gradient of the visible biases from data.
        dEdv_bias_model : torch.Tensor
            Gradient of the visible biases from the model.
        dEdh_bias_data : torch.Tensor
            Gradient of the hidden biases from data.
        dEdh_bias_model : torch.Tensor
            Gradient of the hidden biases from the model.
        """
        # Gradients
        dW = -dEdW_data + dEdW_model
        dv = -dEdv_bias_data + dEdv_bias_model
        dh = -dEdh_bias_data + dEdh_bias_model
        if self.centering:
            dv = dv - torch.matmul(self.oh, dW)
            dh = dh - torch.matmul(self.ov, dW.t())
        # Add regularization term
        if self.regularization == 'l2':
            dW -= self.l2 * 2 * self.W
            dv -= self.l2 * 2 * self.v_bias
            dh -= self.l2 * 2 * self.h_bias
        elif self.regularization == 'l1':
            dW -= self.l1 * torch.sign(self.W)
            dv -= self.l1 * torch.sign(self.v_bias)
            dh -= self.l1 * torch.sign(self.h_bias)
        # Update parameters in-place
        # # and clip
        # gnorm = torch.norm(dW) + torch.norm(dv) + torch.norm(dh)
        # myclip = (self.lr*10.) / gnorm if gnorm > 10 else self.lr
        self.W.add_(self.lr * dW)
        self.v_bias.add_(self.lr * dv)
        self.h_bias.add_(self.lr * dh)

    def Adam_update(self, t, dEdW_data, dEdW_model, dEdv_bias_data,
                    dEdv_bias_model, dEdh_bias_data, dEdh_bias_model):
        """Update the model parameters using Adam optimizer.

        Parameters
        ----------
        t : int
            Current time step.
        dEdW_data : torch.Tensor
            Gradient of the weights from data.
        dEdW_model : torch.Tensor
            Gradient of the weights from the model.
        dEdv_bias_data : torch.Tensor
            Gradient of the visible biases from data.
        dEdv_bias_model : torch.Tensor
            Gradient of the visible biases from the model.
        dEdh_bias_data : torch.Tensor
            Gradient of the hidden biases from data.
        dEdh_bias_model : torch.Tensor
            Gradient of the hidden biases from the model.
        """
        # Gradients
        dW = -dEdW_data + dEdW_model
        dv = -dEdv_bias_data + dEdv_bias_model
        dh = -dEdh_bias_data + dEdh_bias_model
        if self.centering:
            dv = dv - torch.matmul(self.oh, dW)
            dh = dh - torch.matmul(self.ov, dW.t())
        # Add regularization term
        if self.regularization == 'l2':
            dW += self.l2 * 2 * self.W
            dv += self.l2 * 2 * self.v_bias
            dh += self.l2 * 2 * self.h_bias
        elif self.regularization == 'l1':
            dW += self.l1 * torch.sign(self.W)
            dv += self.l1 * torch.sign(self.v_bias)
            dh += self.l1 * torch.sign(self.h_bias)
        # momentum beta1
        self.m_dW = self.beta1 * self.m_dW + (1 - self.beta1) * dW
        self.m_dv = self.beta1 * self.m_dv + (1 - self.beta1) * dv
        self.m_dh = self.beta1 * self.m_dh + (1 - self.beta1) * dh
        # momentum beta2
        self.v_dW = self.beta2 * self.v_dW + (1 - self.beta2) * (dW**2)
        self.v_dv = self.beta2 * self.v_dv + (1 - self.beta2) * (dv**2)
        self.v_dh = self.beta2 * self.v_dh + (1 - self.beta2) * (dh**2)
        # bias correction
        m_dW_corr = self.m_dW / (1 - self.beta1**t)
        m_dv_corr = self.m_dv / (1 - self.beta1**t)
        m_dh_corr = self.m_dh / (1 - self.beta1**t)
        v_dW_corr = self.v_dW / (1 - self.beta2**t)
        v_dv_corr = self.v_dv / (1 - self.beta2**t)
        v_dh_corr = self.v_dh / (1 - self.beta2**t)
        # Update
        self.W = self.W + self.lr * (m_dW_corr /
                                     (torch.sqrt(v_dW_corr) + self.epsilon))
        self.v_bias = self.v_bias + self.lr * (
            m_dv_corr / (torch.sqrt(v_dv_corr) + self.epsilon))
        self.h_bias = self.h_bias + self.lr * (
            m_dh_corr / (torch.sqrt(v_dh_corr) + self.epsilon))

    def reconstruct(self, data, k):
        """Reconstruct the visible units from the data.

        Parameters
        ----------
        data : array-like
            Input data.
        k : int
            Number of Gibbs sampling steps.

        Returns
        -------
        tuple of numpy.ndarray
            Original and reconstructed visible units.
        """
        data = torch.Tensor(data).to(self.device).to(self.mytype)
        v_model = self.forward(data, k)
        return data.detach().cpu().numpy(), v_model.detach().cpu().numpy()

    def generate(self,
                 n_samples,
                 k,
                 h_binarized=True,
                 from_visible=True,
                 beta=None):
        """Generate samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        k : int
            Number of Gibbs sampling steps.
        h_binarized : bool, optional
            Whether to binarize hidden units, by default True.
        from_visible : bool, optional
            Whether to generate from visible units, by default True.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        numpy.ndarray
            Generated samples.
        """
        if beta is None:
            beta = self.model_beta
        if from_visible:
            v = torch.randint(high=2,
                              size=(n_samples, self.n_visible),
                              device=self.device,
                              dtype=self.mytype)
        else:
            if h_binarized:
                h = torch.randint(high=2,
                                  size=(n_samples, self.n_hidden),
                                  device=self.device,
                                  dtype=self.mytype)
            else:
                h = torch.rand(n_samples,
                               self.n_hidden,
                               device=self.device,
                               dtype=self.mytype)
            _, v = self.h_to_v(h)
        v_model = self.forward(v, k, beta)
        return v_model.detach().cpu().numpy()

    def clip_weights(self):
        """Clip the weights to be within the specified range."""
        self.W = torch.clip(self.W, self.min_W, self.max_W)
        self.W_t = self.W.t()

    def clip_bias(self):
        """Clip the biases to be within the specified range."""
        self.v_bias = torch.clip(self.v_bias, self.min_W, self.max_W)
        self.h_bias = torch.clip(self.h_bias, self.min_W, self.max_W)

    ########################################################
    ### Detivatives for the hopfield and RKM energies
    ########################################################

    def derivatives(self, v, h):
        """Compute the derivatives for the specified energy type.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        h : torch.Tensor
            Hidden units.

        Returns
        -------
        tuple of torch.Tensor
            Gradients of the weights, visible biases, and hidden biases.
        """
        if self.energy_type == 'hopfield' or self.energy_type == 'RKM':
            return self.derivatives_hopfield(v, h)
        else:
            # exit error
            print('Error: derivatives not implemented for this energy type')
            sys.exit()

    def derivatives_hopfield(self, v, h):
        """Compute the derivatives for the Hopfield energy.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        h : torch.Tensor
            Hidden units.

        Returns
        -------
        tuple of torch.Tensor
            Gradients of the weights, visible biases, and hidden biases.
        """
        # h has shape (N, n_h) and v has shape (N, n_v), we want result to have shape (N, n_h, n_v)
        if self.centering:
            dEdW = -torch.einsum('ij,ik->ijk', h - self.oh, v - self.ov)
        else:
            dEdW = -torch.einsum('ij,ik->ijk', h, v)
        dEdv_bias = -v
        dEdh_bias = -h
        return dEdW, dEdv_bias, dEdh_bias

    ########################################
    ### Hopfield specific functions
    ########################################

    def _prob_h_given_v(self, v, beta=None):
        """Compute the probability of hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        torch.Tensor
            Probability of hidden units.
        """
        if beta is None:
            beta = self.model_beta
        return torch.sigmoid(beta * self.delta_eh(v))

    def _prob_v_given_h(self, h, beta=None):
        """Compute the probability of visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : float, optional
            Inverse temperature parameter, by default None.

        Returns
        -------
        torch.Tensor
            Probability of visible units.
        """
        if beta is None:
            beta = self.model_beta
        return torch.sigmoid(beta * self.delta_ev(h))

    def delta_eh(self, v):
        """Compute the change in energy for hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.

        Returns
        -------
        torch.Tensor
            Change in energy for hidden units.
        """
        if self.energy_type == 'hopfield':
            return self._delta_eh_hopfield(v)
        else:
            # exit error
            print('Error: delta_eh not implemented for this energy type')
            sys.exit()

    def delta_ev(self, h):
        """Compute the change in energy for visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.

        Returns
        -------
        torch.Tensor
            Change in energy for visible units.
        """
        if self.energy_type == 'hopfield':
            return self._delta_ev_hopfield(h)
        else:
            # exit error
            print('Error: delta_ev not implemented for this energy type')
            sys.exit()

    # **** Hopfield transfer functions
    def _delta_eh_hopfield(self, v):
        """Compute the change in energy for hidden units in the Hopfield network.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.

        Returns
        -------
        torch.Tensor
            Change in energy for hidden units.
        """
        return torch.mm(v, self.W_t) + self.h_bias

    def _delta_ev_hopfield(self, h):
        """Compute the change in energy for visible units in the Hopfield network.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.

        Returns
        -------
        torch.Tensor
            Change in energy for visible units.
        """
        return torch.mm(h, self.W) + self.v_bias

    ########################################
    ### Plotting functions
    ########################################

    def plot_weights(self, t):
        """Plot the weights of the model.

        Parameters
        ----------
        t : int
            Current epoch.
        """
        Ndata = self.W.shape[0]
        # Reshape the matrix into a 3D array
        data_3d = self.W.detach().cpu().numpy().reshape(Ndata, 28, 28)
        # Determine the number of rows and columns for the subplot grid
        num_rows = int(np.ceil(np.sqrt(Ndata)))
        num_cols = int(np.ceil(Ndata / num_rows))
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(nrows=num_rows,
                               ncols=num_cols,
                               figsize=(10, 10))
        # Iterate over the submatrices and plot them
        for i in range(Ndata):
            row = i // num_cols
            col = i % num_cols
            ax[row, col].imshow(data_3d[i], cmap='magma')
            ax[row, col].axis('off')
        # Remove empty subplots if the number of submatrices doesn't fill the entire grid
        if num_rows * num_cols > Ndata:
            for i in range(Ndata, num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                fig.delaxes(ax[row, col])
        # Adjust the spacing between subplots
        plt.suptitle('Weights epoch {}'.format(t))
        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9)
        # Get the minimum and maximum values from the data
        vmin = np.min(self.W.detach().cpu().numpy())
        vmax = np.max(self.W.detach().cpu().numpy())
        # Create a dummy image for the colorbar
        dummy_img = np.zeros((1, 1))  # Dummy image with all zeros
        # Add a colorbar using the dummy image as the mappable
        cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Position of the colorbar
        plt.colorbar(plt.imshow(dummy_img, cmap='magma', vmin=vmin, vmax=vmax),
                     cax=cax)
        # Adjust the height of the colorbar axes to match the height of the figure
        cax.set_aspect('auto')

    # ** Plotting visible biases with imshow (as one matrix of the same size as the weights)
    def plot_visible_bias(self, t):
        """Plot the visible biases of the model.

        Parameters
        ----------
        t : int
            Current epoch.
        """
        # Reshape the vector into a 2D array
        data_2d = self.v_bias.detach().cpu().numpy().reshape(28, 28)
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(5, 5))
        # Plot the 2D array
        im = ax.imshow(data_2d, cmap='magma')
        # Add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Values', rotation=-90, va='bottom')
        # Add title and labels
        ax.set_title('Visible Biases epoch {}'.format(t))
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')

    # ** Plotting bias
    def plot_bias(self, t):
        """Plot the biases of the model.

        Parameters
        ----------
        t : int
            Current epoch.
        """
        h_bias = self.h_bias.detach().cpu().numpy()
        v_bias = self.v_bias.detach().cpu().numpy()
        # Set up the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Plot histogram for hidden biases
        ax1.hist(h_bias, bins=20, color='blue', edgecolor='black')
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Hidden Biases epoch {}'.format(t))
        # Plot histogram for visible biases
        ax2.hist(v_bias, bins=20, color='red', edgecolor='black')
        ax2.set_xlabel('Values')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Visible Biases epoch {}'.format(t))
        # Adjust layout for better readability
        plt.tight_layout()

    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PERFORMANCE METRICS

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def power_forward(self, v):
        """Computes the power dissipated by the RKM in the forward pass.

        Parameters
        ----------
        v : torch.Tensor
            Visible units, shape (N, n_v).

        Returns
        -------
        torch.Tensor
            Power dissipated by the RKM, shape (N,).
        """
        effective_h_bias = self.h_bias + 0.5 * self.offset * (
            (torch.abs(self.h_bias) - self.h_bias) / self.g_h +
            (torch.abs(self.W) - self.W).sum(dim=1))
        num = torch.mm(v, self.W_t) + effective_h_bias
        den = torch.abs(self.W).sum(dim=1) + torch.abs(self.h_bias) / self.g_h
        h_analog = num / den

        W_t = self.W_t
        abs_W_t = torch.abs(self.W_t)
        h_bias = self.h_bias
        abs_h_bias = torch.abs(self.h_bias)

        power_forward = (
            -torch.einsum('ni,ij,nj->n', v, W_t, h_analog) + 0.5 *
            (torch.einsum('ni,ij->n', v**2, abs_W_t) +
             torch.einsum('ij,nj->n', abs_W_t, h_analog**2)) -
            torch.einsum('j,nj->n', effective_h_bias, h_analog) +
            (0.5 / self.g_h) *
            torch.einsum('j,nj->n', abs_h_bias, h_analog**2 + self.g_h**2))

        if self.offset != 0:
            power_forward = power_forward + (+torch.einsum(
                'ni,ij->n',
                (-2 * self.offset * v + self.offset**2 / 4), abs_W_t - W_t) +
                                             (self.offset**2 /
                                              (4 * self.g_h) - self.offset / 2)
                                             * torch.sum(abs_h_bias - h_bias))

        return power_forward

    def power_backward(self, h):
        """Computes the power dissipated by the RKM in the backward pass.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units, shape (N, n_h).

        Returns
        -------
        torch.Tensor
            Power dissipated by the RKM, shape (N,).
        """
        effective_v_bias = self.v_bias + 0.5 * self.offset * (
            (torch.abs(self.v_bias) - self.v_bias) / self.g_v +
            (torch.abs(self.W) - self.W).sum(dim=0))
        num = torch.mm(h, self.W) + effective_v_bias
        den = torch.abs(self.W).sum(dim=0) + torch.abs(self.v_bias) / self.g_v
        v_analog = num / den

        W_t = self.W_t
        abs_W_t = torch.abs(self.W_t)
        v_bias = self.v_bias
        abs_v_bias = torch.abs(self.v_bias)

        power_backward = (
            -torch.einsum('ni,ij,nj->n', v_analog, W_t, h) + 0.5 *
            (torch.einsum('ni,ij->n', v_analog**2, abs_W_t) +
             torch.einsum('ij,nj->n', abs_W_t, h**2)) -
            torch.einsum('i,ni->n', effective_v_bias, v_analog) +
            (0.5 / self.g_v) *
            torch.einsum('i,ni->n', abs_v_bias, v_analog**2 + self.g_v**2))

        if self.offset != 0:
            power_backward = power_backward + (
                +torch.einsum('ij,nj->n', abs_W_t - W_t,
                              (-2 * self.offset * h + self.offset**2 / 4)) +
                (self.offset**2 / (4 * self.g_v) - self.offset / 2) *
                torch.sum(abs_v_bias - v_bias))

        return power_backward

    def av_power_forward(self, v):
        """Computes the average power dissipated by the RKM in the forward pass.

        Parameters
        ----------
        v : torch.Tensor
            Visible units, shape (N, n_v).

        Returns
        -------
        torch.Tensor
            Average power dissipated by the RKM.
        """
        return self.power_forward(v).mean()

    def av_power_backward(self, h):
        """Computes the average power dissipated by the RKM in the backward pass.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units, shape (N, n_h).

        Returns
        -------
        torch.Tensor
            Average power dissipated by the RKM.
        """
        return self.power_backward(h).mean()

    def relaxation_times(self):
        """Computes the relaxation times of the RKM in the forward and backward pass.

        Returns
        -------
        tuple of torch.Tensor
            t_forward : relaxation times of the RKM in the forward pass, shape (n_v,).
            t_backward : relaxation times of the RKM in the backward pass, shape (n_h,).
        """
        t_forward = 2 / (torch.abs(self.W).sum(dim=1) +
                         torch.abs(self.h_bias) / self.g_h)
        t_backward = 2 / (torch.abs(self.W).sum(dim=0) +
                          torch.abs(self.v_bias) / self.g_v)

        return t_forward, t_backward
