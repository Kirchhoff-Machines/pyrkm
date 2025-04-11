from __future__ import annotations

import glob
import pickle
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import ensure_dir


@dataclass
class RBM:
    """A class to represent a Restricted Boltzmann Machine (RBM).

    Parameters
    ----------
    model_name : str
        The name of the model.
    n_visible : int
        The number of visible units.
    n_hidden : int
        The number of hidden units.
    k : int, optional
        The number of Gibbs sampling steps (default is 1).
    lr : float, optional
        The learning rate.
    max_epochs : int, optional
        The maximum number of training epochs.
    energy_type : str, optional
        The type of energy function to use (default is 'hopfield').
    optimizer : str, optional
        The optimizer to use (default is 'SGD', but also Adam is available).
    batch_size : int, optional
        The batch size for training (default is 1).
    train_algo : str, optional
        The training algorithm to use between Contrastive Divergence (CD), Persistent
        Contrastive Divergence (PCD), visible-random (default, vRDM), hidden-random (hRDM).
    average_data : torch.tensor, optional
        The average data tensor for centering and initialization (default is None).
    model_beta : int, optional
        The inverse temperature parameter (default is 1).
    mytype : type, optional
        The data type for tensors (default is torch.float32).
    min_W : float, optional
        The minimum weight value used for clipping (default is -10).
    max_W : float, optional
        The maximum weight value used for clipping (default is 10).
    regularization : bool, optional
        Whether to use L1+L2 regularization (default is False).
    l1_factor : float, optional
        The L1 regularization factor.
    l2_factor : float, optional
        The L2 regularization factor.
    centering : bool, optional
        Whether to use centering (default is False).
    g_v : float, optional
        The visible unit gain, required for gradient centering (default is 0.5).
    g_h : float, optional
        The hidden unit gain, required for gradient centering (default is 0.5).
    """
    model_name: str
    n_visible: int
    n_hidden: int
    k: int = 1
    lr: float = 0.001
    max_epochs: int = 200000
    energy_type: str = 'hopfield'
    optimizer: str = 'SGD'
    regularization: bool = False
    l1_factor: float = 0
    l2_factor: float = 1e-3
    g_v: float = 0.5
    g_h: float = 0.5
    batch_size: int = 1
    train_algo: str = 'vRDM'
    centering: bool = False
    average_data: torch.tensor = None
    model_beta: int = 1
    mytype: type = torch.float32
    min_W: float = -10
    max_W: float = 10

    def __post_init__(self):
        """Initializes the RBM model by setting up the device,
        parameters, optimizer, persistent chains,
        centering, save points, and physical performance."""
        print(f'*** Initializing {self.model_name}')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(self.mytype)
        print(f'The model is working on the following device: {self.device}')
        self.epoch = 0
        self._initialize_parameters()
        self._initialize_optimizer()
        self._initialize_persistent_chains()
        self._initialize_centering()
        self._initialize_save_points()
        self._initialize_physical_performance()

    def _initialize_parameters(self):
        """Initializes the parameters of the RBM model, including weights and biases."""
        self.W = torch.randn(
            (self.n_hidden, self.n_visible),
            dtype=self.mytype,
            device=self.device) * 0.1 / np.sqrt(self.n_visible)
        if self.average_data is not None:
            self.v_bias = self.average_data.to(self.device).to(self.mytype)
        else:
            self.v_bias = torch.randn((self.n_visible, ),
                                      dtype=self.mytype,
                                      device=self.device)
        self.h_bias = torch.zeros((self.n_hidden, ),
                                  dtype=self.mytype,
                                  device=self.device)
        self._clip_parameters()

    def _clip_parameters(self):
        """Clips the weights and biases of the RBM model to be within specified bounds."""
        self.clip_weights()
        self.clip_bias()

    def _initialize_optimizer(self):
        """Initializes the optimizer parameters if the optimizer is Adam."""
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

    def _initialize_persistent_chains(self):
        """Initializes the persistent chains for the PCD training algorithm."""
        if self.train_algo == 'PCD':
            self.persistent_chains = torch.where(
                torch.rand(self.batch_size, self.n_visible) > 0.5, 1.0,
                0.0).to(self.device).to(self.mytype)

    def _initialize_centering(self):
        """Initializes the centering parameters if centering is enabled."""
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

    def _initialize_save_points(self):
        """Initializes the save points for the RBM model."""
        num_points = 50
        self.t_to_save = sorted(
            list(
                set(
                    np.round(
                        np.logspace(np.log10(1), np.log10(self.max_epochs),
                                    num_points)).astype(int).tolist())))

    def _initialize_physical_performance(self):
        """Initializes the physical performance metrics for the RBM model."""
        self.power_f = 0
        self.power_b = 0
        self.energy = 0
        self.W_t = self.W.t()
        self.relax_t_f, self.relax_t_b = self.relaxation_times()

    def pretrain(self, pretrained_model, model_state_path='model_states/'):
        """Loads pretrained parameters from a specified model.

        Parameters
        ----------
        pretrained_model : str
            The name of the pretrained model.
        model_state_path : str, optional
            The path to the directory containing the model states (default is 'model_states/').
        """
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
                    # *** Import pretrained parameters
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
        """Converts visible units to hidden units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        tuple
            The probabilities and samples of the hidden units.
        """
        if beta is None:
            beta = self.model_beta
        else:
            if beta > 1000:
                return self.Deterministic_v_to_h(v, beta)
        return self.Bernoulli_v_to_h(v, beta)

    def h_to_v(self, h, beta=None):
        """Converts hidden units to visible units.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        tuple
            The probabilities and samples of the visible units.
        """
        if beta is None:
            beta = self.model_beta
        else:
            if beta > 1000:
                return self.Deterministic_h_to_v(h, beta)
        return self.Bernoulli_h_to_v(h, beta)

    def Deterministic_v_to_h(self, v, beta):
        """Deterministically converts visible units to hidden units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        beta : float
            The inverse temperature parameter.

        Returns
        -------
        tuple
            The deterministic hidden units.
        """
        h = (self.delta_eh(v) > 0).to(v.dtype)
        return h, h

    def Deterministic_h_to_v(self, h, beta):
        """Deterministically converts hidden units to visible units.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.
        beta : float
            The inverse temperature parameter.

        Returns
        -------
        tuple
            The deterministic visible units.
        """
        v = (self.delta_ev(h) > 0).to(h.dtype)
        return v, v

    def Bernoulli_v_to_h(self, v, beta):
        """Converts visible units to hidden units using Bernoulli sampling.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        beta : float
            The inverse temperature parameter.

        Returns
        -------
        tuple
            The probabilities and samples of the hidden units.
        """
        p_h = self._prob_h_given_v(v, beta)
        sample_h = torch.bernoulli(p_h)
        return p_h, sample_h

    def Bernoulli_h_to_v(self, h, beta):
        """Converts hidden units to visible units using Bernoulli sampling.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.
        beta : float
            The inverse temperature parameter.

        Returns
        -------
        tuple
            The probabilities and samples of the visible units.
        """
        p_v = self._prob_v_given_h(h, beta)
        sample_v = torch.bernoulli(p_v)
        return p_v, sample_v

    def _free_energy_hopfield(self, v, beta=None):
        """Computes the free energy of the visible units using the Hopfield energy function.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        torch.Tensor
            The free energy of the visible units.
        """
        if beta is None:
            beta = self.model_beta
        vbias_term = torch.mv(v, self.v_bias) * beta
        wx_b = torch.mm(v, self.W.t()) + self.h_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b * beta)), axis=1)
        return -hidden_term - vbias_term

    def _energy_hopfield(self, v, h):
        """Computes the energy of the visible and hidden units using the Hopfield energy function.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        torch.Tensor
            The energy of the visible and hidden units.
        """
        energy = -(torch.mm(v, self.W.t()) * h).sum(1) - torch.mv(
            v, self.v_bias) - torch.mv(h, self.h_bias)
        return energy

    def forward(self, v, k, beta=None):
        """Performs a forward pass through the RBM model.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        k : int
            The number of Gibbs sampling steps.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        torch.Tensor
            The reconstructed visible units.
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
        """Trains the RBM model using the specified training algorithm.

        Parameters
        ----------
        train_data : iterable
            The training data.
        test_data : iterable, optional
            The test data (default is an empty list).
        print_error : bool, optional
            Whether to print the training error (default is False).
        print_test_error : bool, optional
            Whether to print the test error (default is False).
        model_state_path : str, optional
            The path to the directory containing the model states (default is 'model_states/').
        print_every : int, optional
            The number of epochs between printing the training status (default is 100).
        """
        while self.epoch < self.max_epochs:
            self.W_t = self.W.t()

            for _, v_data in enumerate(train_data):

                start_time = time.time()
                self.power_f = 0
                self.power_b = 0

                h_data = self.v_to_h(v_data)[1]
                p_f = self.power_forward(v_data)
                self.power_f += p_f.mean()
                self.energy += p_f.sum()

                if self.train_algo == 'PCD':
                    v_model = self.persistent_chains
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

                if self.centering:
                    self.batch_ov = v_data.mean(0)
                    self.batch_oh = h_data.mean(0)
                    self.ov = (1 -
                               self.slv) * self.ov + self.slv * self.batch_ov
                    self.oh = (1 -
                               self.slh) * self.oh + self.slh * self.batch_oh

                dEdW_data, dEdv_bias_data, dEdh_bias_data = self.derivatives(
                    v_data, h_data)
                dEdW_model, dEdv_bias_model, dEdh_bias_model = self.derivatives(
                    v_model, h_model)

                dEdW_data = torch.mean(dEdW_data, dim=0)
                dEdv_bias_data = torch.mean(dEdv_bias_data, dim=0)
                dEdh_bias_data = torch.mean(dEdh_bias_data, dim=0)
                dEdW_model = torch.mean(dEdW_model, dim=0)
                dEdv_bias_model = torch.mean(dEdv_bias_model, dim=0)
                dEdh_bias_model = torch.mean(dEdh_bias_model, dim=0)

                if self.optimizer == 'Adam':
                    self.Adam_update(self.epoch + 1, dEdW_data, dEdW_model,
                                     dEdv_bias_data, dEdv_bias_model,
                                     dEdh_bias_data, dEdh_bias_model)
                elif self.optimizer == 'SGD':
                    self.SGD_update(dEdW_data, dEdW_model, dEdv_bias_data,
                                    dEdv_bias_model, dEdh_bias_data,
                                    dEdh_bias_model)

                self.after_step_keepup()

                self.relax_t_f, self.relax_t_b = self.relaxation_times()

                self.epoch += 1

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
        """Performs operations to keep the model parameters
        within specified bounds after each training step."""
        self.clip_weights()
        self.clip_bias()

    def SGD_update(self, dEdW_data, dEdW_model, dEdv_bias_data,
                   dEdv_bias_model, dEdh_bias_data, dEdh_bias_model):
        """Updates the model parameters using Stochastic Gradient Descent (SGD).

        Parameters
        ----------
        dEdW_data : torch.Tensor
            The gradient of the energy with respect to the weights from the data.
        dEdW_model : torch.Tensor
            The gradient of the energy with respect to the weights from the model.
        dEdv_bias_data : torch.Tensor
            The gradient of the energy with respect to the visible biases from the data.
        dEdv_bias_model : torch.Tensor
            The gradient of the energy with respect to the visible biases from the model.
        dEdh_bias_data : torch.Tensor
            The gradient of the energy with respect to the hidden biases from the data.
        dEdh_bias_model : torch.Tensor
            The gradient of the energy with respect to the hidden biases from the model.
        """
        dW = -dEdW_data + dEdW_model
        dv = -dEdv_bias_data + dEdv_bias_model
        dh = -dEdh_bias_data + dEdh_bias_model
        if self.centering:
            dv = dv - torch.matmul(self.oh, dW)
            dh = dh - torch.matmul(self.ov, dW.t())
        if self.regularization == 'l2':
            dW -= self.l2 * 2 * self.W
            dv -= self.l2 * 2 * self.v_bias
            dh -= self.l2 * 2 * self.h_bias
        elif self.regularization == 'l1':
            dW -= self.l1 * torch.sign(self.W)
            dv -= self.l1 * torch.sign(self.v_bias)
            dh -= self.l1 * torch.sign(self.h_bias)
        self.W.add_(self.lr * dW)
        self.v_bias.add_(self.lr * dv)
        self.h_bias.add_(self.lr * dh)

    def Adam_update(self, t, dEdW_data, dEdW_model, dEdv_bias_data,
                    dEdv_bias_model, dEdh_bias_data, dEdh_bias_model):
        """Updates the model parameters using the Adam optimizer.

        Parameters
        ----------
        t : int
            The current epoch.
        dEdW_data : torch.Tensor
            The gradient of the energy with respect to the weights from the data.
        dEdW_model : torch.Tensor
            The gradient of the energy with respect to the weights from the model.
        dEdv_bias_data : torch.Tensor
            The gradient of the energy with respect to the visible biases from the data.
        dEdv_bias_model : torch.Tensor
            The gradient of the energy with respect to the visible biases from the model.
        dEdh_bias_data : torch.Tensor
            The gradient of the energy with respect to the hidden biases from the data.
        dEdh_bias_model : torch.Tensor
            The gradient of the energy with respect to the hidden biases from the model.
        """
        dW = -dEdW_data + dEdW_model
        dv = -dEdv_bias_data + dEdv_bias_model
        dh = -dEdh_bias_data + dEdh_bias_model
        if self.centering:
            dv = dv - torch.matmul(self.oh, dW)
            dh = dh - torch.matmul(self.ov, dW.t())
        if self.regularization == 'l2':
            dW += self.l2 * 2 * self.W
            dv += self.l2 * 2 * self.v_bias
            dh += self.l2 * 2 * self.h_bias
        elif self.regularization == 'l1':
            dW += self.l1 * torch.sign(self.W)
            dv += self.l1 * torch.sign(self.v_bias)
            dh += self.l1 * torch.sign(self.h_bias)
        self.m_dW = self.beta1 * self.m_dW + (1 - self.beta1) * dW
        self.m_dv = self.beta1 * self.m_dv + (1 - self.beta1) * dv
        self.m_dh = self.beta1 * self.m_dh + (1 - self.beta1) * dh
        self.v_dW = self.beta2 * self.v_dW + (1 - self.beta2) * (dW**2)
        self.v_dv = self.beta2 * self.v_dv + (1 - self.beta2) * (dv**2)
        self.v_dh = self.beta2 * self.v_dh + (1 - self.beta2) * (dh**2)
        m_dW_corr = self.m_dW / (1 - self.beta1**t)
        m_dv_corr = self.m_dv / (1 - self.beta1**t)
        m_dh_corr = self.m_dh / (1 - self.beta1**t)
        v_dW_corr = self.v_dW / (1 - self.beta2**t)
        v_dv_corr = self.v_dv / (1 - self.beta2**t)
        v_dh_corr = self.v_dh / (1 - self.beta2**t)
        self.W = self.W + self.lr * (m_dW_corr /
                                     (torch.sqrt(v_dW_corr) + self.epsilon))
        self.v_bias = self.v_bias + self.lr * (
            m_dv_corr / (torch.sqrt(v_dv_corr) + self.epsilon))
        self.h_bias = self.h_bias + self.lr * (
            m_dh_corr / (torch.sqrt(v_dh_corr) + self.epsilon))

    def reconstruct(self, data, k):
        """Reconstructs the visible units from the data using k Gibbs sampling steps.

        Parameters
        ----------
        data : array-like
            The input data.
        k : int
            The number of Gibbs sampling steps.

        Returns
        -------
        tuple
            The original and reconstructed visible units.
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
        """Generates samples from the RBM model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        k : int
            The number of Gibbs sampling steps.
        h_binarized : bool, optional
            Whether to binarize the hidden units (default is True).
        from_visible : bool, optional
            Whether to generate samples from visible units (default is True).
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        numpy.ndarray
            The generated samples.
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
        """Clips the weights of the RBM model to be within specified bounds."""
        self.W = torch.clip(self.W, self.min_W, self.max_W)
        self.W_t = self.W.t()

    def clip_bias(self):
        """Clips the biases of the RBM model to be within specified bounds."""
        self.v_bias = torch.clip(self.v_bias, self.min_W, self.max_W)
        self.h_bias = torch.clip(self.h_bias, self.min_W, self.max_W)

    def _prob_h_given_v(self, v, beta=None):
        """Computes the probability of the hidden units given the visible units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        torch.Tensor
            The probabilities of the hidden units.
        """
        if beta is None:
            beta = self.model_beta
        return torch.sigmoid(beta * self.delta_eh(v))

    def _prob_v_given_h(self, h, beta=None):
        """Computes the probability of the visible units given the hidden units.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        torch.Tensor
            The probabilities of the visible units.
        """
        if beta is None:
            beta = self.model_beta
        return torch.sigmoid(beta * self.delta_ev(h))

    def delta_eh(self, v):
        """Computes the change in energy with respect to the hidden units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.

        Returns
        -------
        torch.Tensor
            The change in energy with respect to the hidden units.
        """
        return self._delta_eh_hopfield(v)

    def delta_ev(self, h):
        """Computes the change in energy with respect to the visible units.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        torch.Tensor
            The change in energy with respect to the visible units.
        """
        return self._delta_ev_hopfield(h)

    def _delta_eh_hopfield(self, v):
        """Computes the change in energy with respect to the hidden units using the Hopfield energy function.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.

        Returns
        -------
        torch.Tensor
            The change in energy with respect to the hidden units.
        """
        return torch.mm(v, self.W_t) + self.h_bias

    def _delta_ev_hopfield(self, h):
        """Computes the change in energy with respect to the visible units using the Hopfield energy function.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        torch.Tensor
            The change in energy with respect to the visible units.
        """
        return torch.mm(h, self.W) + self.v_bias

    def derivatives(self, v, h):
        """Computes the derivatives of the energy with respect to the weights and biases.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        tuple
            The derivatives of the energy with respect to the weights, visible biases, and hidden biases.
        """
        return self.derivatives_hopfield(v, h)

    def free_energy(self, v, beta=None):
        """Computes the free energy of the visible units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        beta : float, optional
            The inverse temperature parameter (default is None).

        Returns
        -------
        torch.Tensor
            The free energy of the visible units.
        """
        if beta is None:
            beta = self.model_beta
        return self._free_energy_hopfield(v, beta)

    def derivatives_hopfield(self, v, h):
        """Computes the derivatives of the energy with respect to
        the weights and biases using the Hopfield energy function.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        tuple
            The derivatives of the energy with respect to the weights, visible biases, and hidden biases.
        """
        if self.centering:
            dEdW = -torch.einsum('ij,ik->ijk', h - self.oh, v - self.ov)
        else:
            dEdW = -torch.einsum('ij,ik->ijk', h, v)
        dEdv_bias = -v
        dEdh_bias = -h
        return dEdW, dEdv_bias, dEdh_bias

    def plot_weights(self, t):
        """Plots the weights of the RBM model.

        Parameters
        ----------
        t : int
            The current epoch.
        """
        Ndata = self.W.shape[0]
        data_3d = self.W.detach().cpu().numpy().reshape(Ndata, 28, 28)
        num_rows = int(np.ceil(np.sqrt(Ndata)))
        num_cols = int(np.ceil(Ndata / num_rows))
        fig, ax = plt.subplots(nrows=num_rows,
                               ncols=num_cols,
                               figsize=(10, 10))
        for i in range(Ndata):
            row = i // num_cols
            col = i % num_cols
            ax[row, col].imshow(data_3d[i], cmap='magma')
            ax[row, col].axis('off')
        if num_rows * num_cols > Ndata:
            for i in range(Ndata, num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                fig.delaxes(ax[row, col])
        plt.suptitle('Weights epoch {}'.format(t))
        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9)
        vmin = np.min(self.W.detach().cpu().numpy())
        vmax = np.max(self.W.detach().cpu().numpy())
        dummy_img = np.zeros((1, 1))
        cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        plt.colorbar(plt.imshow(dummy_img, cmap='magma', vmin=vmin, vmax=vmax),
                     cax=cax)
        cax.set_aspect('auto')

    def plot_visible_bias(self, t):
        """Plots the visible biases of the RBM model.

        Parameters
        ----------
        t : int
            The current epoch.
        """
        data_2d = self.v_bias.detach().cpu().numpy().reshape(28, 28)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(data_2d, cmap='magma')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Values', rotation=-90, va='bottom')
        ax.set_title('Visible Biases epoch {}'.format(t))
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')

    def plot_bias(self, t):
        """Plots the hidden and visible biases of the RBM model.

        Parameters
        ----------
        t : int
            The current epoch.
        """
        h_bias = self.h_bias.detach().cpu().numpy()
        v_bias = self.v_bias.detach().cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.hist(h_bias, bins=20, color='blue', edgecolor='black')
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Hidden Biases epoch {}'.format(t))
        ax2.hist(v_bias, bins=20, color='red', edgecolor='black')
        ax2.set_xlabel('Values')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Visible Biases epoch {}'.format(t))
        plt.tight_layout()

    def _center(self):
        """Centers the weights and biases of the RBM model.

        Returns
        -------
        tuple
            The centered weights, visible biases, and hidden biases.
        """
        W_centered = self.W_t
        v_bias_centered = (self.v_bias +
                           0.5 * W_centered.sum(dim=1)) / self.g_v
        h_bias_centered = (self.h_bias +
                           0.5 * W_centered.sum(dim=0)) / self.g_h

        return W_centered, v_bias_centered, h_bias_centered

    def _RKM_v_to_h(self, v_centered, W_centered, v_bias_centered,
                    h_bias_centered):
        """Converts centered visible units to hidden units using the RKM method.

        Parameters
        ----------
        v_centered : torch.Tensor
            The centered visible units.
        W_centered : torch.Tensor
            The centered weights.
        v_bias_centered : torch.Tensor
            The centered visible biases.
        h_bias_centered : torch.Tensor
            The centered hidden biases.

        Returns
        -------
        torch.Tensor
            The hidden units.
        """
        h_eq = (torch.mm(v_centered, W_centered) +
                self.g_h * h_bias_centered) / (
                    (torch.abs(W_centered).sum(dim=0) +
                     torch.abs(h_bias_centered)))
        return h_eq

    def _RKM_h_to_v(self, h_centered, W_centered, v_bias_centered,
                    h_bias_centered):
        """Converts centered hidden units to visible units using the RKM method.

        Parameters
        ----------
        h_centered : torch.Tensor
            The centered hidden units.
        W_centered : torch.Tensor
            The centered weights.
        v_bias_centered : torch.Tensor
            The centered visible biases.
        h_bias_centered : torch.Tensor
            The centered hidden biases.

        Returns
        -------
        torch.Tensor
            The visible units.
        """
        v_eq = (torch.mm(h_centered, W_centered.T) +
                self.g_v * v_bias_centered) / (torch.abs(W_centered).sum(dim=1)
                                               + torch.abs(v_bias_centered))
        return v_eq

    def power_forward(self, v):
        """Computes the forward power of the visible units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.

        Returns
        -------
        torch.Tensor
            The forward power of the visible units.
        """
        v_centered = v - 0.5
        W_centered, v_bias_centered, h_bias_centered = self._center()
        h_eq = self._RKM_v_to_h(v_centered, W_centered, v_bias_centered,
                                h_bias_centered)

        power = (torch.matmul(v_centered**2,
                              torch.abs(W_centered / 2).sum(dim=1)) +
                 torch.matmul(h_eq**2,
                              torch.abs(W_centered / 2).sum(dim=0)) -
                 torch.einsum('ij,ji->i', v_centered,
                              torch.matmul(W_centered, h_eq.T)) +
                 torch.matmul(
                     (h_eq**2 + self.g_h**2), torch.abs(h_bias_centered)) -
                 torch.matmul(h_eq, h_bias_centered) * self.g_h)

        return power

    def power_backward(self, h):
        """Computes the backward power of the hidden units.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        torch.Tensor
            The backward power of the hidden units.
        """
        h_centered = h - 0.5
        W_centered, v_bias_centered, h_bias_centered = self._center()
        v_eq = self._RKM_h_to_v(h_centered, W_centered, v_bias_centered,
                                h_bias_centered)

        power = (torch.matmul(h_centered**2,
                              torch.abs(W_centered / 2).sum(dim=0)) +
                 torch.matmul(v_eq**2,
                              torch.abs(W_centered / 2).sum(dim=1)) -
                 torch.einsum('ij,ji->i', h_centered,
                              torch.matmul(W_centered.T, v_eq.T)) +
                 torch.matmul(
                     (v_eq**2 + self.g_v**2), torch.abs(v_bias_centered)) -
                 torch.matmul(v_eq, v_bias_centered) * self.g_v)

        return power

    def av_power_forward(self, v):
        """Computes the average forward power of the visible units.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.

        Returns
        -------
        torch.Tensor
            The average forward power of the visible units.
        """
        return self.power_forward(v).mean()

    def av_power_backward(self, h):
        """Computes the average backward power of the hidden units.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.

        Returns
        -------
        torch.Tensor
            The average backward power of the hidden units.
        """
        return self.power_backward(h).mean()

    def relaxation_times(self):
        """Computes the relaxation times for the forward and backward passes.

        Returns
        -------
        tuple
            The relaxation times for the forward and backward passes.
        """
        W_centered, v_bias_centered, h_bias_centered = self._center()
        t_forward = 1 / (torch.abs(W_centered / 2).sum(dim=0) +
                         torch.abs(h_bias_centered))
        t_backward = 1 / (torch.abs(W_centered / 2).sum(dim=1) +
                          torch.abs(v_bias_centered))

        return t_forward, t_backward
