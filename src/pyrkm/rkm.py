"""Restricted Kirchhoff Machine implementation."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from .rbm import RBM


@dataclass
class RKM(RBM):
    """A class to represent a Restricted Kirchhoff Machine (RKM).
    It is inherited from the RBM class, so look at the RBM class for info about the attributes and methods.
    Also, refer to the paper https://arxiv.org/abs/2509.15842
    (also published in PNAS https://www.pnas.org/doi/abs/10.1073/pnas.2525792123) for more details.

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

    energy_type: str = "RKM"
    offset: float = 0.0
    sampling: str = "bernoulli"
    distribution: str = "gaussian"
    layer_scaled: bool = True

    def __post_init__(self):
        """Initialize the RKM model after the dataclass is created."""
        super().__post_init__()
        if self.average_data is not None:
            self.v_bias = self.average_data.to(self.device).to(self.mytype)

    def v_to_h(self, v: torch.Tensor, beta: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        if self.energy_type == "RKM":
            effective_h_bias = self.h_bias + 0.5 * self.offset * (
                (torch.abs(self.h_bias) - self.h_bias) / self.g_h + (torch.abs(self.W) - self.W).sum(dim=1)
            )
            num = torch.mm(v, self.W_t) + effective_h_bias
            den = torch.abs(self.W).sum(dim=1) + torch.abs(self.h_bias) / self.g_h
            h_analog = num / den

            if self.sampling == "bernoulli":
                if self.layer_scaled:
                    p_h = torch.sigmoid(beta * self.n_visible * h_analog)
                    h = torch.bernoulli(p_h)
                else:
                    p_h = torch.sigmoid(beta * h_analog)
                    h = torch.bernoulli(p_h)
            elif self.sampling == "multi-threshold":
                if self.distribution == "gaussian":
                    t = torch.randn_like(h_analog, dtype=self.mytype, device=self.device) * 1 / beta
                else:
                    t = (torch.rand_like(h_analog, dtype=self.mytype, device=self.device) * 2 - 1) * 1 / beta
                p_h = h_analog
                if self.layer_scaled:
                    h = (p_h > t / self.n_visible).to(v.dtype)
                else:
                    h = (p_h > t).to(v.dtype)
            elif self.sampling == "single-threshold":
                if self.distribution == "gaussian":
                    t = (
                        torch.randn(1, dtype=self.mytype, device=self.device)
                        * 1
                        / beta
                        * torch.ones_like(h_analog, dtype=self.mytype, device=self.device)
                    )
                else:
                    t = (
                        (torch.rand(1, dtype=self.mytype, device=self.device) * 2 - 1)
                        * 1
                        / beta
                        * torch.ones_like(h_analog, dtype=self.mytype, device=self.device)
                    )
                p_h = h_analog
                if self.layer_scaled:
                    h = (p_h > t / self.n_visible).to(v.dtype)
                else:
                    h = (p_h > t).to(v.dtype)
            return p_h, h
        else:
            return super().v_to_h(v, beta)

    def h_to_v(self, h: torch.Tensor, beta: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        if self.energy_type == "RKM":
            effective_v_bias = self.v_bias + 0.5 * self.offset * (
                (torch.abs(self.v_bias) - self.v_bias) / self.g_v + (torch.abs(self.W) - self.W).sum(dim=0)
            )
            num = torch.mm(h, self.W) + effective_v_bias
            den = torch.abs(self.W).sum(dim=0) + torch.abs(self.v_bias) / self.g_v
            v_analog = num / den

            if self.sampling == "bernoulli":
                if self.layer_scaled:
                    p_v = torch.sigmoid(beta * self.n_hidden * v_analog)
                    v = torch.bernoulli(p_v)
                else:
                    p_v = torch.sigmoid(beta * v_analog)
                    v = torch.bernoulli(p_v)
            elif self.sampling == "multi-threshold":
                if self.distribution == "gaussian":
                    t = torch.randn_like(v_analog, dtype=self.mytype, device=self.device) * 1 / beta
                else:
                    t = (torch.rand_like(v_analog, dtype=self.mytype, device=self.device) * 2 - 1) * 1 / beta
                p_v = v_analog
                if self.layer_scaled:
                    v = (p_v > t / self.n_hidden).to(h.dtype)
                else:
                    v = (p_v > t).to(h.dtype)
            elif self.sampling == "single-threshold":
                if self.distribution == "gaussian":
                    t = (
                        torch.randn(1, dtype=self.mytype, device=self.device)
                        * 1
                        / beta
                        * torch.ones_like(v_analog, dtype=self.mytype, device=self.device)
                    )
                else:
                    t = (
                        (torch.rand(1, dtype=self.mytype, device=self.device) * 2 - 1)
                        * 1
                        / beta
                        * torch.ones_like(v_analog, dtype=self.mytype, device=self.device)
                    )
                p_v = v_analog
                if self.layer_scaled:
                    v = (p_v > t / self.n_hidden).to(h.dtype)
                else:
                    v = (p_v > t).to(h.dtype)
            return p_v, v
        else:
            return super().h_to_v(h, beta)

    def derivatives(
        self, v: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if self.energy_type == "hopfield" or self.energy_type == "RKM":
            return self.derivatives_hopfield(v, h)
        else:
            # exit error
            print("Error: derivatives not implemented for this energy type")
            sys.exit()

    def delta_eh(self, v: torch.Tensor) -> torch.Tensor:
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
        if self.energy_type == "hopfield":
            return self._delta_eh_hopfield(v)
        else:
            # exit error
            print("Error: delta_eh not implemented for this energy type")
            sys.exit()

    def delta_ev(self, h: torch.Tensor) -> torch.Tensor:
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
        if self.energy_type == "hopfield":
            return self._delta_ev_hopfield(h)
        else:
            # exit error
            print("Error: delta_ev not implemented for this energy type")
            sys.exit()

    # **** Hopfield transfer functions
    # ****************************************************************************************************
    #
    #                                    PERFORMANCE METRICS
    #
    # ****************************************************************************************************
    # ****************************************************************************************************

    def power_forward(self, v: torch.Tensor) -> torch.Tensor:
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
            (torch.abs(self.h_bias) - self.h_bias) / self.g_h + (torch.abs(self.W) - self.W).sum(dim=1)
        )
        num = torch.mm(v, self.W_t) + effective_h_bias
        den = torch.abs(self.W).sum(dim=1) + torch.abs(self.h_bias) / self.g_h
        h_analog = num / den

        W_t = self.W_t
        abs_W_t = torch.abs(self.W_t)
        h_bias = self.h_bias
        abs_h_bias = torch.abs(self.h_bias)

        power_forward = (
            -torch.einsum("ni,ij,nj->n", v, W_t, h_analog)
            + 0.5 * (torch.einsum("ni,ij->n", v**2, abs_W_t) + torch.einsum("ij,nj->n", abs_W_t, h_analog**2))
            - torch.einsum("j,nj->n", effective_h_bias, h_analog)
            + (0.5 / self.g_h) * torch.einsum("j,nj->n", abs_h_bias, h_analog**2 + self.g_h**2)
        )

        if self.offset != 0:
            power_forward = power_forward + (
                +torch.einsum("ni,ij->n", (-2 * self.offset * v + self.offset**2 / 4), abs_W_t - W_t)
                + (self.offset**2 / (4 * self.g_h) - self.offset / 2) * torch.sum(abs_h_bias - h_bias)
            )

        return power_forward

    def power_backward(self, h: torch.Tensor) -> torch.Tensor:
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
            (torch.abs(self.v_bias) - self.v_bias) / self.g_v + (torch.abs(self.W) - self.W).sum(dim=0)
        )
        num = torch.mm(h, self.W) + effective_v_bias
        den = torch.abs(self.W).sum(dim=0) + torch.abs(self.v_bias) / self.g_v
        v_analog = num / den

        W_t = self.W_t
        abs_W_t = torch.abs(self.W_t)
        v_bias = self.v_bias
        abs_v_bias = torch.abs(self.v_bias)

        power_backward = (
            -torch.einsum("ni,ij,nj->n", v_analog, W_t, h)
            + 0.5 * (torch.einsum("ni,ij->n", v_analog**2, abs_W_t) + torch.einsum("ij,nj->n", abs_W_t, h**2))
            - torch.einsum("i,ni->n", effective_v_bias, v_analog)
            + (0.5 / self.g_v) * torch.einsum("i,ni->n", abs_v_bias, v_analog**2 + self.g_v**2)
        )

        if self.offset != 0:
            power_backward = power_backward + (
                +torch.einsum("ij,nj->n", abs_W_t - W_t, (-2 * self.offset * h + self.offset**2 / 4))
                + (self.offset**2 / (4 * self.g_v) - self.offset / 2) * torch.sum(abs_v_bias - v_bias)
            )

        return power_backward

    def relaxation_times(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the relaxation times of the RKM in the forward and backward pass.

        Returns
        -------
        tuple of torch.Tensor
            t_forward : relaxation times of the RKM in the forward pass, shape (n_v,).
            t_backward : relaxation times of the RKM in the backward pass, shape (n_h,).
        """
        t_forward = 2 / (torch.abs(self.W).sum(dim=1) + torch.abs(self.h_bias) / self.g_h)
        t_backward = 2 / (torch.abs(self.W).sum(dim=0) + torch.abs(self.v_bias) / self.g_v)

        return t_forward, t_backward
