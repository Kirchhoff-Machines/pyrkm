import torch
from dataclasses import dataclass
from rbm_pytorch import RBM


@dataclass()
class SymRBM(RBM):
    g_v: float = 1
    g_h: float = 1

    def v_to_h(self, v, beta=None):
        if beta is None:
            beta = self.model_beta
        if self.energy_type == 'RKM_sync':
            # RKM synchronous update
            # define the threshold as a single uniform random number between 0 and exp(-beta)
            threshold = torch.rand(1, device=self.device,
                                   dtype=self.mytype) * torch.exp(-beta)
            # compute the probability of the hidden units being active
            return self.analog_to_digital(self.analog_deterministic_v_to_h(v),
                                          threshold)
        elif self.energy_type == 'RKM_async':
            # RKM asynchronous update
            # define the threshold as a uniform random number between 0 and exp(-beta)
            threshold = torch.rand(self.n_hidden,
                                   device=self.device,
                                   dtype=self.mytype) * torch.exp(-beta)
            # compute the probability of the hidden units being active
            return self.analog_to_digital(self.analog_deterministic_v_to_h(v),
                                          threshold)
        else:
            if beta > 1000:
                # I assume we are at T=0
                # print('deterministic visible to hidden', flush = True)
                return self.Deterministic_v_to_h(v, beta)
        return self.Bernoulli_v_to_h(v, beta)

    def h_to_v(self, h, beta=None):
        if beta is None:
            beta = self.model_beta
        if self.energy_type == 'RKM_sync':
            # RKM synchronous update
            # define the threshold as a single uniform random number between 0 and exp(-beta)
            threshold = torch.rand(1, device=self.device,
                                   dtype=self.mytype) * torch.exp(-beta)
            # compute the probability of the hidden units being active
            return self.analog_to_digital(self.analog_deterministic_h_to_v(h),
                                          threshold)
        elif self.energy_type == 'RKM_async':
            # RKM asynchronous update
            # define the threshold as a uniform random number between 0 and exp(-beta)
            threshold = torch.rand(self.n_visible,
                                   device=self.device,
                                   dtype=self.mytype) * torch.exp(-beta)
            # compute the probability of the hidden units being active
            return self.analog_to_digital(self.analog_deterministic_h_to_v(h),
                                          threshold)
        else:
            if beta > 1000:
                # I assume we are at T=0
                # print('deterministic hidden to visible', flush = True)
                return self.Deterministic_h_to_v(h, beta)
        return self.Bernoulli_h_to_v(h, beta)

    def Deterministic_v_to_h(self, v, beta):
        h = 2 * (self.delta_eh(v) > 0).to(v.dtype) - 1
        return h, h

    def Deterministic_h_to_v(self, h, beta):
        v = 2 * (self.delta_ev(h) > 0).to(h.dtype) - 1
        return v, v

    def analog_deterministic_v_to_h(self, v):
        p_h = (torch.mm(v, self.W_t) + self.h_bias) / (
            torch.abs(self.W).sum(dim=0) + torch.abs(self.h_bias))
        return p_h

    def analog_deterministic_h_to_v(self, h):
        p_v = (torch.mm(h, self.W) + self.v_bias) / (
            torch.abs(self.W).sum(dim=1) + torch.abs(self.v_bias))
        return p_v

    def analog_to_digital(self, p_data, threshold):
        data = 2 * (p_data > threshold).to(p_data.dtype) - 1
        return p_data, data

    def Bernoulli_v_to_h(self, v, beta):
        p_h = self._prob_h_given_v(v, beta)
        sample_h = 2 * torch.bernoulli(p_h) - 1
        return p_h, sample_h

    def Bernoulli_h_to_v(self, h, beta):
        p_v = self._prob_v_given_h(h, beta)
        sample_v = 2 * torch.bernoulli(p_v) - 1
        return p_v, sample_v

    # **** Hopfield transfer functions
    def _delta_eh_hopfield(self, v):
        return 2 * (torch.mm(v, self.W_t) + self.h_bias)

    def _delta_ev_hopfield(self, h):
        return 2 * (torch.mm(h, self.W) + self.v_bias)

    def derivatives(self, v, h):
        return self.derivatives_hopfield(v, h)

    def power_forward(self, v):
        """Computes the power dissipated by the RKM in the forward pass.

        Args:
            v: visible units, shape (N, n_v)
        Returns:
            Power dissipated by the RKM, shape (N,)
        """
        h_analog = self.v_to_h(v)[0]

        # Compute the power dissipated by the RKM
        power = (torch.matmul(v**2,
                              torch.abs(self.W_t / 2).sum(dim=1)) +
                 torch.matmul(h_analog**2,
                              torch.abs(self.W_t / 2).sum(dim=0)) -
                 torch.einsum('ij,ji->i', v, torch.matmul(
                     self.W_t, h_analog.T)) + torch.matmul(
                         (h_analog**2 + self.g_h**2), torch.abs(self.h_bias)) -
                 torch.matmul(h_analog, self.h_bias) * self.g_h)

        return power

    def power_backward(self, h):
        """Computes the power dissipated by the RKM in the backward pass.

        Args:
            h: hidden units, shape (N, n_h)
        Returns:
            Power dissipated by the RKM, shape (N,)
        """
        v_analog = self.h_to_v(h)[0]

        # Compute the power dissipated by the RKM
        power = (
            torch.matmul(h**2,
                         torch.abs(self.W_t / 2).sum(dim=0)) +
            torch.matmul(v_analog**2,
                         torch.abs(self.W_t / 2).sum(dim=1)) -
            torch.einsum('ij,ji->i', h, torch.matmul(self.W_t.T, v_analog.T)) +
            torch.matmul((v_analog**2 + self.g_v**2), torch.abs(self.v_bias)) -
            torch.matmul(v_analog, self.v_bias) * self.g_v)

        return power

    def relaxation_times(self):
        """Computes the relaxation times of the RKM in the forward and backward
        pass.

        Args:
            None
        Returns:
            t_forward: relaxation times of the RKM in the forward pass, shape (n_v,)
            t_backward: relaxation times of the RKM in the backward pass, shape (n_h,)
        """
        t_forward = 1 / (torch.abs(self.W_t / 2).sum(dim=0) +
                         torch.abs(self.h_bias))
        t_backward = 1 / (torch.abs(self.W_t / 2).sum(dim=1) +
                          torch.abs(self.v_bias))

        return t_forward, t_backward
