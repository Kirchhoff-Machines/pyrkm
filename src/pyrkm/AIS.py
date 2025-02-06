import numpy as np
import sys
import random
import time
import matplotlib.pyplot as plt
import pickle
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

# ********* Functions to compute AIS score
# they do not work as expected (different results from literature)
# Overall, since we can not compute AIS for circuit models or ReLU approach, it is not worth to debug this 

# from old implementation:
# *** For ReLU models the probability is not defined, so free energy indicators are ill defined
#ll2 = ComputeFreeEnergyAIS(test_model, v_bias_base.to(test_model.device), nb=1000, nI=1000)
## Calculate logZ using AIS
#logZ = annealed_importance_sampling(test_model,v_bias_base.to(test_model.device),num_chains=1000,betas=1000).detach().cpu()
## Calculate the loglikelihood of this batch of data
#LL = torch.mean(log_likelihood_v(test_model,logZ, train_data.to(test_model.device), v_bias_base.to(test_model.device)))
# ***

def annealed_importance_sampling(model,
                                 v_bias_base,
                                 num_chains=1000,
                                 betas=1000,
                                 ):
    """ Approximates the partition function for the given model using annealed importance sampling.

    .. seealso:: Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing \
                 http://arxiv.org/pdf/1412.8566.pdf

    :param model: The model.
    :type model: Valid RBM model.

    :param num_chains: Number of AIS runs.
    :type num_chains: int

    :param k: Number of Gibbs sampling steps.
    :type k: int

    :param betas: Number or a list of inverse temperatures to sample from.
    :type betas: int, numpy array [num_betas]

    :return: | Mean estimated log partition function,
    :rtype: float
    """
    # Setup temerpatures 
    betas = torch.linspace(0.0, 1.0, betas)


    ## Sample the first time from the base model (start does not matter since beta=0)
    #_, v = model.h_to_v(torch.zeros((num_chains, model.n_hidden)).to(model.device).to(torch.double), betas[0])
    #_, h = model.v_to_h(v, betas[0])
    v = torch.bernoulli(torch.sigmoid(v_bias_base.unsqueeze(0).repeat(num_chains,1)))
    #_, h = model.v_to_h(v, betas[0])
    #E = model.energy(v,h)
    # Base distro:
    #    P(x)=exp[-beta*(v*v_bias_base)]/[2**nh Zv]
    #    Zv = 1 + exp[beta*v_bias_base]
    #E += torch.mv(v, v_bias_base) 
    

    # Calculate the unnormalized probabilties of v
    lnpvsum = -unnormalized_log_probability_v(model, v, v_bias_base, betas[0], True)

    for beta in betas[1:betas.shape[0] - 1]:

        # Calculate the unnormalized probabilties of v
        lnpvsum += unnormalized_log_probability_v(model, v, v_bias_base, beta, True)

        # Sample k times from the intermidate distribution
        for _ in range(model.k):
            _, h = model.v_to_h(v, beta)
            _, v = model.h_to_v(h, beta)

        # Calculate the unnormalized probabilties of v
        lnpvsum -= unnormalized_log_probability_v(model, v, v_bias_base, beta, True)

        #E += model.energy(v,h)
        
    # Calculate the unnormalized probabilties of v
    lnpvsum += unnormalized_log_probability_v(model, v, v_bias_base, betas[betas.shape[0] - 1], True)

    # Calculate an estimate of logz .
    logz = log_sum_exp(lnpvsum) - np.log(num_chains)

    # Calculate partition function of base distribution (False=uniform; True=biased on data)
    baselogz = _base_log_partition(model,v_bias_base, True)

    # Add the base partition function
    logz = logz + baselogz

    #db = 1/betas.shape[0]
    #d = (-db*E).to(torch.double)

    ## to avoid overflow introduce d0
    #d0 = d.mean()
    #logz = torch.log( torch.mean( torch.exp(d-d0)))+d0

    return logz


def unnormalized_log_probability_v(model, v, v_bias_base, beta=None, use_base_model=False):
    """ Computes the unnormalized log probabilities of v.

    :param v: Visible states.
    :type v: numpy array [batch size, input dim]

    :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                 different betas simultaneously.None is equivalent to pass the value 1.0.

    :type beta: None, float or numpy array [batch size, 1]

    :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
    :type use_base_model: bool

    :return: Unnormalized log probability of v.
    :rtype: numpy array [batch size, 1]
    """
    temp_v = v #- model.ov
    activation = torch.mm(temp_v, model.W.t()) + model.h_bias
    bias = torch.mv(temp_v, model.v_bias)
    if beta is not None:
        activation *= beta
        bias *= beta
        if use_base_model is True:
            bias += (1.0 - beta) * torch.mv(temp_v, v_bias_base)
    return bias + torch.sum(torch.log(1 + torch.exp(activation)), axis=1)
    #return bias + torch.sum(torch.log(torch.exp(activation * (1.0 - model.oh)) + torch.exp(-activation * model.oh)), axis=1)


def _base_log_partition(model, v_bias_base, use_base_model=False):
    """ Returns the base partition function for a given visible bias. .. Note:: that for AIS we need to be able to \
        calculate the partition function of the base distribution exactly. Furthermore it is beneficial if the \
        base distribution is a good approximation of the target distribution. A good choice is therefore the \
        maximum likelihood estimate of the visible bias, given the data.

    :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
    :type use_base_model: bool

    :return: Partition function for zero parameters.
    :rtype: float
    """
    if use_base_model is True:
        #return torch.sum(torch.log(torch.exp(-model.ov * v_bias_base) + torch.exp((1.0 - model.ov) * v_bias_base)) ) + model.n_hidden * np.log(2.0)
        #return torch.sum(torch.log(1 + torch.exp(v_bias_base)) ) + model.n_hidden * np.log(2.0)
        return torch.sum(torch.log(1 + torch.exp(v_bias_base)) ) + model.n_hidden * np.log(2.0)
    else:
        return model.n_visible * np.log(2.0) + model.n_hidden * np.log(2.0)


def log_likelihood_v(model, logz, data, v_bias_base, beta=None):
    """ Computes the log-likelihood (LL) for a given model and visible data given its log partition function.

        :Info: logz needs to be the partition function for the same beta (i.e. beta = 1.0)!

    :param logz: The logarithm of the partition function.
    :type logz: float

    :param data: The visible data.
    :type data: 2D array [num samples, num input dim] 
            
    :param beta: Inverse temperature(s) for the models energy.
    :type beta: None, float, numpy array [batchsize,1]

    :return: The log-likelihood for each sample.
    :rtype: numpy array [num samples]
    """
    return log_probability_v(model, logz, data, v_bias_base, beta)


def log_probability_v(model, logz, v, v_bias_base, beta=None, use_base_model=False):
    """ Computes the log-probability / LogLikelihood(LL) for the given visible units for this model. To estimate \
        the LL we need to know the logarithm of the partition function Z. For small models it is possible to \
        calculate Z, however since this involves calculating all possible hidden states, it is intractable for \
        bigger models. As an estimation method annealed importance sampling (AIS) can be used instead.

    :param logz: The logarithm of the partition function.
    :type logz: float

    :param v: Visible states.
    :type v: numpy array [batch size, input dim]

    :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                 different betas simultaneously.None is equivalent to pass the value 1.0.
    :type beta: None, float or numpy array [batch size, 1]

    :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
    :type use_base_model: bool

    :return: Log probability for visible_states.
    :rtype: numpy array [batch size, 1]
    """
    return unnormalized_log_probability_v(model, v, v_bias_base, beta, use_base_model) - logz


def log_sum_exp(x, axis=0):
    alpha = x.max(axis).values - np.log(np.finfo(np.float64).max) / 2.0
    if axis == 1:
        return torch.squeeze(alpha + torch.log(torch.sum(torch.exp(x.T - alpha), axis=0)))
    else:
        return torch.squeeze(alpha + torch.log(torch.sum(torch.exp(x - alpha), axis=0)))


def log_diff_exp(x, axis=0):
    alpha = x.max(axis).values - np.log(np.finfo(np.float64).max) / 2.0
    if axis == 1:
        return torch.squeeze(alpha + torch.log(torch.diff(torch.exp(x.T - alpha), n=1, axis=0)))
    else:
        return torch.squeeze(alpha + torch.log(torch.diff(torch.exp(x - alpha), n=1, axis=0)))
    
# Uniform prior
# p0(x,h)= 1/Z0
# Z0 = 2**(nvis+nhid)
def ComputeFreeEnergyAIS(model,v_bias_base,nb,nI):

    blist = torch.arange(0,1.000001,1.0/nb)
    x = torch.zeros(model.n_visible +model.n_hidden,nI,device=model.device)
    H = torch.zeros(model.n_hidden,nI,device=model.device)
    E = torch.zeros(nI,device=model.device)

    # initialize xref
    v = torch.bernoulli(torch.sigmoid(v_bias_base).repeat(nI,1))
    v = torch.bernoulli(torch.rand((nI,model.n_visible),device=model.device))
    h = torch.bernoulli(torch.rand((nI,model.n_hidden),device=model.device))
    E = model.energy(v,h).double().to(model.device) 
    print(E.shape)
    
    for idb in range(1,nb+1):
        for _ in range(model.k):
            _, h = model.v_to_h(v,blist[idb])
            _, v = model.h_to_v(h,blist[idb])
        E += model.energy(v,h)

    d = 0
    db = 1.0/nb
    d = -db*E
    d = d.double()
    d0 = torch.mean(d)

    AIS = torch.log(torch.mean(torch.exp(d-d0).double()))+d0
    # AIS = torch.log(torch.mean(torch.exp(Δ)))
    return AIS

def LogLike(model,v,logZ):
    #_, h = model.v_to_h(v)
    F = model.free_energy(v).mean()
    print(F)
    return F - logZ 
       

