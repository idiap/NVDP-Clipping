# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import torch
import torch.nn as nn
import numpy as np
import itertools


def eye_scaled_(tensor, scale=1.0):
    with torch.no_grad():
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad).mul_(scale)
    return tensor


def init_vector_(tensor, init_vector):
    with torch.no_grad():
        tensor.copy_(init_vector)
    return tensor


class Quadratic(torch.nn.Module):
    def __init__(self, size_in, size_out):
        """
        In the constructor we instantiate three parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.linear = torch.nn.Linear(size_in, size_out)
        self.quadratic = torch.nn.Linear(size_in, size_out, bias=False)

        self.bias = self.linear.bias
        self.weight_linear = self.linear.weight
        self.weight_quadratic = self.quadratic.weight

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.linear(x) + self.quadratic(x**2)


class Exponential(nn.Module):
    """
    Simple exponential activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class Vib(nn.Module):
    """
    A Nonparameteric variational information bottleneck layer
    """

    def __init__(
        self,
        ib_dim,
        hidden_dim,
        hidden_size,
        activation,
        **kwargs
    ):
        super().__init__()

        self.prior_mu = nn.Parameter(torch.randn(ib_dim))
        self.prior_std = nn.Parameter(torch.randn(ib_dim))

        self.mu_proj = nn.Linear(hidden_dim, ib_dim)
        self.std_proj = nn.Linear(hidden_dim, ib_dim)
    
        self.activation = activation
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}

        intermediate_dim =(hidden_dim+hidden_size)//2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            self.activations[self.activation],
            nn.Linear(intermediate_dim, hidden_dim),
            self.activations[self.activation])
            
        self.ib_dim = ib_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.activation = activation  

    

    def estimate(self, encoder_output):
        """Estimates mu and std from the given input embeddings."""
        mu = self.mu_proj(encoder_output)
        std = torch.nn.functional.softplus(self.std_proj(encoder_output))
        return mu, std
    
    def reparameterize_gaussian(self, mu, std):
            #if self.training: 
            eps = torch.randn_like(std)  # [Nl,B,H]
            z = eps.mul(std).add_(mu)  # [Nl,B,H]
            #else:
            #    z = mu 
            return z 
    
    def kl_div(self, mu, std):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        
        k = mu.size(1)
        batch_size = mu.shape[0]
        mu_p = self.prior_mu.view(1, -1).expand(batch_size, -1)
        std_p = torch.nn.functional.softplus(self.prior_std.view(1, -1).expand(batch_size, -1))
        mu_diff = mu_p - mu
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()




    def renyi_divergence(self, mu, std, lmbda=1.1): #lmbda should be greater than 1

        """
         Calculate the Renyi divergence between two distributions
    
        """
        pairs_mu = list(zip(*itertools.combinations(torch.tensor(mu), 2)))
        pairs_sigma = list(zip(*itertools.combinations(torch.tensor(std), 2)))
        batch_size = torch.stack(pairs_mu[0]).shape[0]
     
        mu_diff = torch.stack(pairs_mu[0]) - torch.stack(pairs_mu[1])
        mu_diff_sq = torch.mul(mu_diff, mu_diff)

        t1 =  torch.sqrt((1 - lmbda) *  (torch.stack(pairs_sigma[0])**2) + lmbda * (torch.stack(pairs_sigma[1])**2)) #[B, 128])
        renyi_divergence = (
                (lmbda / 2) * torch.nansum(torch.div(mu_diff_sq, t1 ** 2), dim=1)
                + (1 / (1 - lmbda)) * torch.nansum(torch.log( torch.div(t1, ((torch.stack(pairs_sigma[0]) ** (1- lmbda)) * (torch.stack(pairs_sigma[1])**lmbda))  ) ), dim=1)
            ) # [B]
        
        
        renyi_divergence=renyi_divergence/torch.stack(pairs_mu[0]).shape[1]
        
        return renyi_divergence


    def bayesian_differential_privacy(self, mu, std, accountant, q, lmbda=1.1, **kwargs):
            
            
            batch_size = mu.shape[0]
            mu_p = self.prior_mu.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.prior_std.view(1, -1).expand(batch_size, -1))
            lsigma = std_p.mean().item() # estimate per sample

            rsigma = std.mean().item()
            pairs = list(zip(*itertools.combinations(torch.tensor(mu), 2)))
          

            accountant.accumulate(
                ldistr = (torch.stack(pairs[0]) , lsigma, None),
                rdistr = (torch.stack(pairs[1]) , rsigma, None),
                q = q,
                steps = 1
            )

            
   

   

    def forward(self, encoder_output, mask=None, **kwargs):
        """
        The latent layer for NVIB. Notice length comes in as NS and exits Nl (Ns+1) for the prior
        :param encoder_output:[B, Ns, P]
        :param mask: [B,Ns] boolean mask. True is padding
        :return: A tuple of outputs (z, mu, logvar, alpha, pi, memory_key_padding_mask)
                z: sample from the gaussian [B,Nl,P]
                pi: sampled from the dirichlet [B,Nl,1]
                mu: means from the latent layer [B,Nl,P]
                logvar: logged variances from the latent layer [B, Nl, P]
                alpha: logged psuedo-counts from the latent layer [B,Nl,heads]
                memory_key_padding_mask: from the latent layer [B,Nl]


        """
     
        encoder_output = self.mlp(encoder_output)
        mu, std = self.estimate(encoder_output)
      
        z = self.reparameterize_gaussian(mu, std)
       
    
        return z, mu, std 
        

    
