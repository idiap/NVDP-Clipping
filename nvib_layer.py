# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import math

import torch
import torch.nn as nn
import numpy as np
import itertools


# Note:
# Ns: Source length
# Nt: target length
# Nl: latent length
# B: batch size
# H: hidden dimension


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


class Nvib(nn.Module):
    """
    A Nonparameteric variational information bottleneck layer
    """

    def __init__(
        self,
        size_in,
        size_out,
        prior_mu=None,
        prior_var=None,
        prior_log_alpha=None,
        prior_log_alpha_stdev=None,
        delta=1,
        nheads=1,
        alpha_tau=None,
        stdev_tau=None,
        mu_tau=None,
        learnable_prior=False, #True
        perform_latent_clipping= False, 
        max_latent_norm = 10,    
        max_alpha = 1.5,
        **kwargs
    ):
        super().__init__()

        # Dimensionality:
        # size_in: P
        # size_out: P
        # nheads: H
        # head_dim: P/H = D
        # length: Ns
        # latent_length: Nl
        # batch: B

        # Prior mean [P]
        if prior_mu is not None:
            self.prior_mu = nn.Parameter(prior_mu, requires_grad=learnable_prior)
        else:
            self.prior_mu = nn.Parameter(torch.zeros(size_in), requires_grad=learnable_prior)

        # Prior variance [P]
        if prior_var is not None:
            self.prior_var = nn.Parameter(prior_var, requires_grad=False)
        else:
            self.prior_var = nn.Parameter(torch.ones(size_in), requires_grad=False)

        # Prior log alpha [1]
        if prior_log_alpha is not None:
            self.prior_log_alpha = nn.Parameter(prior_log_alpha, requires_grad=False)
        else:
            self.prior_log_alpha = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Prior log alpha standard deviation (important for initialisation) [1]
        if prior_log_alpha_stdev is not None:
            self.prior_log_alpha_stdev = nn.Parameter(prior_log_alpha_stdev, requires_grad=False)
        else:
            self.prior_log_alpha_stdev = nn.Parameter(torch.ones(1), requires_grad=False)

        # Conditional prior delta for the dirichlet KL divergence
        self.delta = float(delta)

        # Layers for parameters
        self.size_in = size_in
        self.size_out = size_out
        self.d = int(size_in / nheads)  # dimension of the head
        self.alpha_activation = Exponential()  # projection for alphas
        self.mu_proj = nn.Linear(size_in, size_out)  # Project to mean
        self.logvar_proj = nn.Linear(size_in, size_out)  # Project log variance
        self.alpha_proj = Quadratic(size_in, 1)  # Project to model size
        self.nheads = nheads  # number of heads

        # Initialisation parameters - 0 is the prior 1 is the posterior
        self.alpha_tau = alpha_tau if alpha_tau is not None else 1 # 4
        self.stdev_tau = stdev_tau if stdev_tau is not None else 1 # 0.1
        self.mu_tau = mu_tau if mu_tau is not None else 1

        # Initialisation parameters for the latent vector clipping  
        self.perform_latent_clipping = perform_latent_clipping
        self.max_latent_norm = max_latent_norm 
        self.max_alpha = max_alpha

        # Initialise the parameters
        self.init_parameters()

    def init_parameters(self):
        """
        Initialise parameters
        """
        # Initialise mu projection
        eye_scaled_(self.mu_proj.weight, self.mu_tau)
        init_vector_(self.mu_proj.bias, self.prior_mu * (1 - self.mu_tau))

        # Initialise logvar projection
        nn.init.constant_(self.logvar_proj.weight, 0)
        init_vector_(
            self.logvar_proj.bias,
            torch.log(
                (torch.sqrt(self.prior_var) * self.stdev_tau)
                ** 2  # Controls the standard deviation
                + torch.finfo(self.prior_var.dtype).tiny
            ),  # nonzero
        )

        # Initialise alpha projection
        nn.init.constant_(self.alpha_proj.weight_quadratic, 1 / (2 * math.sqrt(self.d)))
        nn.init.constant_(self.alpha_proj.weight_linear, 0)
        init_vector_(
            self.alpha_proj.bias,
            self.prior_log_alpha_stdev * (self.alpha_tau),  # Standard deviation of log alpha
        )

    def reparameterize_gaussian(self, mu, logvar):
        """
        Reparameterise for gaussian
        Train = sample
        Test = mean

        :param mu: means [Nl,B,P]
        :param logvar: logged variances [Nl,B,P]
        :return: z: sample from a gaussian distribution or mean [Nl,B,P]
        """
        #if self.training:
        std = torch.exp(0.5 * logvar)  # [Nl,B,P]
        eps = torch.randn_like(std)  # [Nl,B,P]
        z = eps.mul(std).add_(mu)  # [Nl,B,P]
        #else:
        #    z = mu  # [Nl,B,P]
        return z  # [Nl,B,P]

    def reparameterize_dirichlet(self, alpha, mask):
        """
        Reparameterise for dirichlet
        Train = sample
        Test = mean

        :param alpha: psuedo-counts [B,Nl,1]
        :param mask: Mask for the latent space [B,Nl]
        :return pi: dirichlet probability [B,Nl,1]
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
        # Sample from the gamma distribution
        #if self.training:
        #gamma_dist = torch.distributions.Gamma(alpha, torch.ones_like(alpha))
        #gammas = gamma_dist.rsample()
        
        #ADDED THIS INSTEAD
        # rsample() can cause NaNs when the alphas are too large and too small.
        #    We keep the proportion and scale it. Also we need to clamp the masked values
        if mask is not None:
            alpha_sum = torch.sum(alpha.masked_fill(
                mask, 0), dim=1, keepdim=True)
        else:
            alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        # Get proportions
        alpha = alpha / alpha_sum
        # Clamp the proportion to prevent underflow - Clamp the sum of alpha to prevent overflow.
        alpha = torch.clamp(alpha, min=1e-23) * \
            torch.clamp(alpha_sum, max=1e8)
        # Clamp the masked values!
        if mask is not None:
            alpha.masked_fill_(mask, 1e-15)

        gamma_dist = torch.distributions.Gamma(
            alpha, torch.ones_like(alpha))
        gammas = gamma_dist.rsample()

        # Testing the alphas don't have noise
        #else:
        #    gammas = alpha

        if mask is not None:
            gammas.masked_fill_(mask, 0)
        normalising_sum = torch.sum(gammas, 1, keepdim=True) + torch.finfo(gammas.dtype).tiny
        pi = torch.div(gammas, normalising_sum)

        return pi

    def kl_gaussian(self, mu, logvar, alpha, mask=None, **kwargs):
        """
        KL Loss for the Gaussian component with expected K
        :param mu: mean [Nl,B,P]
        :param logvar: logged variance [Nl,B,P]
        :param alpha: psuedo count weight [Nl,B,1]
        :param mask: boolean mask [B,Nl]
        :return: KL [B]
        """
        # Scaling
        # Total number of vectors sampled
        if mask is not None:
            k0 = torch.sum(~mask, 1)  # [B]
        else:
            k0 = torch.full((alpha.size(0),), alpha.size(1), device=alpha.device)  # [B]
        # Input length
        n = k0  # / self.kappa  # [B]

        alpha = alpha.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else alpha
       
        alpha0_q = torch.sum(alpha, dim=1, keepdim=True)  # [B,1]
        expected_pi = (alpha / alpha0_q).squeeze(-1)  # [B,Nl]

        # KL between univariate Gaussians
        var_ratio = logvar.exp() / self.prior_var
        t1 = (mu - self.prior_mu) ** 2 / self.prior_var
        kl = var_ratio + t1 - 1 - var_ratio.log()
        kl = kl.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else kl

        # Mean over embedding dimension
        kl = torch.mean(kl, -1)  # [B, Nl]

        # Scale and sum over sentence length dimension
        kl = 0.5 * k0 * torch.sum(kl * expected_pi, -1)  # [B]
        kl /= n

        return kl

    def kl_dirichlet(self, alpha, mask=None, **kwargs):
        """
        The regularisation for the dirichlet component with expected K

        :param alpha: k dimensional psuedo counts [B,Nl,1]
        :param mask: boolean mask [B,Nl]
        :return: Kl [B]

        Nota Bene: digamma and lgamma cannot be zero
        """

        # Total number of vectors sampled
        if mask is not None:
            k0 = torch.sum(~mask, 1)  # [B]
        else:
            k0 = torch.full((alpha.size(0),), alpha.size(1), device=alpha.device)  # [B]

        # k0 = 1
        # Input length
        n = k0  # / self.kappa  # [B]

        alpha = alpha.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else alpha
       
        alpha0_q = torch.sum(alpha, 1).squeeze(-1).to(torch.float64)  # [B]
        # Conditional prior with lower bound. Sentence length weighted by delta
        alpha0_p = (
            torch.exp(self.prior_log_alpha).repeat(alpha.size(0)) + self.delta * (n - 1)
        ).to(
            torch.float64
        )  # [B]

        # KL between two dirichlet distributions
        kl = (
            torch.lgamma(alpha0_q)
            - torch.lgamma(alpha0_p)
            + (alpha0_q - alpha0_p) * (-torch.digamma(alpha0_q) + torch.digamma(alpha0_q / k0))
            + k0 * (torch.lgamma(alpha0_p / k0) - torch.lgamma(alpha0_q / k0))
        ) / n

        return kl
    
    def renyi_divergence(self, mu, logvar, alpha, mask = None, lmbda=1.1, kappa =1, **kwargs): 
      
        """
         Calculate the Renyi divergence between two distributions
        :param mu: mean [Nl,B,P]
        :param logvar: logged variance [Nl,B,P]
        :param alpha: psuedo count weight [Nl,B,1]
        :param mask: boolean mask [B,Nl]
        :return: renyi [B]


        #self.prior_mu: [P] 
        #self.prior_var: [P]
        #self.prior_log_alpha: [1]

        """


        # Total number of vectors sampled
    
        alpha = alpha.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else alpha

        # elminitate small alphas
        theta = 1e-4
        mask_alpha = alpha > theta 
        alpha = torch.where(alpha > theta, alpha, torch.zeros_like(alpha))

        pairs_mu = list(zip(*itertools.combinations(torch.tensor(mu), 2)))
        pairs_alpha = list(zip(*itertools.combinations(torch.tensor(alpha), 2)))
        pairs_sigma = list(zip(*itertools.combinations(torch.tensor(torch.exp(0.5 * logvar)), 2)))

        if mask is not None:
            k0 = torch.sum(~mask, 1)  # [B]
        else:
            k0 = torch.full((torch.stack(pairs_alpha[0]).size(0),), torch.stack(pairs_alpha[0]).size(1), device=alpha.device)  # [B]


        # k0 = 1
        # Input length
        n = k0  # / self.kappa  # [B]
        
        alpha0_q= torch.sum(torch.stack(pairs_alpha[0]), 1).squeeze(-1).to(torch.float64)  # [B]

        alpha0_p = torch.sum(torch.stack(pairs_alpha[1]), 1).squeeze(-1).to(torch.float64)  # scalar 
    
       
        formula1 =(
            -(1 / (lmbda - 1)) * torch.lgamma(lmbda * alpha0_q - ((lmbda - 1) * alpha0_p))
            - torch.lgamma(alpha0_p) 
            + (lmbda / (lmbda - 1)) * torch.lgamma(alpha0_q)

        ) / n #[B]

       
        alpha0_q_expanded = alpha0_q.unsqueeze(-1).unsqueeze(-1)  #  [B, 1, 1]
        alpha0_p_expanded = alpha0_p.unsqueeze(-1).unsqueeze(-1)  #  [B, 1, 1]

        # Compute the required term
        adjusted_alpha = (alpha0_p_expanded * torch.stack(pairs_alpha[0])) / alpha0_q_expanded  # Shape [B, 129, 1]

        formula2 = ( 
        (1 / (lmbda - 1)) * torch.lgamma( (lmbda *  torch.stack(pairs_alpha[0]) ) - ((lmbda - 1) * adjusted_alpha))
        + torch.lgamma( adjusted_alpha )
        - (lmbda / (lmbda - 1)) * torch.lgamma(torch.stack(pairs_alpha[0]))
        )
     
        formula2 = torch.nanmean(formula2, -1)
        formula2 = torch.nansum(formula2, -1)
        formula2 /= k0


        t1 =  torch.sqrt((1 - lmbda) *  torch.stack(pairs_sigma[1])**2 + (lmbda *  torch.stack(pairs_sigma[0])**2)) #[B, 513, 768])

        formula3 = (
                (lmbda / 2) * torch.nanmean(torch.div((torch.stack(pairs_mu[0]) - torch.stack(pairs_mu[1]))**2, t1 ** 2), dim=2)
                + (1 / (1 - lmbda)) * torch.nanmean(torch.log( torch.div(t1, ((( torch.stack(pairs_sigma[0])) ** lmbda) * (( torch.stack(pairs_sigma[1]) )**(1-lmbda)))  ) ), dim=2)
            ) # [B, seq_len]
        formula3 = formula3.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else formula3
        
        formula3=torch.nansum(formula3, -1)
        formula3/=n
        
        renyi_divergence = formula1  + formula2 + formula3

        return renyi_divergence


    
    def bayesian_differential_privacy(self, mu, logvar, alpha, accountant, q, mask=None, lmbda=1.1, **kwargs):
            
            
        if mask is not None:
            k0 = torch.sum(~mask, 1)  # [B]
        else:
            k0 = torch.full((alpha.size(0),), alpha.size(1), device=alpha.device)  # [B]

        # k0 = 1
        # Input length
        n = k0  # / self.kappa  # [B]
        
        alpha = alpha.masked_fill(mask.unsqueeze(-1), 0) if mask is not None else alpha
        # elminitate small alphas
        theta = 1e-4
        mask_alpha = alpha > theta 
        alpha = torch.where(alpha > theta, alpha, torch.zeros_like(alpha))

        # Expand mask_alpha to match mu's shape
        mask_alpha_broadcasted = mask_alpha.expand(-1, -1, mu.size(2))
        # Apply the mask
        mu = mu * mask_alpha_broadcasted
        logvar = logvar * mask_alpha_broadcasted


       
        lsigma = self.prior_var.sqrt().mean().item() # estimate per sample
        rsigma = torch.exp(0.5 * logvar).mean().item()
    
        lalpha = torch.exp(self.prior_log_alpha)
        

        pairs_mu = list(zip(*itertools.combinations(torch.tensor(mu), 2)))
        pairs_alpha = list(zip(*itertools.combinations(torch.tensor(alpha), 2)))

        
        accountant.accumulate(
            ldistr = (torch.stack(pairs_mu[0]), lsigma, torch.stack(pairs_alpha[0])),
            rdistr = (torch.stack(pairs_mu[1]), rsigma, torch.stack(pairs_alpha[1])),
            q = q,
            steps = 1
        )
        
    def clip_latent_norms(self, mu, logvar, alpha, mu_max_norm=3, lamda=1.1):
        """
        In-place clips the L2-norm of each latent vector.
        Assumes latent_tensor is of shape [Batch, SeqLen, Dim].
        """

        ### clip mu ###

        # Calculate norm over the last dimension (the embedding dimension)
        mu_norms = torch.norm(mu, p=2, dim=2, keepdim=True)

        mu_norms = mu_norms.clamp(min=1e-8)

        mask = mu_norms > self.max_latent_norm

        # Add a small epsilon to avoid division by zero
        scale = self.max_latent_norm / mu_norms

        mu_clipped = torch.where(mask, mu * scale, mu)
    
        
        ### clip sigma ###

        sigma_min = math.sqrt((lamda - 1) / lamda) * torch.sqrt(self.prior_var)
        logvar_min = 2 * torch.log(torch.tensor(sigma_min, device=logvar.device))
        logvar_clipped= logvar.clamp(min=logvar_min)

        ### clip alpha ###
        alpha_clipped = alpha.clamp(min=0, max=self.max_alpha)
        
        
        return mu_clipped, logvar_clipped, alpha_clipped
       
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
        # Get batch size
        B = encoder_output.size(0) # torch.Size([96, 512, 768]), mask shape is [96, 768]


        # Project to mean, log variance and log alpha
        mu = self.mu_proj(encoder_output)
        logvar = self.logvar_proj(encoder_output)
        alpha = self.alpha_activation(self.alpha_proj(encoder_output)) # torch.Size([96, 512, 1])

        # Include the priors in the first position of the latent embeddings
        mu = torch.cat((self.prior_mu.repeat(B, 1, 1), mu), 1)
        logvar = torch.cat((torch.log(self.prior_var).repeat(B, 1, 1), logvar), 1)
        alpha = torch.cat((self.alpha_activation(self.prior_log_alpha).repeat(B, 1, 1), alpha), 1) # torch.Size([96, 513, 1])
        mask = (
            torch.cat((torch.zeros((B, 1), dtype=torch.bool, device=mask.device), mask), 1)
            if mask is not None
            else None
        )
        

        if self.perform_latent_clipping:
            mu, logvar, alpha  = self.clip_latent_norms(mu, logvar, alpha)


        # Reparameterise parameters
        z = self.reparameterize_gaussian(mu, logvar)
        pi = self.reparameterize_dirichlet(alpha, mask)
        
        return z, pi, mu, logvar, alpha, mask
    
