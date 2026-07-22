# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import scipy as sp
import torch

from scipy.stats import t, binom
from scipy.special import logsumexp, gammaln

def scaled_renyi_divergence(lmbdas, ldistr, rdistr):
    """
        Computes scaled Renyi divergence D(p_left|p_right) between pairs of distributions.
        
        
        Parameters
        ----------
        lmbdas : array, required
            Orders of the Renyi divergence.
        ldistr : tuple or array, required
            Parameters of the left Gaussians and Dirichlet (mu_left , sigma_left, alpha_left).
        rdistr : tuple or array, required
            Parameters of the right Gaussians and Dirichlet (mu_right , sigma_right, alpha_right).
        
        Returns
        -------
        out : array
            Scaled Renyi divergences
    """
    
    lmu, lsigma, lalpha = ldistr
    rmu, rsigma, ralpha = rdistr
   

    if not (np.isscalar(lsigma) and np.isscalar(rsigma)):
        raise NotImplementedError("Not implemented for Gaussians with diagonal or full covariances.")
  
    #  computes scaled Renyi divergence between pairs of Gaussian distributions (with the same variance)
    if lsigma==rsigma and lalpha is None and ralpha is None:
        distances = lmu - rmu
        # ensure it is a tensor
        if np.isscalar(distances):
            distances = torch.tensor(distances)
        distances = torch.norm(distances, p=2, dim=-1).view(-1).to(lmbdas)
        return torch.ger(distances**2, lmbdas * (lmbdas - 1) / (2 * lsigma**2))
    
    # computes scaled Renyi divergence between pairs of distributions (Gaussian and Dirichlet)
    else:
        if lalpha is not None and ralpha is not None:

            renyi_gaussian_nvib = scaled_renyi_gaussian_nvib(lmbdas, rmu, lmu, rsigma, lsigma)
            renyi_dirichlet_nvib = scaled_renyi_dirichlet_nvib(lmbdas, ralpha, lalpha)
            return renyi_gaussian_nvib + renyi_dirichlet_nvib 
        
        # computes scaled Renyi divergence between pairs of Gaussian distributions (with different variances)
        t1 = torch.sqrt((1 - lmbdas) * lsigma**2 + lmbdas * rsigma**2)

        distances = lmu - rmu 
        if np.isscalar(distances):
            distances = torch.tensor(distances)
        distances = torch.norm(distances, p=2, dim=-1).view(-1).to(lmbdas)
        term1 = torch.ger(distances**2, lmbdas * (lmbdas - 1) / (2 * t1**2))
       
        term1=torch.nan_to_num(term1, nan=0.0)
        term2 = ((lmbdas - 1)/(1-lmbdas))* torch.log(t1/(lsigma**(1-lmbdas)*rsigma**lmbdas))

        term2=torch.nan_to_num(term2, nan=0.0)
       
        return (term1+term2)/rmu.shape[1]

        

def scaled_renyi_dirichlet_nvib(lmbdas, lalpha, ralpha):
    """
        Computes scaled Renyi divergence D(p_left|p_right) between pairs of Dirichlet distributions.
        
        
        Parameters
        ----------
        lmbdas : array, required
            Orders of the Renyi divergence.
        lalpha : tuple or array, required
            Parameters of the left Gaussians (mu_left [n_samples * n_features], sigma [scalar]).
        ralpha : tuple or array, required
            Parameters of the right Gaussians (mu_right [n_samples * n_features], sigma [scalar]).
        
        Returns
        -------
        out : array
            Scaled Renyi divergences
    """

    lalpha=lalpha.to(lmbdas)
    ralpha=ralpha.to(lmbdas)
    if ralpha.ndim >= 2:
        k0 = torch.full((ralpha.size(0),lmbdas.size(0)), ralpha.size(1)) # [B]
    else:
        k0 = torch.full((lalpha.size(0),lmbdas.size(0)), lalpha.size(1)) 
    #n=k0

     
    if torch.all(lalpha == 1):
        alpha0_q = torch.sum(ralpha, 1).squeeze(-1).to(torch.float64)  # scalar
        # Conditional prior: assume self.prior_log_alpha is a scalar tensor or Python float
        alpha0_p = lalpha.repeat(ralpha.size(0)).to(torch.float64)  # scalar
        
    elif torch.all(ralpha == 1):
      
        alpha0_q = ralpha.repeat(lalpha.size(0)).to(torch.float64) 
        # Conditional prior: assume self.prior_log_alpha is a scalar tensor or Python float
        alpha0_p = torch.sum(lalpha, 1).squeeze(-1).to(torch.float64)  # scalar 
        
    else:
        alpha0_q = torch.sum(ralpha, 1).squeeze(-1).to(torch.float64)
            # Conditional prior: assume self.prior_log_alpha is a scalar tensor or Python float
        alpha0_p = torch.sum(lalpha, 1).squeeze(-1).to(torch.float64)  # scalar 
    

    formula1 = (
        -(1 / (lmbdas.view(1, -1)  - 1)) * torch.lgamma(lmbdas.view(1,-1) * alpha0_q.view(-1, 1)  - ((lmbdas.view(1, -1)  - 1) * alpha0_p.view(-1, 1) ))
        - torch.lgamma(alpha0_p.view(-1, 1) ) 
        + (lmbdas.view(1, -1)  / (lmbdas.view(1, -1)  - 1)) * torch.lgamma(alpha0_q.view(-1, 1) )
    )  # scalar
    formula1 = torch.nan_to_num(formula1, nan=0.0, posinf=0, neginf=0)
    formula1/=k0
        

    alpha0_q_expanded = alpha0_q.unsqueeze(-1).unsqueeze(-1)  #  Shape [B, 1, 1]
    alpha0_p_expanded = alpha0_p.unsqueeze(-1).unsqueeze(-1)  #  Shape [B, 1, 1]

    # Compute adjusted_alpha: [129, 1]
    adjusted_alpha = (alpha0_p_expanded * ralpha) / alpha0_q_expanded  # [B, 129, 1]

    formula2 = ( 
        (1 / (lmbdas - 1)) * torch.lgamma((lmbdas * ralpha) - ((lmbdas - 1) * adjusted_alpha))
        + torch.lgamma(adjusted_alpha)
        - (lmbdas / (lmbdas - 1)) * torch.lgamma(ralpha)
    )
    formula2 = torch.nansum(formula2, 1)
    formula2 /= k0
    formula2 = torch.nan_to_num(formula2, nan=0.0, posinf=0, neginf=0)
    
    renyi = formula1 + formula2
    return renyi


def scaled_renyi_gaussian_vib(lmbdas, rmu, lmu, rsigma, lsigma):

    terms_list = []
    for lmbda in lmbdas:
        t1 = torch.sqrt((1 - lmbda) * lsigma**2 + lmbda * rsigma**2)                                                        
        term1 = (lmbda *(lmbda - 1)/ 2) * torch.nansum((rmu - lmu) ** 2 / t1 ** 2 , dim=1) 
        term1=torch.nan_to_num(term1, nan=0.0) 
        term2 = ((lmbda - 1)/(1-lmbda))* torch.log(t1/(lsigma**(1-lmbda)*rsigma**lmbda)) 
        term2=torch.nan_to_num(term2, nan=0.0) 
        terms_list.append(term1+term2)                          

    renyi = torch.stack(terms_list, dim=1).to(lmbdas)  # shape: [8, 4]
   
    
    return renyi


def scaled_renyi_gaussian_nvib(lmbdas, rmu, lmu, rsigma, lsigma):
    
    if rmu.ndim >= 2:
        k0 = torch.full((rmu.size(0),lmbdas.size(0)), rmu.size(1))  
    else:
        k0 = torch.full((lmu.size(0),lmbdas.size(0)), lmu.size(1)) 
    
    # Reshape lmbdas for broadcasting
    lmbdas_reshaped = lmbdas.view(1, 1, -1) # [1, 1, A]

    # (1 - α) * lsigma² + α * rsigma²
    t1 = torch.sqrt((1 - lmbdas_reshaped) * lsigma**2 + lmbdas_reshaped * rsigma**2)  # [1, 1, A]

    # (μ₁ - μ₀)², broadcasted
    mu_diff_sq = ((rmu - lmu).unsqueeze(-1) ** 2).to(lmbdas)  # [B, 129, 512, 1]

    # t1 squared, broadcasted
    t1_sq = t1.view(1, 1, 1, -1) ** 2  # [1, 1, 1, A]

    # Term 1
    term1_coeff = ((lmbdas * (lmbdas - 1)) / 2) # [A]
    term1= term1_coeff.view(1, 1, -1) * torch.nanmean(mu_diff_sq / t1_sq, dim=2)  # [B, 129, A]
    term1=torch.nan_to_num(term1, nan=0.0) 
    # Term 2
    term2 = ((lmbdas-1) / (1 - lmbdas)).view(1, 1, -1) * torch.log(
        t1 / (lsigma ** (1 - lmbdas_reshaped) * rsigma ** lmbdas_reshaped)
    )  # [1, 1, A]
    term2= term2.expand_as(term1)  # [B, 129, A]
    term2=torch.nan_to_num(term2, nan=0.0) 
    # Combine terms and sum over time
    renyi = torch.nansum(term1 + term2, dim=1)  # [B, A]
    renyi/=k0
    return renyi