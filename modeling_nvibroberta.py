# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""PyTorch RoBERTa model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import ModelOutput, logging

# RoBERTa-specific imports
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaSelfAttention,
    RobertaSelfOutput,
    RobertaIntermediate,
    RobertaOutput,
    create_position_ids_from_input_ids,
)

# Your custom layer imports
from nvib_layer import Nvib

logger = logging.get_logger(__name__)

# --- Custom Output Dataclasses (model-agnostic, reused from your code) ---

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsAndRenyi(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kl_gaussian: Optional[torch.FloatTensor] = None
    kl_dirichlet: Optional[torch.FloatTensor] = None
    renyi_divergence: Optional[torch.FloatTensor] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndRenyi(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kl_gaussian: Optional[torch.FloatTensor] = None
    kl_dirichlet: Optional[torch.FloatTensor] = None
    renyi_divergence: Optional[torch.FloatTensor] = None

# --- Custom Attention and Model Blocks for RoBERTa ---

class RobertaDenoisingSelfAttention(nn.Module):
    # This is your BertDenoisingSelfAttention, adapted for the RoBERTa naming convention.
    # The internal logic is preserved as it operates on generic hidden states.
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size ({config.hidden_size}) not multiple of heads ({config.num_attention_heads})")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        nvib_tuple: Optional[dict] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(nvib_tuple[0]))
        value_layer = self.transpose_for_scores(self.value(nvib_tuple[0]))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Denoising Attention calculation (your logic)
        #############################
        # Denoising Attention TRAIN #
        #############################

        if True:
        #if self.training:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
            pi = nvib_tuple[1].unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1).permute(0, 1, 3, 2)
            pi_attention_mask = pi.le(0)
            pi = torch.clamp(pi.clone(), min=torch.finfo(pi.dtype).tiny)
            Z = nvib_tuple[0].unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1).permute(0, 1, 3, 2)
            l2_norm = (1 / (2 * math.sqrt(self.attention_head_size))) * ((torch.norm(Z, dim=2)).unsqueeze(2) ** 2)
            attention_scores += torch.log(pi) - l2_norm

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask + pi_attention_mask

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            #####
            attention_probs = self.dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            #######
            context_layer = torch.matmul(attention_probs, value_layer)
        
        ############################
        # Denoising Attention EVAL #
        ############################
        else:
            # tuple = (z, pi, mu, logvar, alpha, mask)
            prior_var_u = math.sqrt(self.attention_head_size)
            var = torch.exp(nvib_tuple[3])
            biased_var = var + prior_var_u
            mu = nvib_tuple[2]
            log_alpha = torch.log(
                nvib_tuple[4] + torch.finfo(nvib_tuple[4].dtype).eps)

            w_v = self.value.weight.view(
                self.num_attention_heads, self.attention_head_size, -1
            )  # (embed_dim, embed_dim) -> (heads, head_dim, embed_dim)
            b_v = self.value.bias.view(
                self.num_attention_heads, self.attention_head_size, 1
            )  # (embed_dim) -> (heads, head_dim, 1)

            w_k = self.key.weight.view(
                self.num_attention_heads, self.attention_head_size, -1
            )  # (embed_dim, embed_dim) -> (heads, head_dim, embed_dim)
            b_k = self.key.bias.view(
                self.num_attention_heads, self.attention_head_size, 1
            )  # (embed_dim) -> (heads, head_dim, 1)

            # Project the multihead query and bias into the embed_dim space from the head_dim space
            projected_u = torch.einsum("bhme, hep -> bhmp", query_layer, w_k)
            projected_bias = torch.einsum(
                "bhme, hep -> bhmp", query_layer, b_k)

            attention_scores = torch.einsum("bhmp, bnp -> bhmn", projected_u, mu / biased_var) + (
                projected_bias / math.sqrt(self.attention_head_size)
            )  # NOTE Scaling here

            #
            # (deep breath) include bias terms
            #

            bias = log_alpha.masked_fill(
                nvib_tuple[5].unsqueeze(-1), 0)  # [B,Ns,1]

            # L2 norm term
            bias -= 0.5 * ((torch.norm((mu / torch.sqrt(biased_var)), dim=-1)) ** 2).unsqueeze(
                -1
            )  # [B,Ns,1]

            # NOTE: V2 - we dont use the variance penalty term
            # Variance penalty term
            # bias -= torch.sum(
            #     torch.log(torch.sqrt(biased_var)).masked_fill_(nvib_tuple[5].unsqueeze(-1), 0),
            #     dim=-1,
            # ).unsqueeze(
            #     -1
            # )  # [B,Ns,1]

            # Bias term  - Repeat across heads [B, H,1, Ns]
            bias = bias.unsqueeze(1).repeat(
                1, self.num_attention_heads, 1, 1).permute(0, 1, 3, 2)

            attention_scores += bias

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            ################
            # # NOTE: This is very expensive to compute!
            # # Interpolate the pre-projected value "mu" and post-projected query "projected_u"
            # output = torch.einsum(
            #     "bhmn, bnp -> bhmp", attention_probs, (var / biased_var)
            # ) * projected_u + torch.einsum(
            #     "bhmn, bnp -> bhmp", attention_probs, ((prior_var_u / biased_var) * mu)
            # )
            # # Project into the correct space and add the bias. The bias is theoretically
            # # multiplied by the attention_probs, but it noramlises so we can just add it.
            # context_layer = torch.einsum("bhmp, hep -> bhme", output, w_v) + b_v.unsqueeze(
            #     0
            # ).permute(0, 1, 3, 2)
            ################
            # NOTE: V2 - we dont use the interpolation
            value_layer = self.transpose_for_scores(
                self.value((prior_var_u / biased_var) * mu))
            context_layer = torch.matmul(attention_probs, value_layer)
        
        

            
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class CustomRobertaSelfOutput(nn.Module):
    # This is your custom BertSelfOutput (without the residual connection), adapted for RoBERTa.
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # This is your key modification: no residual connection
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RobertaAttention(nn.Module):
    def __init__(self, config, accountant=None, position_embedding_type=None):
        super().__init__()
        self.config = config
        self.accountant = accountant
        
        if config.nvib_layers.pop(0):
            print("Adding NVIB to RoBERTa layer")
            self.nvib_layer = Nvib(
                size_in=config.hidden_size, size_out=config.hidden_size,
                prior_mu=config.prior_mus.pop(0) if config.prior_mu else None,
                prior_var=config.prior_vars.pop(0) if config.prior_var else None,
                prior_log_alpha=(config.prior_log_alphas.pop(0) if config.prior_log_alphas else None),
                prior_log_alpha_stdev=(config.prior_log_alpha_stdevs.pop(0) if config.prior_log_alpha_stdevs else None),
                delta=config.delta, nheads=config.num_attention_heads,
                alpha_tau=config.alpha_tau, mu_tau=config.mu_tau, stdev_tau=config.stdev_tau,
                learnable_prior=config.learnable_prior,
            )
            self.self = RobertaDenoisingSelfAttention(config, position_embedding_type=position_embedding_type)
            self.output = CustomRobertaSelfOutput(config) # Use your custom output layer
        else:
            self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
            self.output = RobertaSelfOutput(config) # Use the standard RoBERTa output layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        return_loss: Optional[bool] = True,
    ) -> Tuple[torch.Tensor]:

        if hasattr(self, "nvib_layer"):
            latent_tuple = self.nvib_layer(hidden_states, attention_mask.bool().squeeze(1)[:, 0, :])
            attention_mask_nvib = (
                latent_tuple[5].unsqueeze(1).unsqueeze(1).repeat(1, 1, attention_mask.size(2), 1)
                * torch.finfo(attention_mask.dtype).min
            )
            kl_gaussian = self.nvib_layer.kl_gaussian(mu=latent_tuple[2], logvar=latent_tuple[3], alpha=latent_tuple[4], memory_key_padding_mask=latent_tuple[5])
            kl_dirichlet = self.nvib_layer.kl_dirichlet(alpha=latent_tuple[4], memory_key_padding_mask=latent_tuple[5])
            renyi_divergence = self.nvib_layer.renyi_divergence(mu=latent_tuple[2], logvar=latent_tuple[3], alpha=latent_tuple[4])

            if not return_loss and self.accountant:
                self.nvib_layer.bayesian_differential_privacy(
                    mu=latent_tuple[2], logvar=latent_tuple[3], alpha=latent_tuple[4],
                    accountant=self.accountant, q=self.config.eval_batch_size / self.config.test_size,
                )
            
            self_outputs = self.self(
                hidden_states,
                attention_mask=attention_mask_nvib,
                head_mask=head_mask,
                output_attentions=output_attentions,
                nvib_tuple=latent_tuple,
            )
        else:
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
            )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        
        if hasattr(self, "nvib_layer"):
            outputs = outputs + (renyi_divergence, kl_gaussian, kl_dirichlet)
        return outputs


class RobertaLayer(nn.Module):
    def __init__(self, config, accountant=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config, accountant)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # Cross-attention logic would need to be ported if used
            raise NotImplementedError("Cross attention not implemented for this custom RobertaLayer")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        return_loss: Optional[bool] = True,
    ) -> Tuple[torch.Tensor]:
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
            return_loss=return_loss
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        # Feed-forward part
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config, accountant=None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config, accountant) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        return_loss: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentionsAndRenyi]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_kl_gaussian = ()
        all_kl_dirichlet = ()
        all_renyi_divergence = ()

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                return_loss=return_loss
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
            if hasattr(layer_module.attention, "nvib_layer"):
                all_renyi_divergence += (layer_outputs[-3],)
                all_kl_gaussian += (layer_outputs[-2],)
                all_kl_dirichlet += (layer_outputs[-1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions, all_kl_gaussian, all_kl_dirichlet, all_renyi_divergence] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentionsAndRenyi(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            kl_gaussian=all_kl_gaussian,
            kl_dirichlet=all_kl_dirichlet,
            renyi_divergence=all_renyi_divergence,
        )

class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output