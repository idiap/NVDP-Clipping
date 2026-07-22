# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass
import copy 
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss



from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

from transformers.modeling_outputs import SequenceClassifierOutput

# Bert-specific imports
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertEncoder,
    BertPooler,
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION as _BERT_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    _CONFIG_FOR_DOC as _BERT_CONFIG_FOR_DOC, 
    _SEQ_CLASS_EXPECTED_OUTPUT as _BERT_SEQ_CLASS_EXPECTED_OUTPUT, 
    _SEQ_CLASS_EXPECTED_LOSS as _BERT_SEQ_CLASS_EXPECTED_LOSS,  
    BERT_INPUTS_DOCSTRING,

)


# RoBERTa-specific imports
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaClassificationHead,
  
)


from modeling_nvibbert import BertEncoder as NVIB_BertEncoder
from modeling_nvibbert import BertPooler as NVIB_BertPooler
from modeling_nvibroberta import RobertaEncoder as NVIB_RobertaEncoder
from modeling_nvibroberta import RobertaPooler as NVIB_RobertaPooler

from vib_layer import Vib 



class BertForSequenceClassificationWithExtraLayer(BertPreTrainedModel):
    def __init__(self, config, accountant=None):
        
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.accountant = accountant
        
        self.bert = BertModel(config)
        
       
        # add NVIB Layer:
        if self.config.NVIB:
            # Define Nvib layer
            new_layer_config = copy.deepcopy(config) 
            new_layer_config.num_hidden_layers = 1
            new_layer_config.nvib_layers = [0]

            new_layer_config.nvib_layers = [
                i in new_layer_config.nvib_layers for i in range(new_layer_config.num_hidden_layers)
            ]
            
            self.nvibbert = NVIB_BertEncoder(new_layer_config, accountant) 
            self.pooler = NVIB_BertPooler(config)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        elif self.config.VIB:
            new_layer_config = copy.deepcopy(config) 
            new_layer_config.num_hidden_layers = 1
            
            self.vibbert = BertEncoder(new_layer_config) 
            self.pooler = BertPooler(config)
            self.vib_layer = Vib(config.ib_dim, config.hidden_dim, config.hidden_size, config.activation)
            self.classifier = nn.Linear(config.ib_dim, config.num_labels)
            
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
           
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()


            

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_BERT_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_BERT_CONFIG_FOR_DOC,
        expected_output=_BERT_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_BERT_SEQ_CLASS_EXPECTED_LOSS,
    )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        scaling_factor: Optional[float] = 1,
        return_loss:  Optional[bool] = True
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
       
        if self.config.NVIB:

            
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
            
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length )), device=device)

            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
            
            if self.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (
                    encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(
                        encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

           
            sequence_output = outputs.last_hidden_state

            nvibbert_sequence_output= self.nvibbert(
                sequence_output, 
                attention_mask= extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                return_loss=return_loss,
            )
           
             
            # added for averaging last hidden state instead of taking only the CLS token: 
            last_hidden_state = nvibbert_sequence_output.last_hidden_state
            if attention_mask is not None:
                 # Expand attention mask dimensions for broadcasting
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # Sum of hidden states (masked) and divide by the sum of the mask for averaging
                sum_hidden = torch.sum(last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                average_output = sum_hidden / sum_mask
            else:
                average_output = last_hidden_state.mean(dim=1)
            
            pooled_output = average_output
            
            
            # Calculate KL divergences
           
            klg_loss = torch.stack(nvibbert_sequence_output.kl_gaussian, dim=0).mean()
            kld_loss = torch.stack(nvibbert_sequence_output.kl_dirichlet, dim=0).mean()

            renyi_raw = torch.stack(nvibbert_sequence_output.renyi_divergence, dim=0)
            mean_renyi=renyi_raw.mean()
            std_renyi = renyi_raw.std(unbiased=False)
            z_scores = (renyi_raw - mean_renyi) / std_renyi
            renyi_filtered = torch.where(torch.abs(z_scores) > 0.5, torch.tensor(0.0), renyi_raw)
            renyi_divergence = (renyi_filtered.max(), renyi_filtered.mean())
            

        elif self.config.VIB:

            sequence_output = outputs.last_hidden_state

            vibbert_sequence_output= self.vibbert(sequence_output)

             
            # added for averaging last hidden state instead of taking only the CLS token: 
            last_hidden_state = vibbert_sequence_output.last_hidden_state
            if attention_mask is not None:
                 # Expand attention mask dimensions for broadcasting
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # Sum of hidden states (masked) and divide by the sum of the mask for averaging
                sum_hidden = torch.sum(last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                average_output = sum_hidden / sum_mask
            else:
                average_output = last_hidden_state.mean(dim=1)

            
            pooled_output = average_output
            latent_tuple = self.vib_layer(pooled_output)

            pooled_output = latent_tuple[0]
            
            klg_loss = self.vib_layer.kl_div(
                mu = latent_tuple[1],
                std = latent_tuple[2]
            )
            
            kld_loss = None

            renyi_raw =  self.vib_layer.renyi_divergence(
                mu = latent_tuple[1],
                std = latent_tuple[2]
            )
            mean_renyi=renyi_raw.mean()
            std_renyi = renyi_raw.std(unbiased=False)
            z_scores = (renyi_raw - mean_renyi) / std_renyi
            renyi_filtered = torch.where(torch.abs(z_scores) > 0.5, torch.tensor(0.0), renyi_raw)
            renyi_divergence = (renyi_filtered.max(), renyi_filtered.mean())
            
             
            
            # for accumlating the privacy and calculating it at the end of the epoch in the model.py class
            if not return_loss:
                self.vib_layer.bayesian_differential_privacy(
                    mu = latent_tuple[1],
                    std = latent_tuple[2],
                    accountant = self.accountant,
                    q = self.config.eval_batch_size/self.config.test_size ,
                )
            
        else:

            klg_loss = None
            kld_loss = None
            renyi_divergence = None
             
            #added for averaging last hidden state instead of taking only the CLS token: 
            last_hidden_state = outputs[0]
            if attention_mask is not None:
                 # Expand attention mask dimensions for broadcasting
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # Sum of hidden states (masked) and divide by the sum of the mask for averaging
                sum_hidden = torch.sum(last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                average_output = sum_hidden / sum_mask
            else:
                average_output = last_hidden_state.mean(dim=1)

            pooled_output = average_output
            

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None and return_loss:
            task_loss = None
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                task_loss_fct = MSELoss()
                if self.num_labels == 1:
                    task_loss = task_loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    task_loss = task_loss_fct(logits, labels)
            
            elif self.config.problem_type == "single_label_classification":
                task_loss_fct = CrossEntropyLoss()
                task_loss = task_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            elif self.config.problem_type == "multi_label_classification":
                task_loss_fct = BCEWithLogitsLoss()
                task_loss = task_loss_fct(logits, labels)

            if self.config.NVIB:
                loss = task_loss + self.config.NVIB_lambda_klg * scaling_factor * klg_loss + \
                    self.config.NVIB_lambda_kld * scaling_factor * kld_loss 
            elif self.config.VIB:
                loss = task_loss + self.config.VIB_lambda_klg * scaling_factor * klg_loss 
            else:
                loss = task_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputAndRenyi(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            task_loss = task_loss,
            klg_loss = klg_loss,
            kld_loss = kld_loss,
            renyi_divergence = renyi_divergence,
        ) if loss is not None else SequenceClassifierOutputAndRenyi(logits=logits, renyi_divergence = renyi_divergence) # for predictions



class RobertaForSequenceClassificationWithExtraLayer(RobertaPreTrainedModel):

    def __init__(self, config, accountant=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.accountant = accountant

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Logic for adding NVIB, or other custom layers
        if self.config.NVIB:
            new_layer_config = copy.deepcopy(config)
            new_layer_config.num_hidden_layers = 1
            new_layer_config.nvib_layers = [0]  
            
            new_layer_config.nvib_layers = [

                i in new_layer_config.nvib_layers for i in range(new_layer_config.num_hidden_layers)

            ]

            self.nvibbert = NVIB_RobertaEncoder(new_layer_config, accountant)
            self.pooler = NVIB_RobertaPooler(config)  # Assuming this is generic
            self.classifier = RobertaClassificationHead(config)

        else:
            self.classifier = RobertaClassificationHead(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()
    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,#is removed for RoBERTa
        position_ids: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        scaling_factor: Optional[float] = 1,
        return_loss: Optional[bool] = True
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, 
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

       

        if self.config.NVIB:

            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
            
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length )), device=device)

            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
            
            if self.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (
                    encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(
                        encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

           
            sequence_output = outputs.last_hidden_state

            nvibbert_sequence_output= self.nvibbert(
                sequence_output, 
                attention_mask= extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                return_loss=return_loss,
            )
           

             
            # added for averaging last hidden state instead of taking only the CLS token: 
            last_hidden_state = nvibbert_sequence_output.last_hidden_state
            ''' 
            if attention_mask is not None:
                 # Expand attention mask dimensions for broadcasting
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # Sum of hidden states (masked) and divide by the sum of the mask for averaging
                sum_hidden = torch.sum(last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                average_output = sum_hidden / sum_mask
            else:
                average_output = last_hidden_state.mean(dim=1)
            '''

            
            pooled_output = last_hidden_state
            
            
            # Calculate KL divergences
           
            klg_loss = torch.stack(nvibbert_sequence_output.kl_gaussian, dim=0).mean()
            kld_loss = torch.stack(nvibbert_sequence_output.kl_dirichlet, dim=0).mean()

            renyi_raw = torch.stack(nvibbert_sequence_output.renyi_divergence, dim=0)
            mean_renyi=renyi_raw.mean()
            std_renyi = renyi_raw.std(unbiased=False)
            z_scores = (renyi_raw - mean_renyi) / std_renyi
            renyi_filtered = torch.where(torch.abs(z_scores) > 0.5, torch.tensor(0.0), renyi_raw)
            renyi_divergence = (renyi_filtered.max(), renyi_filtered.mean())
                
            
        else:

            klg_loss = None
            kld_loss = None
            renyi_divergence = None
            pooled_output  = outputs[0]
            

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None and return_loss:
            task_loss = None
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                task_loss_fct = MSELoss()
                if self.num_labels == 1:
                    task_loss = task_loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    task_loss = task_loss_fct(logits, labels)
            
            elif self.config.problem_type == "single_label_classification":
                task_loss_fct = CrossEntropyLoss()
                task_loss = task_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            elif self.config.problem_type == "multi_label_classification":
                task_loss_fct = BCEWithLogitsLoss()
                task_loss = task_loss_fct(logits, labels)

            if self.config.NVIB:
                loss = task_loss + self.config.NVIB_lambda_klg * scaling_factor * klg_loss + \
                    self.config.NVIB_lambda_kld * scaling_factor * kld_loss 
            else:
                loss = task_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


        return SequenceClassifierOutputAndRenyi(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            task_loss = task_loss,
            klg_loss = klg_loss,
            kld_loss = kld_loss,
            renyi_divergence = renyi_divergence,
        ) if loss is not None else SequenceClassifierOutputAndRenyi(logits=logits, renyi_divergence = renyi_divergence) # for predictions



@dataclass
class SequenceClassifierOutputAndRenyi(SequenceClassifierOutput):
    """Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    task_loss: Optional[torch.FloatTensor] = None
    klg_loss: Optional[torch.FloatTensor] = None
    kld_loss: Optional[torch.FloatTensor] = None
    renyi_divergence: Optional[torch.FloatTensor] = None

