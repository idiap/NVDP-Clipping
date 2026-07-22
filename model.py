# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

# ----------------------------------------------------------------------------------------------------
import torch
import transformers
import lightning as pl
import datasets
from datetime import datetime
import numpy as np 
from typing import Optional
import statistics
# ----------------------------------------------------------------------------------------------------
from transformers import (
    AutoConfig,
)
import evaluate


from DP_models import BertForSequenceClassificationWithExtraLayer
from DP_models import RobertaForSequenceClassificationWithExtraLayer
from utils import write_to_csv
from bayesian_privacy_accountant import BayesianPrivacyAccountant
# ----------------------------------------------------------------------------------------------------




class GLUETransformer(pl.LightningModule):

    def __init__(
        self,
        num_labels: int,
        eval_splits: Optional[list] = None,
        test_splits: Optional[list] = None,
        train_size: int = 0,
        test_size: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.__dict__.update(kwargs)
        
        self.args = kwargs
        
        self.num_labels = num_labels
        self.eval_splits = eval_splits
        self.test_splits = test_splits
        self.train_size = train_size
        self.test_size = test_size

        

        self.save_hyperparameters()


        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels, 
            hidden_dropout_prob = self.dropout,
            cache_dir=self.cache_dir,
            attention_probs_dropout_prob = self.dropout,
        )
        # for nvib 
        self.config.deterministic = self.deterministic
        self.config.NVIB = self.NVIB
        self.config.NVIB_lambda_klg = self.NVIB_lambda_klg 
        self.config.NVIB_lambda_kld = self.NVIB_lambda_kld
        self.config.alpha_tau = self.alpha_tau
        self.config.stdev_tau = self.stdev_tau
        self.config.mu_tau = self.mu_tau
        self.config.delta = self.delta
        self.config.learnable_prior = self.learnable_prior
        self.config.prior_mu = self.prior_mu
        self.config.prior_var = self.prior_var
        self.config.prior_log_alphas = self.prior_log_alphas
        self.config.prior_log_alpha_stdevs = self.prior_log_alpha_stdevs
        self.config.train_batch_size = self.train_batch_size
        self.config.eval_batch_size = self.eval_batch_size
        self.config.train_size = self.train_size
        self.config.test_size = self.test_size
        self.config.perform_latent_clipping = self.perform_latent_clipping
        self.config.max_latent_norm = self.max_latent_norm
        self.config.max_alpha = self.max_alpha

      

        # only for vib
        self.config.VIB = self.VIB
        self.config.ib_dim = self.ib_dim
        self.config.hidden_dim = (768 + self.ib_dim) // 2 
        self.config.VIB_lambda_klg = self.VIB_lambda_klg
        self.config.activation = self.activation
        

        self.accountant = BayesianPrivacyAccountant(powers=1.1, total_steps=1)
       

        MODEL_CLASS_MAP = {
            "bert": BertForSequenceClassificationWithExtraLayer,
            "roberta": RobertaForSequenceClassificationWithExtraLayer,
        }
        model_class = MODEL_CLASS_MAP.get(self.config.model_type)

        if model_class is None:
            raise ValueError(f"Model type '{self.config.model_type}' is not supported by this script.")

        self.model = model_class.from_pretrained(
            self.model_name_or_path, config=self.config, cache_dir=self.cache_dir, accountant = self.accountant,
            attn_implementation="eager"
        )

        if hasattr(self.model, 'bert'):
            base_model = self.model.bert
        elif hasattr(self.model, 'roberta'):
            base_model = self.model.roberta
        else:
            raise NotImplementedError("Could not find the base model attribute (e.g., 'bert', 'roberta').")
        
        base_model.gradient_checkpointing_enable()


        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.glue_the_metric = {
            'cola': 'matthews_correlation',
            'sst2': 'accuracy',
            'mrpc': 'f1',
            'qqp': 'f1',
            'stsb': 'spearmanr',
            'mnli': 'accuracy',
            'mnli_mismatched': 'accuracy',
            'mnli_matched': 'accuracy',
            'qnli': 'accuracy',
            'rte': 'accuracy',
            'wnli': 'accuracy',
        }
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_renyi_overall = []
        self.train_renyi_overall_mean = []
        self.predict_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def on_train_epoch_start(self):
        # put model in train mode
        self.model.train()
        return


    def training_step(self, batch):

        
        if self.NVIB and self.NVIB_use_scaling_factor:
            from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda
             
            scaling_factor = _get_linear_schedule_with_warmup_lr_lambda(
                current_step=self.trainer.global_step,
                num_warmup_steps=float(
                    self.NVIB_scaling_factor_warmup_percentage) * self.trainer.estimated_stepping_batches,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            
            
        else:
            scaling_factor = 1.0
        
        outputs = self(**batch, scaling_factor=scaling_factor)
        loss = outputs["loss"]
        task_loss = outputs["task_loss"]


        # log loss
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            f"train_task_loss",
            task_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        klg_loss = outputs.klg_loss
        kld_loss = outputs.kld_loss
        renyi_divergence = outputs.renyi_divergence 

        if self.NVIB:
            self.log(
                f"train_klg_loss",
                klg_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            self.log(
                f"train_kld_loss",
                kld_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss
    
   
    def on_train_epoch_end(self):
        
        self.train_step_outputs.clear() 
       
        

    def common_on_eval_epoch_start(self):
        # put model in eval mode
        self.model.eval()
        return

    def on_validation_epoch_start(self):
        # call common evaluation at the start of an epoch
        self.common_on_eval_epoch_start()
        return

    def on_test_epoch_start(self):
        # call common evaluation at the start of an epoch
        self.common_on_eval_epoch_start()
        return
    # -------------


    def common_eval_step(self, split, batch):
        outputs = self(**batch)
        val_loss = outputs["loss"]
        logits = outputs["logits"]

        val_task_loss = outputs["task_loss"]


        # log loss
        self.log(
            f"[{split}] loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            f"[{split}] task loss",
            val_task_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        klg_loss = outputs.klg_loss
        kld_loss = outputs.kld_loss
        renyi_divergence = outputs.renyi_divergence 

        if self.NVIB:
            self.log(
                f"[{split}] klg_loss",
                klg_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            self.log(
                f"[{split}] kld_loss",
                kld_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )


        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
            import torch.nn.functional as F
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()


        labels = batch["labels"]
        
        if split == "validation":
            self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels, "renyi_diveregnce": renyi_divergence, "klg_loss": klg_loss, "kld_loss": kld_loss, "task_loss": val_task_loss})
        elif split == "test":
            self.test_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels, "renyi_diveregnce": renyi_divergence, "klg_loss": klg_loss, "kld_loss": kld_loss, "task_loss": val_task_loss})
        
        return val_loss

    def validation_step(self, batch):
       return self.common_eval_step("validation", batch)

    def test_step(self, batch):
       return self.common_eval_step("test", batch)

    def common_on_eval_epoch_end(self, split, outputs):
        csv_results = {}
        if self.hparams.task_name == "mnli":
            accumulate_the_metric = 0
            accumulate_counts = 0
            for i, output in enumerate(outputs):
                # matched or mismatched
                if split == "validation": split_ = self.hparams.eval_splits[i].split("_")[-1] 
                elif split == "test" : split_ = self.hparams.test_splits[i].split("_")[-1] 
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"{split}_loss_{split_}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split_}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
                accumulate_the_metric += list(split_metrics.values())[0]
                accumulate_counts += 1
            self.log(f"{split}_the_metric", accumulate_the_metric / accumulate_counts)

             # check for NVIB and VIB
            if self.NVIB or self.VIB:
                renyi_divergence_max = torch.stack([x["renyi_diveregnce"][0] for x in outputs]).max()
                renyi_divergence_mean= torch.stack([x["renyi_diveregnce"][1] for x in outputs]).mean()
                task_loss = torch.stack([x["task_loss"] for x in outputs]).mean()
                self.log(f"{split}_task_loss", task_loss, prog_bar=True)

                kld_loss = torch.stack([x["kld_loss"] for x in outputs if x["kld_loss"] is not None]).mean() if any(x["kld_loss"] is not None for x in outputs) else None
                if kld_loss is not None : self.log(f"{split}_kld_loss", kld_loss, prog_bar=True) 
                
                klg_loss = torch.stack([x["klg_loss"] for x in outputs if x["klg_loss"] is not None]).mean() if any(x["klg_loss"] is not None for x in outputs) else None
                if klg_loss is not None : self.log(f"{split}_klg_loss", klg_loss, prog_bar=True) 

                    
            # write to csv
            csv_results['current_epoch'] = self.current_epoch
            temp = {k + '_' + split: v for k, v in split_metrics.items()}
            csv_results.update(temp)
            csv_results[f'{split}_loss'] = loss.item()

            if self.NVIB or self.VIB:
                csv_results[f'{split}_renyi_divergence'] = f"{round(renyi_divergence_max.item(), 3)};{round(renyi_divergence_mean.item(), 3)}" 
                csv_results[f'{split}_task_loss'] = task_loss.item()
                # adding the headers to the test portion during prediction
                csv_results['predict_renyi_divergence'] = 0
                csv_results['predict_bdp'] = 0

            if self.NVIB or self.VIB:
                csv_results[f'{split}_klg_loss'] = klg_loss.item()
            if self.NVIB:
                csv_results[f'{split}_kld_loss'] = kld_loss.item()
            
            write_to_csv(csv_results, self.args, self.output_file)

            return loss
    
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{split}_loss", loss, prog_bar=True)
        metrics_results = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics_results, prog_bar=True)
        the_metric_name = self.glue_the_metric[self.hparams.task_name]
        self.log(f"{split}_the_metric", metrics_results[the_metric_name])
       # check for NVIB and VIB
        if self.NVIB or self.VIB:
            renyi_divergence_max = torch.stack([x["renyi_diveregnce"][0] for x in outputs]).max()
            renyi_divergence_mean= torch.stack([x["renyi_diveregnce"][1] for x in outputs]).mean()
         
            #self.log(f"{split}_renyi_divergence", [renyi_divergence_max, renyi_divergence_mean], prog_bar=True)
            task_loss = torch.stack([x["task_loss"] for x in outputs]).mean()
            self.log(f"{split}_task_loss", task_loss, prog_bar=True)

            
            kld_loss = torch.stack([x["kld_loss"] for x in outputs if x["kld_loss"] is not None]).mean() if any(x["kld_loss"] is not None for x in outputs) else None
            if kld_loss is not None : self.log(f"{split}_kld_loss", kld_loss, prog_bar=True) 
            
            klg_loss = torch.stack([x["klg_loss"] for x in outputs if x["klg_loss"] is not None]).mean() if any(x["klg_loss"] is not None for x in outputs) else None
            if klg_loss is not None : self.log(f"{split}_klg_loss", klg_loss, prog_bar=True) 

       
        # write to csv 
        csv_results['current_epoch'] = self.current_epoch
        temp = {k + '_' + split: v for k, v in metrics_results.items()}
        csv_results.update(temp)
        csv_results[f'{split}_loss'] = loss.item()
        if self.NVIB or self.VIB:
            csv_results[f'{split}_renyi_divergence'] = f"{round(renyi_divergence_max.item(), 3)};{round(renyi_divergence_mean.item(), 3)}" 
            csv_results[f'{split}_task_loss'] = task_loss.item()
            #  adding the headers to the test portion during prediction
            csv_results['predict_renyi_divergence'] = 0
            csv_results['predict_bdp'] = 0

            if self.hparams.task_name == "mrpc":
                csv_results['test_renyi_divergence'] = 0
                csv_results['test_klg_loss'] = 0
                csv_results['test_kld_loss'] = 0

        if self.NVIB or self.VIB:
            csv_results[f'{split}_klg_loss'] = klg_loss.item()
        if self.NVIB:
            csv_results[f'{split}_kld_loss'] = kld_loss.item()

        
          
        write_to_csv(csv_results, self.args, self.output_file)
        return loss

    def on_validation_epoch_end(self):
        print("validation is over")
        loss_val = self.common_on_eval_epoch_end("validation", self.validation_step_outputs)
        self.validation_step_outputs.clear() 
        return loss_val

    def on_test_epoch_end(self):
        loss_test =  self.common_on_eval_epoch_end("test", self.test_step_outputs)
        self.test_step_outputs.clear() 
        return loss_test

    def on_predict_start(self):
        self.model.eval()
        return

    def predict_step(self, batch):
        idx = batch['idx'] 
        outputs = self(**batch, return_loss = False)
        logits = outputs["logits"]
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        
        if self.NVIB or self.VIB:
            renyi_divergence = outputs['renyi_divergence']
            self.predict_step_outputs.append({"preds": preds, "renyi_diveregnce": renyi_divergence})
        

        return idx, preds
    

    def on_predict_epoch_end(self):
        
        if self.NVIB or self.VIB:
            running_eps = self.accountant.get_privacy(target_delta=1e-5) if self.accountant else None
            csv_results = {}
            renyi_divergence_max = torch.stack([x["renyi_diveregnce"][0] for x in self.predict_step_outputs]).max()
            renyi_divergence_mean= torch.stack([x["renyi_diveregnce"][1] for x in self.predict_step_outputs]).mean()
            csv_results['predict_renyi_divergence'] = f"{round(renyi_divergence_max.item(), 3)};{round(renyi_divergence_mean.item(), 3)}" 
            csv_results['predict_bdp'] = f"{round(running_eps[0].item(), 3)}" 
            print(csv_results['predict_renyi_divergence'])
            print(csv_results['predict_bdp'])
            write_to_csv(csv_results, self.args, self.output_file)
            self.predict_step_outputs.clear() 
        return 



    def configure_optimizers(self):

       
        optimizer = getattr(torch.optim, self.optimizer)(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay = self.weight_decay,
        )

        if self.schedular:

           scheduler = getattr(transformers, self.schedular)(
                optimizer,
                num_warmup_steps=float(
                    self.hparams.warmup_ratio
                ) * self.trainer.estimated_stepping_batches,
                num_training_steps=self.trainer.estimated_stepping_batches
            )
           
           configuration = (
                [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
            )
           
        else:
            configuration = [optimizer]

        return configuration



            

