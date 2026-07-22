# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

# ----------------------------------------------------------------------------------------------------
import os
import lightning as pl
# ----------------------------------------------------------------------------------------------------
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
# ----------------------------------------------------------------------------------------------------


class GLUEDataModule(pl.LightningDataModule):

    task_text_field_map = {
        'cola': ['sentence'], # no label for test
        'sst2': ['sentence'], # no label for test
        'mrpc': ['sentence1', 'sentence2'],
        'qqp': ['question1', 'question2'], # no label for test
        'stsb': ['sentence1', 'sentence2'], # no label for test
        'mnli': ['premise', 'hypothesis'], # no label for test
        'mnli_mismatched': ['premise', 'hypothesis'], # no label for test
        'mnli_matched': ['premise', 'hypothesis'], # no label for test
        'qnli': ['question', 'sentence'],
        'rte': ['sentence1', 'sentence2'], # no label for test
        'wnli': ['sentence1', 'sentence2'], # no label for test
        'ax': ['premise', 'hypothesis'] # no label for test
    }

    glue_task_num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'qqp': 2,
        'stsb': 1,
        'mnli': 3,
        'mnli_mismatched': 3,
        'mnli_matched': 3,
        'qnli': 2,
        'rte': 2,
        'wnli': 2,
        'ax': 3
    }

    loader_columns = [
        'idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'start_positions',
        'end_positions',
        'labels',
    ]

    def __init__(
        self,
        **kwargs
        
    ):
        super().__init__()
        self.__dict__.update(kwargs)
        self.text_fields = self.task_text_field_map[self.task_name]
        self.num_labels = self.glue_task_num_labels[self.task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, cache_dir=self.cache_dir)

    def setup(self, stage= None):
        self.dataset = load_dataset('glue', self.task_name, cache_dir=self.cache_dir)
        

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                num_proc=self.num_workers,
                remove_columns=['label'],
                cache_file_name=os.path.join(
                    self.cache_dir,
                    f"{self.task_name}_{split}_preprocessing_map.hf"
                ),
                new_fingerprint=f"{self.task_name}_{split}_preprocessing_map"

            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]
        self.test_splits = [x for x in self.dataset.keys() if 'test' in x]
   
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'], 
            shuffle=True,
            drop_last=True,
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers

        )
    
    def common_eval_dataloader(self, split):
         # DataLoader wrap for the eval set of the dataset
        if split == "validation": e_splits = self.eval_splits 
        elif split == "test": e_splits = self.test_splits
        if len(e_splits) == 1:
            return DataLoader(
                self.dataset[split], 
                batch_size=self.eval_batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers
            )
         # this is mainly for mnli data, it has validation_mismatched and validation_matched
        elif len(e_splits) > 1:
            return [DataLoader(
                self.dataset[x], 
                batch_size=self.eval_batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers
            ) for x in self.e_splits]



    def val_dataloader(self):
        return self.common_eval_dataloader(
            split="validation"
        )


    def test_dataloader(self):
        return self.common_eval_dataloader(
            split="test"
        )
        
        
    

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['label']
        # Add the idx feature to be retained
        features['idx'] = example_batch['idx']  # Ensure idx is retained


        return features
