# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from data import GLUEDataModule
from model import GLUETransformer
import lightning as pl
from args import get_args 
import os 
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def main():
    args = get_args()
    pl.seed_everything(args.seed)
    print(args)

    processed_data = GLUEDataModule(

        **vars(args),
    )
    processed_data.setup()

    model = GLUETransformer(
        num_labels = processed_data.num_labels, 
        eval_splits = processed_data.eval_splits,
        test_splits = processed_data.test_splits,
        train_size = processed_data.dataset['train'].num_rows,
        test_size = processed_data.dataset['test'].num_rows,
        **vars(args)
    )
    
   
      # Ensure output_dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    
    # Create a ModelCheckpoint callback to save checkpoints in the checkpoint directory
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'best{args.exp_type}',
        monitor='validation_the_metric',
        mode='max', #'min',
        save_weights_only=False,
        save_last=False
    )


    early_stopping_callback = EarlyStopping(
        monitor='validation_the_metric',  # The metric to monitor
        mode='max',  # 'min' for minimizing the metric
        patience=3,  # Number of epochs to wait for improvement
        verbose=True,  # Log when early stopping occurs
    )

    trainer_args = {
        'precision': args.precision,
        'max_epochs':  args.max_epochs,
        'min_epochs': args.min_epochs, 
        'deterministic': args.deterministic, 
        'gradient_clip_val': args.gradient_clip_val, 
        'log_every_n_steps': args.log_every_n_steps, 
        'num_sanity_val_steps': args.num_sanity_val_steps,
        'enable_checkpointing': args.enable_checkpointing, 
        'check_val_every_n_epoch': args.check_val_every_n_epoch, 
        'default_root_dir': args.log_dir,
        'callbacks': [checkpoint_callback, early_stopping_callback],
    }

    trainer = pl.Trainer(**trainer_args)


    # Check if the output directory exists and is not empty
    checkpoint_path = None
    if args.load_from_checkpoint and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        # Find the latest checkpoint in the directory
        checkpoint_filename = f"best{args.exp_type}.ckpt"      
        checkpoint_path = os.path.join(args.output_dir, checkpoint_filename) 

    trainer.fit(model, datamodule=processed_data)
    trainer.validate(model, datamodule=processed_data)
    
    # make predictions
    predictions = trainer.predict(model, dataloaders=processed_data.test_dataloader(), ckpt_path= os.path.join(args.output_dir,f'best{args.exp_type}.ckpt'))
    
    # Initialize lists to accumulate idx and predictions
    all_idx = []
    all_predictions = []

    # Loop through the predictions for each batch
    for batch_predictions in predictions:
        batch_idx, batch_preds = batch_predictions  # Unpack idx and preds for each batch

        if args.task_name == 'rte' or args.task_name == 'qnli':
            batch_preds = ['not_entailment' if pred == 1 else 'entailment' for pred in batch_preds.tolist()]
        else:
            batch_preds = batch_preds.tolist()
        
        all_idx.extend(batch_idx.tolist())         
        all_predictions.extend(batch_preds) 

    # Save predictions to a TSV file
    predictions_file_path = os.path.join(args.output_dir, f'{args.task_name}{args.exp_type}.tsv')

    with open(predictions_file_path, "w") as f:
        f.write("index\tprediction\n")
        # Loop over the predictions and write each 'id [TAB] label' line
        for idx, prediction in zip(all_idx, all_predictions):
            f.write(f"{idx}\t{prediction}\n")

    print(f"Predictions saved to {predictions_file_path}")
    

if __name__ == "__main__":
    main()

