# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Dina EL ZEIN <dina.el-zein@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import os
import logging
import torch 
import random
import numpy as np

glue_tasks = ['cola','sst2','mrpc','qqp','stsb', 'mnli','mnli-mm','qnli','rte','wnli','ax']

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)



def get_args():
    parser = argparse.ArgumentParser()
    # -----/ Seed \-----
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # -----/ Seed \-----

    # -----/ DataLoader and Trainer config \-----

    parser.add_argument(
        "--cache_dir",
        default="./cache",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",

    )

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model logging will be written.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=False,
        help="The file where the results are stored.",
    )

    


    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(glue_tasks),
    )
    # -----\ DataLoader and Trainer config /-----

    # -----/ DataLoader config \-----
    PERSISTENT_WORKERS: True

    parser.add_argument(
        "--train_batch_size", default=96, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",

    )

    parser.add_argument(
        "--pin_memory", default=True, type=bool,
    )
    parser.add_argument(
        "--num_workers", default=8, type=int,
    )
    parser.add_argument(
        "--prefetch_factor", default=2, type=int,
    )

    parser.add_argument(
        "--persistent_workers", default=True, type=bool,
    )

     # -----\ DataLoader config /-----

    # -----/ Trainer config \-----

    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout if we apply some.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--precision", default='16-mixed', type=str)

    parser.add_argument("--optimizer", default='AdamW', type=str)
    parser.add_argument("--schedular", default='get_linear_schedule_with_warmup', type=str)
   
    parser.add_argument("--warmup_ratio", default=0.06, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--deterministic", action="store_true", help="If specified, learns the reduced dimensions\
            through mlp in a deterministic manner.")

    parser.add_argument("--log_every_n_steps", default=1, type=int)
    parser.add_argument("--enable_checkpointing", default=True, type=bool)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--set_float32_matmul_precision", default="meduim", type=str)
  
    parser.add_argument("--max_epochs", default=15, type=int)
    parser.add_argument("--min_epochs", default=3, type=int)
    parser.add_argument("--gradient_clip_val", default=1, type=int)
    parser.add_argument("--num_sanity_val_steps", default=0, type=int)

    parser.add_argument("--load_from_checkpoint", default = False, type=bool, help="loads the model from the last checkpoint")

  # -----\ Trainer config /-----

  # -----\ gpu  /-----  
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
 # -----\ gpu /-----

 

  # -----/ VIB \-----

    parser.add_argument(
        "--VIB",
        action="store_true",
        help="if specified, adds the VIB layer on top of Bert",
    )

    parser.add_argument(
        "--VIB_use_scaling_factor",
        action="store_true",
        help="if specified, adds the a scaling factor for the NVIB layer",
    )

    parser.add_argument(
        "--VIB_scaling_factor_warmup_percentage",
        type = float, 
        default = 0,
    )

    parser.add_argument(
        "--VIB_lambda_klg",
        type = float, 
        default = 1e-4, 
        help="how much weight do we add on klg",
    )

    parser.add_argument(
        "--ib_dim",
        type = int, 
        default = 128, 
        help="Specifies the dimension of the information bottleneck",
    )

    parser.add_argument(
        "--activation",
        type = str, 
        default = 'relu', 
        choices=["tanh", "sigmoid", "relu"], 
    )




  # -----/ VIB \-----

  # -----/ NVIB \-----
    
    parser.add_argument(
        "--NVIB",
        action="store_true",
        help="if specified, adds the NVIB layer on top of Bert",
    )

    parser.add_argument(
        "--NVIB_use_scaling_factor",
        action="store_true",
        help="if specified, adds the a scaling factor for the NVIB layer",
    )

    parser.add_argument(
        "--NVIB_scaling_factor_warmup_percentage",
        type = float, 
        default = 0,
    )

    parser.add_argument(
        "--NVIB_lambda_klg",
        type = float, 
        default = 1e-4, 
        help="how much weight do we add on klg",
    )

    parser.add_argument(
        "--NVIB_lambda_kld",
        type = float, 
        default = 1e-4, 
        help="how much weight do we add on kld",
    )

    parser.add_argument(
        "--alpha_tau",
        type = float, 
        default = -10, 
        help="dirichlet initialisation, try range [0, 10].",
    )

    parser.add_argument(
        "--stdev_tau",
        type = float, 
        default = 0.1, 
        help="gaussian initialisation, try range [0, 1]",
    )

    parser.add_argument(
        "--mu_tau",
        type = float, 
        default = 1, 
        help="gaussian initialisation, try range [0, 1]",
    )

    parser.add_argument(
        "--delta",
        type = float, 
        default = 0, 
        help="Conditional prior delta for the dirichlet KL divergence",
    )


    parser.add_argument(
        "--learnable_prior",
        type = bool, 
        default = False, 
    )

    parser.add_argument(
        "--prior_mu",
        type = float, 
        default = None, 
    )

    parser.add_argument(
        "--prior_var",
        type = float, 
        default = None, 
    )

    parser.add_argument(
        "--prior_log_alphas",
        type = float, 
        default = None, 
    )

    parser.add_argument(
        "--prior_log_alpha_stdevs",
        type = float, 
        default = None, 
    )


    
    parser.add_argument(
        "--exp_type",
        type=str,
        default="baseline",
    )
    
    parser.add_argument(
        "--perform_latent_clipping",
        action="store_true",
        help="if specified, performs the latent clipping strategy",
    )
    parser.add_argument(
        "--max_latent_norm",
        type = float, 
        default = 8, 
    )
    parser.add_argument(
        "--max_alpha",
        type = float, 
        default = 1.5, 
    )
    
   
    # -----/ NVIB \-----


    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
  

    return args

