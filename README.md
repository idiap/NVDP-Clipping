# Nonparametric Variational Differential Privacy via Embedding Parameter Clipping


__This repository contains the official implementation for the paper: **"Nonparametric Variational Differential Privacy via Embedding Parameter Clipping"**.__

This work is an enhancement of our previous paper, "Differential Privacy for Transformer Embeddings of Text with Nonparametric Variational Information Bottleneck."

> **Abstract:** The nonparametric variational information bottleneck (NVIB) provides the foundation for nonparametric variational differential privacy (NVDP), a framework for building privacy-preserving language models. However, the learned latent representations can drift into regions with high information content, leading to poor privacy guarantees, but also low utility due to numerical instability during training. In this work, we introduce a principled parameter clipping strategy to directly address this issue. Our method is mathematically derived from the objective of minimizing the Rényi Divergence (RD) upper bound, yielding specific, theoretically grounded constraints on the posterior mean, variance, and mixture weight parameters. We apply our technique to an NVIB based model and empirically compare it against an unconstrained baseline. Our findings demonstrate that the clipped model consistently achieves tighter RD bounds, implying stronger privacy, while simultaneously attaining higher performance on several downstream tasks. This work presents a simple yet effective method for improving the privacy-utility trade-off in variational models, making them more robust and practical.

## Overview 

This repository implements a principled parameter clipping strategy to improve the robustness and privacy guarantees of the **Nonparametric Variational Differential Privacy (NVDP)** framework.

While the original NVDP method provides a strong privacy-utility trade-off, we identified that its worst-case privacy guarantees can be loose if posterior parameters ($\mu$, $\sigma$, $\alpha$) drift during training. This work introduces a solution: a clipping mechanism applied directly to these parameters within the NVIB layer.

The clipping bounds are not arbitrary; they are **mathematically derived** from the objective of minimizing the Rényi Divergence. This results in an **NVDP-Clipped** model that is more robust, achieving both tighter privacy bounds and, in many cases, better downstream performance.



## Setup and Installation

To get started, clone the repository and set up the Python environment. We recommend using a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/idiap/NVDP-Clipping
cd nvdp-clipping

# 2. Create and activate a virtual environment 
micromamba create -f env.yml
micromamba activate nvdp-clipping
```

## Running Experiments 

The scripts in the [sample_commands](#sample_commands) directory provide the easiest way to reproduce our results. The key difference between this codebase and our original NVDP implementation ([link to original NVDP repository]) is the addition of flags to enable our proposed parameter clipping.



For example, to run the NVDP-Clipped experiment on the MRPC dataset, simply execute the corresponding script:

```bash
bash sample_commands/mrpc_nvib.sh
```

You can easily run experiments for other tasks or with different backbones (e.g., BERT-Large, RoBERTa) by executing the other scripts in the directory. To customize an experiment, you can modify the model name, learning rate, or clipping hyperparameters directly within the shell script.


## Citation

To cite this work in your publications:

```bash
@inproceedings{el2026nonparametric,
  title={Nonparametric Variational Differential Privacy via Embedding Parameter Clipping},
  author={El Zein, Dina and Kumar, Shashi and Henderson, James},
  booktitle={ICLR 2026 Workshop on Principled Design for Trustworthy AI-Interpretability, Robustness, and Safety across Modalities}
}
```