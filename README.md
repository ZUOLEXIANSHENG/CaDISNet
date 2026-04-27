# CaDISNet
Fine-Grained Finger Motor Imagery Decoding via CausalDisentanglement and Variational Information Bottleneck
# CaDISNet

CaDISNet is a causal disentanglement framework for cross-subject EEG motor imagery decoding.  
It is designed to improve subject-independent generalization by explicitly separating task-relevant semantic information from subject-specific nuisance variation in EEG representations.

This repository currently provides the core CaDISNet model and two training pipelines for fine-grained finger motor imagery decoding.

## Overview

Cross-subject EEG decoding remains challenging because neural signals vary substantially across subjects due to physiology, recording conditions, and subject-specific background activity. CaDISNet addresses this problem with a dual-stream causal representation learning framework that combines:

- semantic / variation disentanglement
- Hilbert-Schmidt Independence Criterion (HSIC) regularization
- variational information bottleneck (VIB)
- domain-adversarial learning
- contrastive regularization on variation features
- latent feature export for t-SNE visualization

The objective is to learn EEG representations that preserve motor-imagery semantics while reducing subject identity leakage.

## Main Features

- Cross-subject EEG motor imagery decoding
- Dual-stream latent representation learning
- Semantic branch for task classification
- Variation branch for subject-specific nuisance factors
- HSIC-based independence constraint
- Domain-adversarial purification
- Reconstruction objective for representation completeness
- Training-time export of latent features for t-SNE analysis
- TensorBoard logging support

## Repository Structure

```text
CaDISNet.py
CaDISNet_model_training 3class new.py
CaDISNet_model_training 2class new .py
plot_tsne_causal.py
Paper_Latex/
