# MolCraft

A **deep learning** project for **de novo generation** and **optimization** of **molecules**.

## Status
In progress.

## Highlights
- Tokenization, model training, and SMILES generation, are all performed in the TF graph.

## Current Goals
1. Generation of novel chemical compounds based on SMILES, using generative transformer models. 
2. Finetuning of transformer models via deep reinforcement learning; to improve e.g. QED and/or SAS.

## Dependencies
- Python ~= 3.10
    - Keras ~= 3.0.0
    - TensorFlow ~= 2.15.0
    - TensorFlow Text ~= 2.15.0
    - RDKit ~= 2023.3.1

## Installation
For GPU users:
```
git clone git@github.com:akensert/molcraft.git
pip install -e .[gpu]
```
For CPU users:
```
git clone git@github.com:akensert/molcraft.git
pip install -e .
```
