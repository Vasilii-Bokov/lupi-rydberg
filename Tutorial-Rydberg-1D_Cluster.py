#!/usr/bin/env python3
"""
Rydberg 1D Training Script for ALICE Cluster
Converted from Jupyter notebook
"""

import numpy as np
import pickle
from time import time
from joblib import Parallel, delayed
import sys
from src.models.mlp import MLP
from src.eval import RydbergEvaluator
from src.eval.eval_rydberg import (est_density_from_z_measurements, determine_phase_1D,
                                   est_order_param_1D, phase2img, est_phase_diagram,
                                   est_order_param_1D_fourier_from_measurements,
                                   est_order_param_1D_fourier,
                                   est_order_param_1D_from_measurements)
from src.data.loading.dataset_rydberg import RydbergDataset, unif_sample_on_grid

import argparse
from constants import *
from src.training.rydberg_trainers import RydbergConditionalTransformerTrainer
from src.models.transformer import init_conditional_transformer
from src.models.mlp import MLP

import torch
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange  # Changed from tqdm.notebook
import os
import random

from src.utils import plot_phase_diagram

warnings.filterwarnings('ignore')

# REMOVED: Jupyter magic commands
# REMOVED: get_ipython() calls

print("="*60)
print("Rydberg 1D Training Script")
print("="*60)

# ============ Configuration ============
# Base schedule
base_time = 3.5
ts = np.array([0, 0.2, base_time])
omegas = np.array([0, 5, 5])
deltas = np.array([-10, -10, 15])
total_time = 15
time_ratio = total_time/base_time

# System parameters
n_qubits = 31
dim = 1
ny = 1
nx = n_qubits
z2_threshold = 0.7
z3_threshold = 0.6

# Data folder
folder = f'data/1D-Phase_{nx}/1D-Phase_{nx}/{float(np.round(total_time,2))}µs/'
print(f"Data folder: {folder}")

# Check if data exists
if not os.path.exists(folder):
    print(f"WARNING: Data folder does not exist: {folder}")
    print("Please ensure data is uploaded to the cluster")
    sys.exit(1)

# ============ Load Dataset ============
print("\nLoading dataset...")
extra_variables = ["detuning"]
meta_dataset = RydbergDataset(dim=dim, nx=nx, ny=ny, folder=folder, n_threads=1,
                              var_name='interaction_range', variables=extra_variables)
meta_dataset.est_order_params()
meta_dataset.info["phase"] = determine_phase_1D(meta_dataset.info["Z2"], 
                                                meta_dataset.info["Z3"],
                                                z2_threshold=z2_threshold,
                                                z3_threshold=z3_threshold)
print("Dataset loaded successfully")

# ============ Prepare Training Data ============
sns.set_style('white')
hue_order = ['Disordered', 'Z2', 'Z3']
plot_df = meta_dataset.info.copy()
plot_df = plot_df.loc[(plot_df['detuning'] >= -1) & 
                      (plot_df['interaction_range'] <= 2.8) & 
                      (plot_df['interaction_range'] > 1)]

def prepare_train_set(meta_dataset, df=None, n_measurements: int = -1, x_bins=10, y_bins=10):
    train_set = {}
    if df is None: 
        df = meta_dataset.info
    train_idxes, train_df = unif_sample_on_grid(df.copy(), x_bins=x_bins, y_bins=y_bins)
    train_keys = meta_dataset.keys[train_idxes]
    train_set.update(meta_dataset.prepare_train_set(train_keys, n_measurements=n_measurements))
    return train_set, train_idxes

# Load or create training set
load_pretrained = False  # Set to True if loading pretrained model

if load_pretrained:
    print("\nLoading pretrained model data...")
    train_idxes = np.load('logs/rydberg_1D/train_idxes.npy')
    train_set = pickle.load(open('logs/rydberg_1D/train_set.pkl', 'rb'))
else:
    print("\nPreparing training set from scratch...")
    train_set, train_idxes = prepare_train_set(meta_dataset, df=plot_df)

# Sample N random training examples
N = 20
print(f"\nSampling {N} training examples...")
train_set_keys = list(train_set.keys())
selected_indices = random.sample(range(len(train_set_keys)), N)
selected_dict_items = {train_set_keys[i]: train_set[train_set_keys[i]] for i in selected_indices}
selected_array_items = [train_idxes[i] for i in selected_indices]
train_set = selected_dict_items
train_idxes = selected_array_items

# Save training set
output_dir = f'logs/rydberg_1D/{N}'
os.makedirs(output_dir, exist_ok=True)
np.save(f'{output_dir}/train_idxes.npy', train_idxes)
pickle.dump(train_set, open(f'{output_dir}/train_set.pkl', 'wb'))
print(f"Training set saved to {output_dir}")

# ============ Hyperparameters ============
def parse_args(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='logs/rydberg/debug/')
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--nx', type=int, default=19)
    parser.add_argument('--ny', type=int, default=1)
    parser.add_argument('--total_time', type=float, default=6)
    parser.add_argument('--tf-arch', type=str, default='transformer_l4_d128_h4')
    parser.add_argument('--train-id', type=str, default="debug")
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--ns', type=int, default=800)
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-samples', type=int, default=10000)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1])
    parser.add_argument('--epoch-mode', type=int, default=1, choices=[0, 1])
    parser.add_argument('--condition-mode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(args)

def get_hyperparams(**kwargs):
    hparams = argparse.Namespace(
        lr=1e-3,
        wd=0,
        bs=512,
        dropout=0.0,
        lr_scheduler=WARMUP_COSINE_SCHEDULER,
        warmup_frac=0.,
        final_lr=1e-7,
        smoothing=0.0,
        use_padding=0,
        val_frac=0.25,
        cattn=0
    )
    for k, v in kwargs.items():
        setattr(hparams, k, v)
    return hparams

args = parse_args()
hparams = get_hyperparams()

# ============ Device Selection ============
# FIXED: Use GPU 0 (SLURM manages GPU assignment)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============ Model Setup ============
print("\nSetting up model...")
num_outcomes = 2
n_vars = len(list(train_set.keys())[0])
rng = np.random.default_rng(seed=args.seed)

d_model = TF_ARCHS[args.tf_arch]['d_model']
n_head = TF_ARCHS[args.tf_arch]['n_head']
n_layers = TF_ARCHS[args.tf_arch]['n_layers']
assert d_model % n_head == 0, 'd_model must be integer multiple of n_head!'

encoder = MLP(input_size=n_vars, output_size=d_model,
              n_layers=1, hidden_size=128, activation='ELU',
              input_layer_norm=False,
              output_batch_size=None, device=device,
              output_factor=1.)

transformer = init_conditional_transformer(
    n_outcomes=num_outcomes,
    encoder=encoder,
    n_layers=n_layers,
    d_model=d_model,
    d_ff=4 * d_model,
    n_heads=n_head,
    dropout=hparams.dropout,
    version=hparams.use_padding,
    use_prompt=False,
)

# ============ Trainer Setup ============
trainer = RydbergConditionalTransformerTrainer(
    model=transformer,
    train_dataset=train_set,
    test_dataset=None,
    iterations=args.iterations,
    lr=hparams.lr,
    final_lr=hparams.final_lr,
    lr_scheduler=hparams.lr_scheduler,
    warmup_frac=hparams.warmup_frac,
    weight_decay=hparams.wd,
    batch_size=hparams.bs,
    rng=rng,
    smoothing=hparams.smoothing,
    eval_every=args.eval_every,
    transfomer_version=hparams.use_padding,
    device=device
)

model_name = f'transformer_nq-{n_qubits}_iter-{args.iterations//1000}k'
print(f'Model name: {model_name}')
print(f'Training iterations: {args.iterations}')

# ============ Training ============
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
trainer.train()

# ============ Save Model ============
print("\nSaving model...")
model_path = f'{output_dir}/{model_name}.pth'
torch.save(transformer, model_path)
print(f"✅ Model saved to: {model_path}")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)