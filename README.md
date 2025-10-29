# Generative Quantum States

A transformer-based generative model for quantum many-body systems, with applications to Rydberg atom arrays and Heisenberg models.

**Note:** The transformer model implementation is based on the work from [PennyLaneAI/generative-quantum-states](https://github.com/PennyLaneAI/generative-quantum-states).

## Overview

This project implements conditional transformer models to learn and generate quantum states from experimental or simulated measurement data. The models can predict quantum phase diagrams and estimate order parameters with high accuracy.

This repository contains the code for the research article on **Learning Under Privileged Information (LUPI) with Quantum Features**. The project compares two approaches to quantum phase detection:
1. **Transformer-based generative models** for learning quantum state distributions
2. **SVM+ (Support Vector Machines Plus)** - a LUPI algorithm that leverages quantum features as privileged information during training

The key contribution is demonstrating how quantum-derived features can enhance classical machine learning models through the LUPI framework, establishing a practical pathway to quantum advantage in hybrid classical-quantum learning scenarios.

## Features

### LUPI (Quantum-Enhanced Classical Learning)
- **Complete SVM+ Implementation**: Binary and multiclass LUPI algorithms from scratch
- **Interactive Notebooks**: Step-by-step tutorials for data preparation, training, and evaluation
- **Quantum Privileged Features**: Uses quantum order parameters available only during training
- **Benchmark Demonstrations**: Synthetic datasets showing LUPI advantages
- **Quantitative Results**: ~10% accuracy improvement over standard SVM on quantum phase detection

### Transformer Models
- **Conditional Transformer Architecture**: Generates quantum measurements conditioned on system parameters
- **Multiple Quantum Systems**: Support for Rydberg 1D arrays and 2D Heisenberg models
- **Phase Diagram Learning**: Automatically learns phase boundaries and order parameters

### Computational Infrastructure
- **Cluster Computing**: Optimized for HPC environments with GPU support
- **Boundary Sampling**: Advanced sampling strategies for phase transition regions
- **Automated Workflows**: Scripts for batch training and evaluation
- **Comparative Analysis**: Tools for benchmarking LUPI vs Transformer approaches

## Research Article

This repository contains the implementation code for the research article:

**"Learning Under Privileged Information with Quantum Features"**

### Organization

The project is organized into two main components:

1. **`rydberg_lupi/` folder**: Contains all LUPI (SVM+) experiments
   - Jupyter notebooks for interactive exploration
   - Python modules for SVM+ algorithms
   - Experimental results (`.pkl` files) from various runs
   - All quantum-enhanced classical learning code in one place

2. **Root directory**: Transformer-based generative models and cluster computing scripts
   - Transformer training and evaluation
   - Automated workflows for HPC clusters
   - Comparative analysis tools

### Abstract Summary

Quantum machine learning (QML) has been proposed as a promising application of quantum computers. However, practical deployment faces significant hardware limitations. This work explores an alternative approach: using quantum devices as feature extractors to enhance classical learning algorithms rather than training directly on quantum computers.

**Key Contributions:**
- **LUPI-Quantum Connection**: Establishes the conceptual link between Learning Under Privileged Information (LUPI) and quantum computing, where quantum-derived features serve as privileged information available only during training
- **Advantageous Feature Extraction**: Introduces the notion of advantageous quantum feature extraction with provable advantages for synthetic learning problems
- **Practical Application**: Demonstrates the framework on quantum phase detection in Rydberg atom systems
- **Performance Improvements**: Shows notable improvements over models trained without quantum features

**Results:** Quantum-enhanced classical models (SVM+) successfully leverage quantum-derived features to achieve superior performance compared to standard classical approaches. The transformer models provide baseline comparisons and demonstrate the effectiveness of generative modeling for quantum state learning.

**Conclusion:** Leveraging quantum features to enhance classical learning through the LUPI framework constitutes a viable route to achieve practical quantum advantages through hybrid classical-quantum approaches.

## Project Structure

```
generative-quantum-states-main/
├── rydberg_lupi/                        # LUPI Experiments (Quantum-Enhanced Classical Learning)
│   ├── Notebooks
│   │   ├── Rydberg_data.ipynb          # Data preparation for LUPI experiments
│   │   ├── Rydberg_data (Copy).ipynb   # Backup/alternative version
│   │   ├── SVMplus_Rydberg.ipynb       # SVM+ implementation & experiments
│   │   ├── SVM+_Rydberg.ipynb          # Alternative SVM+ experiments
│   │   ├── SVM__Rydberg.ipynb          # Standard SVM baseline comparison
│   │   ├── SVMs.ipynb                  # Additional SVM comparisons
│   │   └── Untitled-1.ipynb            # Scratch/experimental notebook
│   │
│   ├── Python Modules
│   │   ├── SVMplus.py                  # Standalone SVM+ implementation
│   │   ├── SVMdeltatplus.py            # Delta variant of SVM+
│   │   └── constants.py                # Configuration constants
│   │
│   └── Results (Experimental Data)
│       ├── exp_res_1.pkl               # Experiment results set 1
│       ├── exp_res_2.pkl               # Experiment results set 2
│       ├── exp_res_3.pkl               # Experiment results set 3
│       ├── exp_res_bd.pkl              # Boundary sampling results
│       ├── exp_res_bd_1.pkl            # Boundary results set 1
│       ├── exp_res_bd_2.pkl            # Boundary results set 2
│       ├── exp_res_bd_3.pkl            # Boundary results set 3
│       ├── exp_res_bdn_big_1.pkl       # Large-scale boundary results
│       ├── exp_res_bdn_big_2.pkl
│       ├── exp_res_bdn_big_3.pkl
│       ├── exp_res_big_1.pkl           # Large-scale uniform results
│       ├── exp_res_big_2.pkl
│       └── exp_res_big_3.pkl
│
├── src/                                 # Core source code
│   ├── models/                          # Model architectures (MLP, Transformer)
│   ├── training/                        # Training loops and utilities
│   ├── data/                            # Dataset classes
│   └── eval/                            # Evaluation metrics
│
├── Transformer Notebooks & Scripts
│   ├── notebooks/
│   │   ├── Tutorial-Rydberg-1D.ipynb   # Interactive transformer tutorial
│   │   └── Tutorial-2D-Heisenberg.ipynb
│   ├── Tutorial-Rydberg-1D_Cluster.py  # Batch training script
│   └── Tutorial-Rydberg-1D_Cluster_Multi.py
│
├── Data Generation & Evaluation (Transformers)
│   ├── generate_multiple_datasets.py    # Uniform sampling
│   ├── generate_multiple_datasets_bd.py # Boundary sampling
│   ├── Compute_Accuracy_Cluster.py      # Accuracy evaluation
│   ├── Compute_Accuracy_Cluster_BD.py
│   ├── collect_accuracy_results.py      # Results aggregation
│   └── collect_accuracy_results_bd.py
│
├── Heisenberg Model (LUPI & Transformers)
│   ├── heisenberg_kernel_comparison.py  # Kernel methods comparison
│   ├── heisenberg_train_transformer.py
│   ├── heisenberg_sample_transformer.py
│   └── heisenberg_evaluate_properties.py
│
├── Cluster Job Scripts
│   ├── submit_all_datasets.sh           # Submit training jobs
│   ├── submit_all_datasets_bd.sh
│   ├── submit_accuracy_jobs.sh          # Submit evaluation jobs
│   ├── submit_accuracy_jobs_bd.sh
│   └── job_scripts/                     # Individual SLURM scripts
│
├── rydberg/                             # Rydberg-specific modules
├── data/                                # Training data storage
├── logs/                                # Model checkpoints and outputs
└── requirements.txt                     # Python dependencies
```

### Key Files by Function

#### LUPI (SVM+) Experiments - `rydberg_lupi/` folder
All LUPI-related code and results are organized in the `rydberg_lupi/` directory:

**Notebooks:**
- **Rydberg_data.ipynb**: Data loading and preprocessing for quantum phase detection
  - Loads Rydberg atom quantum measurement data
  - Extracts regular features (system parameters) and privileged features (order parameters)
  - Prepares train/test splits for LUPI experiments
  
- **SVMplus_Rydberg.ipynb**: Complete SVM+ algorithm with demonstrations
  - `SVMPlus` class: Binary LUPI classifier with dual optimization
  - `MulticlassSVMPlus` class: Multiclass extension (One-vs-One strategy)
  - Synthetic benchmarks: Noisy moons dataset showing ~10% improvement
  - Quantum phase classification: Disordered/Z2/Z3 phases
  
- **SVM__Rydberg.ipynb**: Standard SVM baseline for comparison
  - Establishes baseline performance without privileged information
  - Fair comparison with same data splits and parameters
  
- **SVM+_Rydberg.ipynb**, **SVMs.ipynb**: Additional SVM experiments and variations

**Python Modules:**
- **SVMplus.py**: Standalone SVM+ implementation (can be imported)
- **SVMdeltatplus.py**: Delta variant of SVM+ algorithm
- **constants.py**: Shared configuration parameters

**Results:** 
Experimental results stored as pickle files with various configurations:
- `exp_res_*.pkl`: Standard uniform sampling results (3 runs)
- `exp_res_bd*.pkl`: Boundary sampling results (3 runs)
- `exp_res_big*.pkl`: Large-scale experiments (3 runs each for uniform and boundary)

#### Transformer Experiments
- **Tutorial-Rydberg-1D_*.py**: Scripts for training transformer models on Rydberg systems
- **Compute_Accuracy_Cluster*.py**: Accuracy evaluation scripts for cluster computing
- **generate_multiple_datasets*.py**: Dataset generation with uniform and boundary sampling

#### Analysis & Utilities
- **heisenberg_*.py**: Scripts for 2D Heisenberg model experiments
- **collect_accuracy_results*.py**: Aggregate results from multiple runs

## Requirements

### Python Dependencies

```bash
numpy
torch
pennylane
pennylane-lightning
tqdm
scipy
joblib
matplotlib
seaborn
scikit-learn
pandas
tensorboard
qutip
jax
neural-tangents
torch-geometric
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

### For Cluster Usage (ALICE/SLURM)

Load required modules:
```bash
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

## Quick Start

### 1. Running Jupyter Tutorials Locally

#### LUPI (SVM+) Tutorials

Start with the LUPI notebooks in the `rydberg_lupi/` folder to understand quantum-enhanced classical learning:

**Step 1: Data Preparation**
```bash
jupyter notebook rydberg_lupi/Rydberg_data.ipynb
```
- Loads Rydberg atom quantum measurement data
- Extracts regular and privileged features
- Prepares datasets for LUPI experiments

**Step 2: SVM+ Implementation and Experiments**
```bash
jupyter notebook rydberg_lupi/SVMplus_Rydberg.ipynb
```
- Complete SVM+ algorithm implementation
- Synthetic benchmark (noisy moons) demonstration
- Quantum phase classification experiments
- Multiclass LUPI for 3-phase detection

**Step 3: Baseline Comparison**
```bash
jupyter notebook rydberg_lupi/SVM__Rydberg.ipynb
```
- Standard SVM baseline results
- Direct comparison with SVM+ performance
- Demonstrates quantum advantage quantitatively

**Additional Notebooks:**
```bash
# Alternative SVM+ experiments
jupyter notebook rydberg_lupi/SVM+_Rydberg.ipynb

# Multiple SVM comparisons
jupyter notebook rydberg_lupi/SVMs.ipynb
```

#### Transformer Tutorials

Alternatively, explore the generative modeling approach:

```bash
jupyter notebook notebooks/Tutorial-Rydberg-1D.ipynb
```

or

```bash
jupyter notebook notebooks/Tutorial-2D-Heisenberg.ipynb
```

### 2. Training on a Cluster

#### Uniform Sampling Workflow

To train models with uniform sampling across the phase diagram:

**Step 1: Generate Multiple Datasets**
```bash
# Edit parameters in the file first (set N and M values)
python3 generate_multiple_datasets.py
```

**Step 2: Submit Training Jobs**
```bash
# Trains transformers on all generated datasets
./submit_all_datasets.sh
# Wait for training to complete
```

**Step 3: Compute Model Accuracies**
```bash
# Evaluates trained models on test data
./submit_accuracy_jobs.sh
```

**Step 4: Monitor Progress**
```bash
squeue -u your_username
# Wait for all accuracy jobs to complete (~30-60 min each)
```

**Step 5: Collect Results**
```bash
# Aggregates all results into a single CSV file
python3 collect_accuracy_results.py
```

Results will be saved to: `accuracy_results_N{N}_M{M}_all_datasets.csv`

#### Boundary Sampling Workflow

To train models with sampling concentrated near phase boundaries:

**Step 1: Generate Boundary Datasets**
```bash
# Edit parameters in the file first (set N and M values)
python3 generate_multiple_datasets_bd.py
```

**Step 2: Submit Training Jobs**
```bash
# Trains transformers on boundary-sampled datasets
./submit_all_datasets_bd.sh
# Wait for training to complete
```

**Step 3: Compute Model Accuracies**
```bash
# Evaluates trained models on test data
./submit_accuracy_jobs_bd.sh
```

**Step 4: Monitor Progress**
```bash
squeue -u your_username
# Wait for all accuracy jobs to complete (~30-60 min each)
```

**Step 5: Collect Results**
```bash
# Aggregates all results into a single CSV file
python3 collect_accuracy_results_bd.py
```

Results will be saved to: `accuracy_results_N{N}_M{M}_boundary_all_datasets.csv`

### 3. Single Model Training

To train a single model interactively:

```bash
python3 Tutorial-Rydberg-1D_Cluster.py
```

Or submit as a job:

```bash
sbatch run_rydberg_training.sh
```

## Configuration

### Key Parameters

Before running the workflow, configure these parameters in the respective scripts:

- **N**: Number of training samples (e.g., 20, 30, 40, 50)
- **M**: Size of each dataset (e.g., 1000, 5000)
- **Iterations**: Training iterations (default: 50,000)
- **Batch Size**: Training batch size (default: 512)
- **Learning Rate**: Initial learning rate (default: 1e-3)

Edit parameters in:
- `generate_multiple_datasets.py` or `generate_multiple_datasets_bd.py`
- `Tutorial-Rydberg-1D_Cluster.py`
- Job submission scripts in `job_scripts/`

### Boundary Sampling Parameters

For boundary sampling (`generate_multiple_datasets_bd.py`):

- **Z2_THRESHOLD**: Critical Z2 order parameter value (default: 0.7)
- **Z3_THRESHOLD**: Critical Z3 order parameter value (default: 0.6)
- **TOLERANCE**: Width of boundary region (default: 0.12)
- **BOUNDARY_WEIGHT**: Sampling weight for boundary points (default: 6.0)

## Output Files

### Training Outputs

- `logs/rydberg_1D/{N}/transformer_nq-{nq}_N-{N}_iter-{iter}k.pth`: Trained model checkpoint
- `logs/rydberg_1D/{N}/train_set.pkl`: Training dataset
- `logs/rydberg_1D/{N}/train_idxes.npy`: Training sample indices

### Accuracy Results

- `accuracy_results_N{N}_M{M}_all_datasets.csv`: Uniform sampling results
- `accuracy_results_N{N}_M{M}_boundary_all_datasets.csv`: Boundary sampling results
- `test_results.pkl`: Detailed test results with predictions

### Available Result Files

The repository includes pre-computed results for various system sizes:

- N=20, M=1000 (uniform and boundary)
- N=30, M=1000 and M=5000 (uniform and boundary)
- N=40, M=1000 (uniform and boundary)
- N=50, M=5000 (boundary only)
- N=60, M=5000 (boundary only)

## Cluster-Specific Usage (ALICE)

### Setting Up Environment

```bash
# Load required modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Create virtual environment with system packages
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install additional packages
pip install -r requirements.txt
```

### Submitting Jobs

The project includes pre-generated job scripts for batch processing:

- `job_dataset_*.sh`: Dataset generation jobs
- `job_accuracy_*.sh`: Accuracy computation jobs
- `submit_all_datasets.sh`: Submit all training jobs
- `submit_accuracy_jobs.sh`: Submit all evaluation jobs

### Monitoring Jobs

```bash
# Check job status
squeue -u your_username

# View job output
cat logs/rydberg_train_*.out

# View errors
cat logs/rydberg_train_*.err
```

## Model Architecture

### Conditional Transformer

- **Encoder**: MLP that processes system parameters
- **Transformer**: Multi-head self-attention with positional encoding
- **Decoder**: Generates quantum measurement outcomes
- **Conditioning**: System parameters (detuning, interaction range, etc.)

### Hyperparameters

- Model dimension: 256
- Number of layers: 6
- Attention heads: 8
- Feed-forward dimension: 4 × model dimension
- Dropout: 0.1

## Evaluation Metrics

- **Order Parameters**: Z2, Z3 correlations for phase classification
- **Phase Accuracy**: Correct phase prediction rate
- **RMSE**: Root mean square error for order parameter estimation
- **Phase Diagram Reconstruction**: Visual comparison with ground truth

## Methods Comparison

This project implements and compares two distinct approaches to quantum phase detection:

### 1. SVM+ (Learning Under Privileged Information)

The LUPI approach is implemented across three main notebooks that demonstrate the framework from data preparation to final results:

#### **Rydberg_data.ipynb** - Data Preparation
This notebook handles the loading and preprocessing of Rydberg atom system data for LUPI experiments.

**Key Features:**
- Loads quantum measurement data from Rydberg 1D arrays
- Extracts system parameters (detuning, interaction range)
- Computes order parameters (Z2, Z3) for phase classification
- Prepares datasets with both regular features and privileged quantum information
- Determines phase labels (Disordered, Z2, Z3) using threshold-based classification

**Output:** Preprocessed datasets ready for SVM+ training with:
- **Regular features**: Classical observables available at test time
- **Privileged features**: Quantum-derived information (order parameters) available only during training

#### **SVMplus_Rydberg.ipynb** - LUPI Algorithm Implementation
This notebook contains the complete implementation of the SVM+ algorithm and demonstrates its application to quantum phase detection.

**Core Implementation:**
- **`SVMPlus` class**: Binary classification with privileged information
  - Dual optimization formulation following Vapnik's framework
  - Custom kernel functions for both regular and privileged feature spaces
  - Supports RBF, polynomial, and linear kernels
  - Optimized using constrained optimization (SLSQP method)
  
- **`MulticlassSVMPlus` class**: Extension to multiclass problems
  - One-vs-One strategy for handling multiple quantum phases
  - Voting mechanism for final predictions

**LUPI Framework:**
```
Training Phase:  Uses both X (regular features) and X* (privileged quantum features)
Test Phase:      Uses only X (regular features)
Advantage:       Quantum information guides training but isn't required at deployment
```

**Experiments:**
1. **Synthetic Benchmark**: Noisy moons dataset demonstrating LUPI advantages
   - Privileged information: Noise characteristics (levels, directions)
   - Results: ~10% accuracy improvement over standard SVM
   
2. **Quantum Phase Detection**: Rydberg atom system classification
   - Regular features: System parameters (detuning, interaction range)
   - Privileged features: Quantum order parameters (Z2, Z3 correlations)
   - Multiclass classification: Disordered, Z2 ordered, Z3 ordered phases

**Key Results:**
- SVM+ consistently outperforms standard SVM on both training and test sets
- Fewer misclassification errors (3-5 fewer errors per 100 samples)
- Demonstrates practical quantum advantage through hybrid classical-quantum learning

#### **SVM__Rydberg.ipynb** - Baseline Comparison
This notebook provides standard SVM baseline results for comparison with SVM+.

**Purpose:**
- Establishes baseline performance using only classical features
- Fair comparison: Same data, same train/test splits, same kernel parameters
- Demonstrates the added value of privileged quantum information

**Results Summary:**
- **Training Accuracy**: SVM standard ~94%, SVM+ ~97%
- **Test Accuracy**: SVM standard ~77%, SVM+ ~87%
- **Improvement**: +10% test accuracy with privileged quantum features

**Key Insight:** The performance gap demonstrates that quantum-derived features, when used as privileged information through the LUPI framework, provide measurable advantages even when not available at test time.

### 2. Transformer-Based Generative Models

**Approach:** Learn the full probability distribution of quantum measurements conditioned on system parameters.

### 2. Transformer-Based Generative Models

**Approach:** Learn the full probability distribution of quantum measurements conditioned on system parameters.

**Implementation:** The transformer model is based on the architecture from [PennyLaneAI/generative-quantum-states](https://github.com/PennyLaneAI/generative-quantum-states), adapted for the Rydberg atom system.

**Architecture Components:**
- **Encoder**: MLP that processes system parameters (detuning, interaction range)
- **Conditional Transformer**: Multi-head self-attention mechanism
  - 6 layers, 8 attention heads, 256-dimensional embeddings
  - Learns correlations in quantum measurement sequences
  - Conditions generation on system parameters
- **Decoder**: Outputs probability distributions over measurement outcomes

**Training Process:**
- **Input**: System parameters (detuning, interaction range, etc.)
- **Target**: Sequences of quantum measurements from Rydberg arrays
- **Loss**: Cross-entropy between predicted and observed measurement distributions
- **Training**: 50,000 iterations with Adam optimizer, learning rate scheduling

**Capabilities:**
- Generates synthetic quantum measurements that match true distributions
- Reconstructs full phase diagrams from limited training data
- Estimates order parameters (Z2, Z3) from generated samples
- Identifies phase boundaries without explicit phase labels

**Notebooks:**
- `Tutorial-Rydberg-1D.ipynb`: Interactive tutorial for training and evaluating transformers
- `Tutorial-2D-Heisenberg.ipynb`: Extension to 2D Heisenberg models

**Cluster Training Scripts:**
- `Tutorial-Rydberg-1D_Cluster.py`: Batch training for multiple dataset sizes
- Automated workflows for generating datasets, training, and evaluation

**Advantages:**
- Captures complete quantum state information
- Can generate synthetic quantum measurements
- End-to-end differentiable training
- Scales well with system size

**Use Cases:**
- Phase diagram reconstruction
- Quantum state tomography
- Generative sampling for further analysis

### SVM+ vs Transformer: Complementary Approaches

The two methods serve different but complementary purposes in quantum learning:

| Aspect | SVM+ (LUPI) | Transformer |
|--------|-------------|-------------|
| **Goal** | Direct phase classification | Full distribution learning |
| **Training** | Uses quantum features as privileged info | Learns from measurement sequences |
| **Testing** | Requires only classical features | Generates quantum-like samples |
| **Quantum Access** | Training time only | Can be used to reduce quantum queries |
| **Sample Efficiency** | More efficient for specific tasks | Requires more training data |
| **Output** | Phase labels, decision boundaries | Probability distributions, samples |
| **Interpretability** | Support vectors, decision function | Learned correlations in attention |

**Research Contribution:** This project demonstrates that both approaches successfully leverage quantum information, with SVM+ achieving practical quantum advantages through the LUPI framework, while transformers provide a generative baseline for understanding the complete quantum state space.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{quantum_lupi_2025,
  title={Learning Under Privileged Information with Quantum Features},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  note={Code available at: https://github.com/[your-repo]}
}
```

### Related Work

The transformer model implementation is based on:
- **PennyLane Generative Quantum States**: https://github.com/PennyLaneAI/generative-quantum-states

## License

See LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the tutorial notebooks for examples
- Review the cluster documentation for HPC-specific questions

## Notes

- Always use `sbatch` for compute-intensive work on HPC clusters
- Login nodes are for light tasks only (editing files, submitting jobs)
- Training a single model typically takes 30-60 minutes on GPU
- Generating datasets can take several hours depending on size
- Monitor disk space usage in your cluster home directory
