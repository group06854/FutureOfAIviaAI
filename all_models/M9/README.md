# VGAE for Link Prediction in FutureOfAIviaAI

This project implements a **Variational Graph Autoencoder (VGAE)** for the link prediction task and integrates it into the [FutureOfAIviaAI](https://github.com/artificial-scientist-lab/FutureOfAIviaAI) repository. The model represents a fundamentally novel approach based on graph neural networks and variational autoencoders.

## Quick Start

Follow these steps to set up the environment and run the model evaluation.

### 1. Create and Activate Environment
Using `conda` for dependency management is recommended.
```bash
# Create a new isolated environment
conda create -n vgae_env python=3.9 -y

# Activate the environment
conda activate vgae_env
```

### 2. Install Dependencies
Install all required libraries, including PyTorch and PyTorch Geometric.
```bash
# Install PyTorch (choose the option compatible with your system)
# For CPU:
pip install torch torchvision torchaudio
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other key dependencies
pip install torch-geometric networkx numpy scipy scikit-learn
```

### 3. Prepare Data
Place the dataset file in the project's root directory.
```bash
# Ensure the data file is located here
# e.g., ./SemanticGraph_delta_1_cutoff_0_minedge_1.pkl
```

### 4. Run Evaluation
Execute the main script. It will automatically load the data, train the VGAE model, and output the AUC-ROC metric.
```bash
python simple_model.py
```
Upon completion, you will see the result:
```
Area Under Curve for Evaluation: 0.XX
```

## Project Structure
- **create_data.py**: A simple python file for creating the datasets SemanticGraph_delta_N_cutoff_M_minedge_P.pkl from the full semantic network all_edges.pkl.
- **evaluate_model.py**: Runs my simple baseline model on all datasets
- **simple_model.py**: The main file containing the VGAE model implementation, training, and evaluation interface.
- **utils.py**: Contains useful functions, such as the creation of datasets from the full semantic network (unbiased for test-set, and biased [i.e. same number of positive and negative solutions] for training if desired), and AUC computation.

## Implementation Features
- **Inductive Learning**: The model can handle new graph nodes not seen during training.
- **Compatibility**: The implementation is fully compatible with the original repository's evaluation pipeline.
- **Self-Supervised Approach**: Training is performed without explicit labels by reconstructing the graph structure.

## Technical Details
- **Framework**: PyTorch + PyTorch Geometric
- **Primary Metric**: AUC-ROC (Area Under the ROC Curve)
- **Architecture**: GCN-based Encoder + Inner Product Decoder
- **Loss Function**: Reconstruction Loss + β * KL-Divergence

## Report
This repository offers a Variational Graph Autoencoder (VGAE) as a solution. Unlike models based on hand-crafted features (M5, M6) or fixed embeddings (M2, M7), VGAE performs end-to-end learning by automatically extracting structural representations of graph vertices. The key feature of the architecture is the combination of graph convolutional networks (GCN) as an encoder and variational inference, which allows the model not only to restore the observed connections, but also to train the regularized latency space through minimizing KL divergence. This approach is inductive, which is critically important for dynamically growing scientific graphs.

The method was implemented while maintaining full compatibility with the existing repository infrastructure. The main function vgae_link_prediction() was developed in accordance with the signature of analogues from the M6 and M7 models, which allowed it to be integrated into the standard evaluation pipeline. evaluate_model.py . During the development process, key technical issues were successfully resolved, including the correct temporary separation of data to avoid leakage (data leak), adaptation to the boundaries of the Kaggle runtime environment (working with a read-only file system by redirecting entries to the /kaggle/working/ directory).

The effectiveness of the model was evaluated on the SemanticGraph dataset using a standard metric for the repository — the area under the ROC curve (AUC-ROC). The resulting VGAE result (
for years_delta: 3, min_edges: 1, vertex_degree_cutoff: 0, AUC: 0.76
 years_delta: 1, min_edges: 1, vertex_degree_cutoff: 0, AUC: 0.82) is competitive against existing models, especially considering that this approach does not require time-consuming feature engineering and demonstrates advantages in specific scenarios. Such scenarios include working with sparse data (few positive examples), the need for rapid prototyping without the feature creation stage, and, most importantly, prediction in a dynamically changing graph with the appearance of new vertices, where the inductive nature of VGAE gives it an advantage over transductive methods such as Node2Vec (M7). Thus, the implemented method not only filled a methodological gap in the repository by introducing the first graph autoencoder, but also confirmed its practical value by offering a fully automated, scalable and theoretically sound approach to the task of predicting scientific collaborations.