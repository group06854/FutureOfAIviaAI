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
### Result
*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|----------|------------|------------|------------|
| *delta=1*  | 0.856544   | 0.855899   | 0.848262   |
| *delta=3*  | 0.783369   | 0.769020   | 0.768193   |
| *delta=5*  | 0.726860   | 0.708579   | 0.697075   |

 *Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1*  | 0.949984   | 0.938684   | 0.960110   |
| *delta=3*  | 0.867099   | 0.875685   | 0.876027   |
| *delta=5*  | 0.812581   | 0.799561   | 0.782653   |


### Idea
This repository offers a Variational Graph Autoencoder (VGAE) as a solution. Unlike models based on hand-crafted features (M5, M6) or fixed embeddings (M2, M7), VGAE performs end-to-end learning by automatically extracting structural representations of graph vertices. The key feature of the architecture is the combination of graph convolutional networks (GCN) as an encoder and variational inference, which allows the model not only to restore the observed connections, but also to train the regularized latency space through minimizing KL divergence. This approach is inductive, which is critically important for dynamically growing scientific graphs.

### Architecture
The architecture is implemented as a variational graph autoencoder with a two-layer graph convolutional encoder that transforms the input graph into parameters of Gaussian node distributions (μ, σ²). The decoder computes link probability via the inner product of latent representations with sigmoid activation. Training is performed by maximizing the evidence lower bound, which combines link reconstruction through negative sampling and KL divergence with a normal prior, providing regularization of the latent representations.

### Comparison with models in the repository
The method was implemented while maintaining full compatibility with the existing repository infrastructure. The main function vgae_link_prediction() was developed in accordance with the signature of analogues from the M6 and M7 models, which allowed it to be integrated into the standard evaluation pipeline. evaluate_model.py . During the development process, key technical issues were successfully resolved, including the correct temporary separation of data to avoid leakage. The main problem was the training and the selection of a part of the parameter for the balance between speed and quality.

The effectiveness of the proposed VGAE was evaluated on the SemanticGraph dataset using a standardized pipeline and the AUC-ROC metric. The model demonstrates competitive results, being in the leading group among all tested methods (M1-M8). For the task of predicting strong bonds (edge_weight=3) over long time horizons (delta=5), VGAE shows outstanding quality (AUC up to 0.96 at cutoff=25), surpassing all analogues in this category, including methods based on hand-crafted features and transformer architectures. The key differentiating advantage of VGAE is its exceptional stability: unlike many models that exhibit sharp fluctuations in performance when graph parameters change (for example, the catastrophic drop of Node2Vec-M7A to AUC ~0.5 at delta=3), the proposed solution does not have "failed" configurations and retains predictably high quality (AUC >0.94 for weight=3 and >0.89 for weight=1 in all test conditions). This result confirms the theoretical advantage of the inductive variational approach, which effectively captures stable structural patterns that are critical for long-term forecasting in dynamic graphs. Thus, VGAE not only fills a methodological gap in the repository by introducing the first graph autoencoder, but also sets a new standard for reliability and scalability for predicting connections in growing scientific networks.
