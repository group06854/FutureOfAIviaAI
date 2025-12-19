"""
VGAE model for link prediction with model saving
Compatible with existing repository: FutureOfAIviaAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from datetime import datetime
from utils import NUM_OF_VERTICES, create_training_data_biased

# ============================================================================
# 1. VGAE ARCHITECTURE
# ============================================================================

class VariationalGCNEncoder(nn.Module):
    """
    Variational GCN Encoder for VGAE
    Takes graph -> produces μ and logσ²
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logvar = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class VGAE(nn.Module):
    """
    Complete VGAE Model
    Encoder + Reparameterization + Inner Product Decoder
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = VariationalGCNEncoder(in_channels, hidden_channels, out_channels)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + ε * σ"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu
    
    def forward(self, x, edge_index):
        """Full forward pass: returns z, μ, logσ²"""
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z, edge_index):
        """Inner product decoder: sigmoid(z_i^T * z_j)"""
        row, col = edge_index
        return torch.sigmoid((z[row] * z[col]).sum(dim=1))

# ============================================================================
# 2. TRAINER WITH SAVE/LOAD
# ============================================================================

class VGAETrainer:
    """
    Trainer for VGAE with full model persistence
    Handles: training, saving, loading, inference
    """
    
    def __init__(self, latent_dim=32, hidden_dim=64, device=None, model_dir='saved_models'):
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model directory
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model initialization
        self.model = VGAE(
            in_channels=1,  # Single feature per node
            hidden_channels=hidden_dim,
            out_channels=latent_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Training history
        self.history = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': [],
            'epochs_trained': 0, 'best_loss': float('inf'),
            'model_dir': model_dir
        }
    
    # ==================== TRAINING ====================
    
    def vgae_loss(self, z, mu, logvar, edge_index, beta=0.001):
        """VGAE loss: reconstruction + β * KL divergence"""
        # Reconstruction (positive edges)
        pos_pred = self.model.decode(z, edge_index)
        recon_loss = -torch.log(pos_pred + 1e-15).mean()
        
        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index, 
            num_nodes=z.size(0),
            num_neg_samples=edge_index.size(1)
        ).to(self.device)
        
        neg_pred = self.model.decode(z, neg_edge_index)
        recon_loss += -torch.log(1 - neg_pred + 1e-15).mean()
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def train_epoch(self, data, beta=0.001):
        """Single training epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        x = data.get('x', torch.ones((data['num_nodes'], 1), device=self.device))
        edge_index = data['edge_index'].to(self.device)
        
        # Forward pass
        z, mu, logvar = self.model(x, edge_index)
        
        # Loss calculation
        total_loss, recon_loss, kl_loss = self.vgae_loss(z, mu, logvar, edge_index, beta)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), recon_loss.item(), kl_loss.item()
    
    def train(self, data, epochs=200, beta=0.001, save_every=50, save_best=True):
        """Full training loop with automatic saving"""
        print(f"Training VGAE for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss, recon_loss, kl_loss = self.train_epoch(data, beta)
            
            # Update history
            self.history['total_loss'].append(total_loss)
            self.history['recon_loss'].append(recon_loss)
            self.history['kl_loss'].append(kl_loss)
            self.history['epochs_trained'] += 1
            
            # Save best model
            if save_best and total_loss < self.history['best_loss']:
                self.history['best_loss'] = total_loss
                self.save_model("vgae_best.pth")
            
            # Periodic checkpoint
            if save_every and (epoch + 1) % save_every == 0:
                self.save_model(f"vgae_epoch_{epoch+1}.pth")
            
            # Progress output
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss:.4f}")
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_model(f"vgae_final_{timestamp}.pth")
        
        return self.history
    
    # ==================== SAVE/LOAD ====================
    
    def save_model(self, filename="vgae_model.pth"):
        """Save complete model state"""
        save_path = os.path.join(self.model_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'latent_dim': self.model.encoder.conv_mu.out_channels,
            'hidden_dim': self.model.encoder.conv1.out_channels,
            'config': {
                'device': str(self.device),
                'model_dir': self.model_dir,
                'save_time': datetime.now().isoformat()
            }
        }
        
        torch.save(checkpoint, save_path)
        return save_path
    
    def load_model(self, filepath):
        """Load model from checkpoint"""
        if not os.path.exists(filepath):
            filepath = os.path.join(self.model_dir, filepath)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history.update(checkpoint['history'])
        
        print(f"Model loaded from {filepath}")
        print(f"Previously trained: {self.history['epochs_trained']} epochs")
        
        return checkpoint
    
    # ==================== INFERENCE ====================
    
    def get_embeddings(self, data):
        """Extract node embeddings (uses μ as deterministic embedding)"""
        self.model.eval()
        with torch.no_grad():
            x = data.get('x', torch.ones((data['num_nodes'], 1), device=self.device))
            edge_index = data['edge_index'].to(self.device)
            
            # Get μ only (deterministic embedding)
            mu, _ = self.model.encoder(x, edge_index)
            return mu.cpu()
    
    def predict_links(self, embeddings, vertex_pairs):
        """Predict link probabilities for given vertex pairs"""
        self.model.eval()
        embeddings = embeddings.to(self.device)
        vertex_pairs = torch.tensor(vertex_pairs, device=self.device)
        
        with torch.no_grad():
            rows, cols = vertex_pairs[:, 0], vertex_pairs[:, 1]
            scores = torch.sigmoid((embeddings[rows] * embeddings[cols]).sum(dim=1))
        
        return scores.cpu().numpy()

# ============================================================================
# 4. MAIN INTERFACE FUNCTION (for evaluate_model.py compatibility)
# ============================================================================

def link_prediction_semnet(full_dynamic_graph_sparse, unconnected_vertex_pairs,
                         year_start, years_delta, vertex_degree_cutoff, 
                         min_edges, hyper_parameters, data_source=""):
    """
    MAIN FUNCTION: Compatible interface with evaluate_model.py
    
    Parameters exactly match link_prediction_semnet from simple_model.py
    
    Returns: sorted_predictions_eval (indices from most to least probable)
    """
    print(f"\n{'='*60}")
    print(f"VGAE LINK PREDICTION")
    print(f"Dataset: {data_source}")
    print(f"Parameters: delta={years_delta}, cutoff={vertex_degree_cutoff}, min_edges={min_edges}")
    print(f"{'='*60}")
    
    # 1. Extract hyperparameters
    # hyper_parameters = [edges_used, percent_positive_examples, batch_size, lr_enc, rnd_seed]
    edges_used = hyper_parameters[0] if len(hyper_parameters) > 0 else 500000
    percent_positive_examples = hyper_parameters[1] if len(hyper_parameters) > 1 else 1
    # batch_size и lr_enc 
    rnd_seed = hyper_parameters[4] if len(hyper_parameters) > 4 else 42
    
    # 2. VGAE specific hyperparameters
    latent_dim = 32  
    epochs = 40     
    lr = 0.01        
    beta = 0.001 
    
    # 3. Create model directory
    if data_source:
        model_dir = f"vgae_models/{data_source.replace('.pkl', '')}"
    else:
        model_dir = "vgae_models/default"
    os.makedirs(model_dir, exist_ok=True)
    
    # 4. Prepare graph data 
    train_dynamic_graph_sparse, train_edges_for_checking, train_edges_solution = create_training_data_biased(
        full_dynamic_graph_sparse,
        year_start - years_delta, 
        years_delta,              
        min_edges=min_edges,
        edges_used=edges_used,
        vertex_degree_cutoff=vertex_degree_cutoff,
        data_source=data_source
    )
    

    edge_index = torch.tensor(train_dynamic_graph_sparse[:, :2].T, dtype=torch.long)
    
    data = {
        'edge_index': edge_index,
        'num_nodes': NUM_OF_VERTICES  # From utils.py
    }
    
    # 6. Initialize and train model
    trainer = VGAETrainer(latent_dim=latent_dim, model_dir=model_dir)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
    
    print(f"Training on {NUM_OF_VERTICES} vertices, {edge_index.shape[1]} edges")
    trainer.train(data, epochs=epochs, beta=beta)
    
    # 7. Get embeddings and make predictions
    embeddings = trainer.get_embeddings(data)
    predictions = trainer.predict_links(embeddings, np.array(unconnected_vertex_pairs))
    
    # 8. Sort indices (most probable first)
    sorted_indices = np.argsort(predictions)[::-1]
    
    print(f"\nPredictions complete")
    print(f"Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Model saved in: {model_dir}")
    print(f"{'='*60}\n")
    
    return sorted_indices