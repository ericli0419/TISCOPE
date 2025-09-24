"""
TISCOPE Model Definition.
This module defines the core TISCOPE model architecture.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric import seed_everything
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops

from .layer import *

seed = 42
EPS = 1e-15


class TISCOPE(nn.Module):
    """
    TISCOPE model for spatial transcriptomics data integration and projection.

    This model uses a GAT encoder to learn latent representations and
    reconstructs both the graph structure and node features.
    """

    def __init__(self, num_features, heads, decoder_config, latent_dim=10):
        """
        Initializes the TISCOPE model.

        Args:
            num_features (int): Number of input features (genes).
            heads (int): Number of attention heads for GAT layers.
            decoder_config (list): Configuration for the feature decoder (NN).
                                   Format: [['fc', out_channels, n_domain, 'activation'], ...]
            latent_dim (int, optional): Dimension of the latent space. Defaults to 10.
        """
        super(TISCOPE, self).__init__()
        seed_everything(seed)
        self.latent_dim = latent_dim
        # Encoder: Maps input features to latent space
        self.encoder = GAT_Encoder(
            in_channels=num_features,
            num_heads={"first": heads, "second": heads, "mean": heads},
            hidden_dims=[128, 128],
            dropout=[0.3, 0.3],
            concat={"first": True, "second": True},
            latent_dim=latent_dim,  # Pass latent_dim to encoder
        )
        # Decoder for graph structure (edge prediction)
        self.decoder = InnerProductDecoder()
        # Decoder for node features (reconstruction)
        self.decoder_x = NN(self.encoder.latent_dim, decoder_config)

        # Separate parameters for different parts if needed for optimization
        self.params1 = list(self.encoder.parameters())
        self.params2 = list(self.decoder_x.parameters())

    def forward(self, x, edge_index, domain_labels):
        """
        Performs forward pass through the model.

        Args:
            x (torch.Tensor): Node features [num_nodes, num_features].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].
            domain_labels (torch.Tensor): Domain labels for batch-specific normalization [num_nodes].

        Returns:
            tuple: (z, attention_weights, reconstructed_x)
                - z (torch.Tensor): Latent representations [num_nodes, latent_dim].
                - attention_weights (tuple): Attention weights from GAT layers.
                - reconstructed_x (torch.Tensor): Reconstructed node features [num_nodes, num_features].
        """
        z, attn_w = self.encoder(x, edge_index)
        x_recon = self.decoder_x(z, domain_labels)
        return z, attn_w, x_recon

    def graph_loss(self, z, pos_edge_index, neg_edge_index=None):
        """
        Computes the graph reconstruction loss using positive and negative sampling.

        Args:
            z (torch.Tensor): Latent node representations [num_nodes, latent_dim].
            pos_edge_index (torch.Tensor): Positive edges [2, num_pos_edges].
            neg_edge_index (torch.Tensor, optional): Negative edges [2, num_neg_edges].
                                                     If None, sampled automatically.

        Returns:
            torch.Tensor: The combined positive and negative log-likelihood loss.
        """
        # Positive loss
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(pos_pred + EPS).mean()

        # Prepare for negative sampling (remove self-loops added for positive)
        pos_edge_index_cleaned, _ = remove_self_loops(pos_edge_index)
        # Add self-loops back if needed by the decoder (check InnerProductDecoder behavior)
        # pos_edge_index_cleaned, _ = add_self_loops(pos_edge_index_cleaned)

        # Negative sampling and loss
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index_cleaned, z.size(0))
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()

        return pos_loss + neg_loss

    def load_model(self, path):
        """
        Loads a pre-trained model state dictionary.

        Args:
            path (str): Path to the saved model state dictionary (.pt file).
        """
        try:
            pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            # Filter out mismatched keys (e.g., if model structure changed slightly)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")


class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss plateaus.
    """

    def __init__(self, patience=10, verbose=False, checkpoint_file=""):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait after last improvement.
                            Default: 10.
            verbose (bool): If True, prints a message for each improvement.
                            Default: False.
            checkpoint_file (str): Path to save the best model checkpoint.
                                   Default: ''.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        """
        Checks if training should be stopped based on the current loss.

        Args:
            loss (float): Current validation loss.
            model (nn.Module): The model being trained.
        """
        if np.isnan(loss):
            self.early_stop = True
            print("Early stopping triggered due to NaN loss.")
            return

        score = -loss  # We want to maximize score (minimize loss)

        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            # Loss increased (score decreased)
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")
                # Load the best model found during training
                model.load_model(self.checkpoint_file)
        else:
            # Loss decreased (score increased), save best model
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0  # Reset counter

    def save_checkpoint(self, loss, model):
        """
        Saves the model when validation loss decreases.

        Args:
            loss (float): Current validation loss.
            model (nn.Module): The model to save.
        """
        if self.verbose:
            print(f"Loss decreased ({self.loss_min:.6f} --> {loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss
