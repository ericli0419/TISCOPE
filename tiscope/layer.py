"""
TISCOPE Model Layers.
This module defines custom layers and blocks used in the TISCOPE model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch_geometric.nn import GATv2Conv

activation = {"relu": nn.ReLU(), "rrelu": nn.RReLU(), "sigmoid": nn.Sigmoid(), "leaky_relu": nn.LeakyReLU(), "tanh": nn.Tanh(), "": None}


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]

        return out


class Block(nn.Module):
    """
    Basic block consist of:
        fc -> bn -> act -> dropout
    """

    def __init__(self, input_dim, output_dim, norm="", act="", dropout=0):
        """
        Parameters
        ----------
        input_dim
            dimension of input
        output_dim
            dimension of output
        norm
            batch normalization,
                * '' represent no batch normalization
                * 1 represent regular batch normalization
                * int>1 represent domain-specific batch normalization of n domain
        act
            activation function,
                * relu -> nn.ReLU
                * rrelu -> nn.RReLU
                * sigmoid -> nn.Sigmoid()
                * leaky_relu -> nn.LeakyReLU()
                * tanh -> nn.Tanh()
                * '' -> None
        dropout
            dropout rate
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if type(norm) == int:
            if norm == 1:  # TO DO
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

        self.act = activation[act]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, y=None):
        h = self.fc(x)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == "DSBatchNorm":
                h = self.norm(h, y)
            else:
                h = self.norm(h)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class NN(nn.Module):
    """
    Neural network consist of multi Blocks
    """

    def __init__(self, input_dim, cfg):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            model structure configuration, 'fc' -> fully connected layer

        Example
        -------
        >>> latent_dim = 10
        >>> dec_cfg = [['fc', x_dim, n_domain, 'sigmoid']]
        >>> decoder = NN(latent_dim, dec_cfg)
        """
        super().__init__()
        net = []
        for i, layer in enumerate(cfg):
            if i == 0:
                d_in = input_dim
            if layer[0] == "fc":
                net.append(Block(d_in, *layer[1:]))
            d_in = layer[1]
        self.net = nn.ModuleList(net)

    def forward(self, x, y=None):
        for layer in self.net:
            x = layer(x, y)
        return x


class InnerProduct(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, act=torch.relu):
        super(InnerProduct, self).__init__()
        self.act = act

    def forward(self, z):
        recon_a = self.act(torch.mm(z, z.t()))
        return recon_a


class GAT_Encoder(nn.Module):
    """
    Graph Attention Encoder using GATv2Conv layers.

    Encodes graph-structured data into low-dimensional latent representations.
    """

    def __init__(self, in_channels, num_heads, hidden_dims, dropout, concat, latent_dim=10):
        """
        Initializes the GAT Encoder.

        Args:
            in_channels (int): Number of input features per node.
            num_heads (dict): Number of attention heads for each GAT layer.
                              Keys: 'first', 'second', 'mean'.
            hidden_dims (list): List of hidden dimensions for the first two layers.
            dropout (list): List of dropout rates for the first two layers.
            concat (dict): Whether to concatenate multi-head outputs for each layer.
                           Keys: 'first', 'second'.
            latent_dim (int, optional): Dimension of the final latent representation.
                                        Default is 10.
        """
        super(GAT_Encoder, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv = GATv2Conv  # Alias for the convolution layer type

        # First GAT layer
        self.hidden_layer1 = self.conv(in_channels=in_channels, out_channels=hidden_dims[0], heads=self.num_heads["first"], dropout=dropout[0], concat=concat["first"])
        # Calculate input dimension for the second layer
        in_dim_hidden1 = hidden_dims[0] * self.num_heads["first"] if concat["first"] else hidden_dims[0]

        # Second GAT layer
        self.hidden_layer2 = self.conv(in_channels=in_dim_hidden1, out_channels=hidden_dims[1], heads=self.num_heads["second"], dropout=dropout[1], concat=concat["second"])
        # Calculate input dimension for the final projection layer
        in_dim_hidden2 = hidden_dims[1] * self.num_heads["second"] if concat["second"] else hidden_dims[1]

        # Final layer to project to latent_dim (mean attention)
        self.conv_z = self.conv(
            in_channels=in_dim_hidden2,
            out_channels=self.latent_dim,
            heads=self.num_heads["mean"],
            concat=False,  # Mean aggregation for final layer
            dropout=0.2,  # Fixed dropout for final layer
        )

    def forward(self, x, edge_index):
        """
        Forward pass through the GAT encoder.

        Args:
            x (torch.Tensor): Node features [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].

        Returns:
            tuple: (z, attention_weights)
                - z (torch.Tensor): Latent representations [num_nodes, latent_dim].
                - attention_weights (tuple): Attention weights from each GAT layer.
        """
        # Layer 1
        h1, attn_w1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)
        h1 = F.leaky_relu(h1)
        # Layer 2
        h2, attn_w2 = self.hidden_layer2(h1, edge_index, return_attention_weights=True)
        h2 = F.leaky_relu(h2)
        h2 = F.dropout(h2, p=0.4, training=self.training)  # Apply dropout
        # Final projection layer
        z, attn_wz = self.conv_z(h2, edge_index, return_attention_weights=True)
        return z, (attn_w1, attn_w2, attn_wz)
