#!/usr/bin/env python

import os
import random
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
import torch.nn as nn
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
from tqdm.auto import tqdm

from .layer import *
from .model import *
from .utils import *


def TISCOPE_integration(
    adata,
    slice_name="batch",
    batch_name="batch",
    k=10,
    batch_size=4096,
    seed=3407,
    GPU=0,
    epoch=1000,
    lr=0.003,
    patience=10,
    num_heads=6,
    knn_corrected=True,
    slice_inference=True,
    infer_device="cpu",
    alpha=0.1,
    outdir="./",
    verbose=False,
):
    """
    TISCOPE:

    This function performs integrative analysis of spatial omics data.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spot/cell x gene expression matrix.
    slice_name : str, optional
        Column name in adata.obs that identifies different tissue slices/batches for training.
        Default: 'batch'
    batch_name : str, optional
        Column name in adata.obs that identifies biological batches. Default: 'batch'
    k : int, optional
        Number of nearest neighbors for constructing spatial adjacency matrix. Default: 10
    batch_size : int, optional
        Number of samples per batch during training. Default: 4096
    seed : int, optional
        Random seed for reproducibility. Default: 3407
    GPU : int, optional
        GPU device index to use (if available). Default: 0
    epoch : int, optional
        Maximum number of training epochs. Default: 1000
    lr : float, optional
        Learning rate for optimizer. Default: 0.003
    patience : int, optional
        Patience for early stopping. Default: 10
    num_heads : int, optional
        Number of attention heads in the model. Default: 6
    knn_corrected : bool, optional
        Whether to use batch-balanced KNN for neighborhood graph construction. Default: True
    slice_inference : bool, optional
        Whether to perform inference separately for each slice. Default: True
    infer_device : str, optional
        Device to use for model inference ('cpu' or 'cuda'). Default: 'cpu'
    outdir : str, optional
        Output directory for results. Default: './'
    verbose : bool, optional
        Whether to print detailed progress information. Default: False

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with low-dimensional representations stored in adata.obsm['latent'],
        and computed neighbors and UMAP embeddings.

    Output Files
    ------------
    {outdir}/adata.h5ad
        AnnData object containing the analysis results.
    {outdir}/model.pt
        Trained model checkpoint.
    """

    # Set random seeds for reproducibility
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Step 1: Graph Construction
    print("Constructing spatial graph...")
    adata_batch_list = []
    slices = np.unique(adata.obs[slice_name])
    batches = np.unique(adata.obs[batch_name])
    n_neighbors = k
    spatial_key = "spatial"
    adj_key = "spatial_connectivities"

    # Process each slice separately to construct spatial graphs
    if len(slices) > 1:
        for s in slices:
            print(f"Processing slice {s}...")
            adata_batch = adata[adata.obs[slice_name] == s]

            # Compute spatial neighborhood graph using SquidPy
            print("Computing spatial neighborhood graph...")
            sq.gr.spatial_neighbors(adata_batch, coord_type="generic", spatial_key=spatial_key, n_neighs=n_neighbors)

            # Ensure adjacency matrix is symmetric
            adata_batch.obsp[adj_key] = adata_batch.obsp[adj_key].maximum(adata_batch.obsp[adj_key].T)
            adata_batch_list.append(adata_batch)
        adata = ad.concat(adata_batch_list, join="inner")
    else:
        # Single slice case
        adata_batch = adata.copy()
        print("Computing spatial neighborhood graph...")
        sq.gr.spatial_neighbors(adata_batch, coord_type="generic", spatial_key=spatial_key, n_neighs=n_neighbors)

        # Ensure adjacency matrix is symmetric
        adata_batch.obsp[adj_key] = adata_batch.obsp[adj_key].maximum(adata_batch.obsp[adj_key].T)
        adata_batch_list.append(adata_batch)

    # Combine spatial graphs from all slices as disconnected components
    batch_connectivities = []
    len_before_batch = 0

    for i in range(len(adata_batch_list)):
        if i == 0:  # First batch
            after_batch_connectivities_extension = sp.csr_matrix((adata_batch_list[0].shape[0], (adata.shape[0] - adata_batch_list[0].shape[0])))
            batch_connectivities.append(sp.hstack((adata_batch_list[0].obsp[adj_key], after_batch_connectivities_extension)))
        elif i == (len(adata_batch_list) - 1):  # Last batch
            before_batch_connectivities_extension = sp.csr_matrix((adata_batch_list[i].shape[0], (adata.shape[0] - adata_batch_list[i].shape[0])))
            batch_connectivities.append(sp.hstack((before_batch_connectivities_extension, adata_batch_list[i].obsp[adj_key])))
        else:  # Middle batches
            before_batch_connectivities_extension = sp.csr_matrix((adata_batch_list[i].shape[0], len_before_batch))
            after_batch_connectivities_extension = sp.csr_matrix((adata_batch_list[i].shape[0], (adata.shape[0] - adata_batch_list[i].shape[0] - len_before_batch)))
            batch_connectivities.append(sp.hstack((before_batch_connectivities_extension, adata_batch_list[i].obsp[adj_key], after_batch_connectivities_extension)))
        len_before_batch += adata_batch_list[i].shape[0]

    adata.obsp[adj_key] = sp.vstack(batch_connectivities)

    # Extract features and adjacency matrix
    n = adata.shape[0]
    A = adata.obsp["spatial_connectivities"].copy()
    features = adata.X.A.copy()

    # Step 2: Compute neighborhood gene expression profiles
    print("Computing neighborhood molecular profiles...")
    neighbor_avg = compute_neighbor_average(adata, A)

    # Create environmental features from neighborhood expression
    df_env = pd.DataFrame(neighbor_avg / n_neighbors, index=adata.obs.index, columns=adata.var.index)
    env_feat = torch.tensor(np.nan_to_num(df_env.values.astype("float32")), dtype=torch.float)

    # Step 3: Construct KNN graph for biological neighborhood
    print("Constructing microenvironment similarity neighborhood graph...")
    if df_env.shape[1] < 31:
        if knn_corrected:
            import bbknn.matrix

            distances, connectivities, parameters = bbknn.matrix.bbknn(df_env.values.astype("float32"), batch_list)
            adj = connectivities.copy()
        else:
            # Direct KNN for low-dimensional features
            train_neighbors = NearestNeighbors(n_neighbors=31, metric="euclidean").fit(np.nan_to_num(df_env.values.astype("float32")))
            adj = train_neighbors.kneighbors_graph(np.nan_to_num(df_env.values.astype("float32")))
    else:
        # Use PCA for dimensionality reduction
        if knn_corrected:
            # Batch-balanced KNN to account for batch effects
            pca_model = PCA(n_components=30, random_state=seed)
            pca_matrix = pca_model.fit_transform(df_env.values.astype("float32"))
            batch_list = adata.obs[batch_name].values
            import bbknn.matrix

            distances, connectivities, parameters = bbknn.matrix.bbknn(pca_matrix, batch_list)
            adj = connectivities.copy()
        else:
            # Standard KNN
            pca_model = PCA(n_components=30, random_state=seed)
            decomposed_x = pca_model.fit_transform(df_env.values.astype("float32"))
            train_neighbors = NearestNeighbors(n_neighbors=31, metric="euclidean").fit(np.nan_to_num(decomposed_x.astype("float32")))
            adj = train_neighbors.kneighbors_graph(np.nan_to_num(decomposed_x.astype("float32")))

    # Combine spatial and biological adjacency matrices
    A = A + adj
    A[A > 1] = 1  # Binarize combined adjacency

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]

    # Dataset statistics
    n = num_nodes
    e = edge_index.shape[1]
    d = node_feat.shape[1]
    print(f"Dataset stats - Nodes: {n} | Edges: {e} | Features: {d}")

    # Preprocess graph for undirected, no self-loops
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_index = edge_index.to(torch.long)

    # Prepare training data with domain labels
    le = LabelEncoder()
    le.fit(adata.obs[batch_name].cat.categories)
    y = le.transform(adata.obs[batch_name])
    y = torch.tensor(y).long()

    # Create PyTorch Geometric data object
    data_obj = Data(edge_index=edge_index, x=node_feat)
    data_obj.num_nodes = node_feat.shape[0]
    data_obj.n_id = torch.arange(data_obj.num_nodes)
    data_obj.domain = y
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    num_features = data_obj.num_features

    # Step 4: Model initialization
    device = torch.device(f"cuda:{GPU}") if torch.cuda.is_available() else torch.device("cpu")
    model = TISCOPE(num_features, num_heads, [["fc", d, len(adata.obs[batch_name].cat.categories), "relu"]]).to(device)

    model.train()
    print("Model architecture:")
    print(model)
    print("Starting TISCOPE training...")

    # Loss function and early stopping
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, checkpoint_file=os.path.join(outdir, "model.pt"))

    # Optimizer with different weight decay for different parameter groups
    optimizer = torch.optim.Adam([{"params": model.params1, "weight_decay": 1e-4}, {"params": model.params2, "weight_decay": 1e-4}], lr=lr)

    # Data loader for neighborhood sampling
    train_loader = NeighborLoader(data_obj, num_neighbors=[20], batch_size=batch_size, shuffle=True)

    # Step 5: Training loop
    for epoch in tqdm(range(0, epoch + 1)):
        model.to(device)
        model.train()

        epoch_loss = defaultdict(float)

        for batch in train_loader:
            x_i = batch.x.to(device)
            edge_index_i = batch.edge_index.to(device)
            y_i = batch.domain.to(device)

            optimizer.zero_grad()
            z_i, _, out_i = model(x_i, edge_index_i, y_i)

            # Compute losses
            graph_loss = model.graph_loss(z_i, edge_index_i)
            recon_loss = criterion(out_i, x_i)

            # Combined loss with weighting
            loss = {"recon_loss": 20 * (1 - alpha) * recon_loss, "graph_loss": alpha * graph_loss}

            sum(loss.values()).backward()
            optimizer.step()

            # Accumulate losses for logging
            for k, v in loss.items():
                epoch_loss[k] += loss[k].item()

        # Logging
        if verbose and epoch % 5 == 0:
            print(f"Epoch: {epoch:02d}, Loss: {sum(epoch_loss.values()):.4f}")

        # Early stopping check
        epoch_loss = {k: v / len(train_loader) for k, v in epoch_loss.items()}
        early_stopping(sum(epoch_loss.values()), model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Step 6: Inference
    print("Performing inference...")
    if slice_inference:
        if infer_device == "cpu":
            device = "cpu"

        # Load trained model
        model = TISCOPE(num_features, num_heads, [["fc", d, len(adata.obs[batch_name].cat.categories), "relu"]]).to(device)
        pretrained_dict = torch.load(os.path.join(outdir, "model.pt"), map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        # Inference per slice to save memory
        embedding = pd.DataFrame(index=adata.obs.index, columns=[f"latent_{i}" for i in range(10)])
        for s in slices:
            slice_data = adata[adata.obs[slice_name] == s]
            edge_index = torch.tensor(slice_data.obsp["spatial_connectivities"].nonzero(), dtype=torch.long)
            x = torch.tensor(slice_data.X.A, dtype=torch.float)
            y = torch.tensor(le.transform(slice_data.obs[slice_name])).long()

            with torch.no_grad():
                z, _, _ = model(x.to(device), edge_index.to(device), y.to(device))

            embedding.loc[slice_data.obs.index] = z.cpu().numpy()

            # Clean up to free memory
            del x, y, z, edge_index
            torch.cuda.empty_cache()

        adata.obsm["latent"] = embedding.loc[adata.obs.index].values.astype(float)
    else:
        # Whole dataset inference
        if infer_device == "cpu":
            device = "cpu"

        # Load trained model
        model = TISCOPE(num_features, num_heads, [["fc", d, len(adata.obs[batch_name].cat.categories), "relu"]]).to(device)
        pretrained_dict = torch.load(os.path.join(outdir, "model.pt"), map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        with torch.no_grad():
            z, _, _ = model(node_feat.to(device), edge_index.to(device), y.to(device))
        adata.obsm["latent"] = z.cpu().numpy().copy()

    # Step 7: Post-processing (neighbors and UMAP)
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10, use_rep="latent", random_state=42, key_added="SPACE")
    sc.tl.umap(adata, random_state=42, neighbors_key="SPACE")

    # Save results
    output_path = os.path.join(outdir, "adata.h5ad")
    adata.write(output_path)
    print(f"Results saved to {output_path}")

    return adata


def TISCOPE_projection(
    adata,
    adata_ref,
    slice_name="batch",
    batch_name="batch",
    k=10,
    seed=3407,
    GPU=0,
    num_heads=6,
    knn_corrected=True,
    slice_inference=True,
    infer_device="cpu",
    model_path="./",
    outdir="./",
    outfile="adata_projection.h5ad",
    verbose=False,
):
    """
    Project new data using a pre-trained TISCOPE model.

    This function projects new spatial omics data into the latent space
    learned by a pre-trained TISCOPE model, enabling integration with reference data.

    Parameters
    ----------
    adata : anndata.AnnData
        New AnnData object to be projected (query data).
    adata_ref : anndata.AnnData
        Reference AnnData object used for training the model.
    slice_name : str, optional
        Column name identifying tissue slices. Default: 'batch'
    batch_name : str, optional
        Column name identifying biological batches. Default: 'batch'
    k : int, optional
        Number of nearest neighbors for spatial graph. Default: 10
    seed : int, optional
        Random seed for reproducibility. Default: 3407
    GPU : int, optional
        GPU device index. Default: 0
    num_heads : int, optional
        Number of attention heads. Default: 6
    knn_corrected : bool, optional
        Use batch-balanced KNN. Default: True
    slice_inference : bool, optional
        Perform inference per slice. Default: True
    infer_device : str, optional
        Device for inference. Default: 'cpu'
    model_path : str, optional
        Pre-trained model directory. Default: './'
    outdir : str, optional
        Output directory. Default: './'
    outfile : str, optional
        Output filename. Default: 'adata_projection.h5ad'
    verbose : bool, optional
        Print verbose output. Default: False

    Returns
    -------
    adata : anndata.AnnData
        Combined AnnData object with reference and query data projected into shared latent space.
    """

    # Set random seeds for reproducibility
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs(outdir, exist_ok=True)

    # Ensure query data has same genes as reference
    adata = adata[:, adata_ref.var.index]

    # Graph construction (same as SPACEX_function)
    print("Constructing graph for projection...")
    adata_batch_list = []
    slices = np.unique(adata.obs[slice_name])
    batches = np.unique(adata.obs[batch_name])
    n_neighbors = k
    spatial_key = "spatial"
    adj_key = "spatial_connectivities"

    # Process each slice
    for s in slices:
        print(f"Processing slice {s}...")
        adata_batch = adata[adata.obs[slice_name] == s]

        print("Computing spatial neighborhood graph...")
        sq.gr.spatial_neighbors(adata_batch, coord_type="generic", spatial_key=spatial_key, n_neighs=n_neighbors)

        # Ensure symmetric adjacency
        adata_batch.obsp[adj_key] = adata_batch.obsp[adj_key].maximum(adata_batch.obsp[adj_key].T)
        adata_batch_list.append(adata_batch)

    adata = ad.concat(adata_batch_list, join="inner")

    # Combine graphs from all slices
    batch_connectivities = []
    len_before_batch = 0

    for i in range(len(adata_batch_list)):
        if i == 0:  # First batch
            after_batch_extension = sp.csr_matrix((adata_batch_list[0].shape[0], (adata.shape[0] - adata_batch_list[0].shape[0])))
            batch_connectivities.append(sp.hstack((adata_batch_list[0].obsp[adj_key], after_batch_extension)))
        elif i == (len(adata_batch_list) - 1):  # Last batch
            before_batch_extension = sp.csr_matrix((adata_batch_list[i].shape[0], (adata.shape[0] - adata_batch_list[i].shape[0])))
            batch_connectivities.append(sp.hstack((before_batch_extension, adata_batch_list[i].obsp[adj_key])))
        else:  # Middle batches
            before_extension = sp.csr_matrix((adata_batch_list[i].shape[0], len_before_batch))
            after_extension = sp.csr_matrix((adata_batch_list[i].shape[0], (adata.shape[0] - adata_batch_list[i].shape[0] - len_before_batch)))
            batch_connectivities.append(sp.hstack((before_extension, adata_batch_list[i].obsp[adj_key], after_extension)))
        len_before_batch += adata_batch_list[i].shape[0]

    adata.obsp[adj_key] = sp.vstack(batch_connectivities)

    # Feature extraction and graph construction
    n = adata.shape[0]
    A = adata.obsp["spatial_connectivities"].copy()
    features = adata.X.A.copy()

    # Neighborhood feature computation
    neighbor_avg = compute_neighbor_average(adata, A)

    # Create environmental features from neighborhood expression
    df_env = pd.DataFrame(neighbor_avg / n_neighbors, index=adata.obs.index, columns=adata.var.index)
    env_feat = torch.tensor(np.nan_to_num(df_env.values.astype("float32")), dtype=torch.float)

    # KNN graph construction
    if df_env.shape[1] < 31:
        if knn_corrected:
            import bbknn.matrix

            distances, connectivities, parameters = bbknn.matrix.bbknn(df_env.values.astype("float32"), batch_list)
            adj = connectivities.copy()
        else:
            # Direct KNN for low-dimensional features
            train_neighbors = NearestNeighbors(n_neighbors=31, metric="euclidean").fit(np.nan_to_num(df_env.values.astype("float32")))
            adj = train_neighbors.kneighbors_graph(np.nan_to_num(df_env.values.astype("float32")))
    else:
        if knn_corrected:
            pca_model = PCA(n_components=30, random_state=seed)
            pca_matrix = pca_model.fit_transform(df_env.values.astype("float32"))
            batch_list = adata.obs[batch_name].values
            import bbknn.matrix

            distances, connectivities, parameters = bbknn.matrix.bbknn(pca_matrix, batch_list)
            adj = connectivities.copy()
        else:
            pca_model = PCA(n_components=30, random_state=seed)
            decomposed_x = pca_model.fit_transform(df_env.values.astype("float32"))
            train_neighbors = NearestNeighbors(n_neighbors=31, metric="euclidean").fit(np.nan_to_num(decomposed_x.astype("float32")))
            adj = train_neighbors.kneighbors_graph(np.nan_to_num(decomposed_x.astype("float32")))

    # Combine spatial and biological graphs
    A = A + adj
    A[A > 1] = 1

    # Convert to PyTorch format
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)

    # Graph preprocessing
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=node_feat.shape[0])
    edge_index = edge_index.to(torch.long)
    num_features = node_feat.shape[1]

    # Device setup
    device = torch.device(f"cuda:{GPU}") if torch.cuda.is_available() else torch.device("cpu")

    # Inference
    if slice_inference:
        if infer_device == "cpu":
            device = "cpu"

        # Load pre-trained model
        model = TISCOPE(num_features, num_heads, [["fc", num_features, len(adata_ref.obs[batch_name].cat.categories), "relu"]]).to(device)
        pretrained_dict = torch.load(os.path.join(model_path, "model.pt"), map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        # Per-slice inference
        embedding = pd.DataFrame(index=adata.obs.index, columns=[f"latent_{i}" for i in range(10)])
        for s in slices:
            slice_data = adata[adata.obs[slice_name] == s]
            edge_index = torch.tensor(slice_data.obsp["spatial_connectivities"].nonzero(), dtype=torch.long)
            x = torch.tensor(slice_data.X.A, dtype=torch.float)

            with torch.no_grad():
                z, _ = model.encoder(x.to(device), edge_index.to(device))

            embedding.loc[slice_data.obs.index] = z.cpu().numpy()

            del x, z, edge_index
            torch.cuda.empty_cache()

        adata.obsm["latent"] = embedding.loc[adata.obs.index].values.astype(float)
    else:
        # Whole dataset inference
        if infer_device == "cpu":
            device = "cpu"

        model = TISCOPE(num_features, num_heads, [["fc", num_features, len(adata_ref.obs[batch_name].cat.categories), "relu"]]).to(device)
        pretrained_dict = torch.load(os.path.join(model_path, "model.pt"), map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        with torch.no_grad():
            z, _ = model.encoder(node_feat.to(device), edge_index.to(device))
        adata.obsm["latent"] = z.cpu().numpy().copy()

    # Combine reference and query data
    adata = ad.concat([adata_ref, adata], label="projection", keys=["reference", "query"], index_unique=None)

    # Compute neighbors and UMAP on combined data
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10, use_rep="latent", random_state=42, key_added="SPACE")
    sc.tl.umap(adata, random_state=42, neighbors_key="SPACE")

    # Save results
    output_path = os.path.join(outdir, outfile)
    adata.obs = adata.obs.astype(str)  # Ensure string type for compatibility
    adata.write(output_path)
    print(f"Projection results saved to {output_path}")

    return adata
