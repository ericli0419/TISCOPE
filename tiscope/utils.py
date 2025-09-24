"""
Utility functions for TISCOPE.
"""

import gc
import time

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse


def convert_to_dense(X):
    """
    Converts a sparse matrix or AnnData view to a dense NumPy array.

    Args:
        X (np.ndarray, scipy.sparse matrix, or anndata.View):
            Input data structure.

    Returns:
        np.ndarray: Dense representation of the input.

    Raises:
        TypeError: If the input type is not supported.
        ValueError: If the resulting dense matrix contains non-finite values.
    """
    if isinstance(X, np.ndarray):
        return X
    elif issparse(X) or str(type(X)).endswith("ArrayView'>"):
        X_dense = X.toarray()
    else:
        raise TypeError(f"Input format not recognized. Must be dense/sparse matrix or AnnData view. Got {type(X)}")

    if not np.isfinite(X_dense).all():
        raise ValueError("Matrix contains non-finite values (NaN or Inf).")
    return X_dense


def compute_neighbor_average(adata: AnnData, adjacency_matrix):
    """
    Computes the average feature value of neighbors for each node.

    This is done by multiplying the adjacency matrix (weights)
    with the feature matrix (X).

    Args:
        adata (AnnData): AnnData object containing the feature matrix `adata.X`.
        adjacency_matrix (scipy.sparse matrix or np.ndarray):
            Adjacency matrix where element (i, j) represents the weight
            of the edge from node i to node j.

    Returns:
        np.ndarray: A dense matrix where each row represents the
                    neighbor-averaged features for the corresponding node.
    """
    start_time = time.perf_counter()
    X_dense = convert_to_dense(adata.X)
    # Matrix multiplication: (N x N) @ (N x F) -> (N x F)
    # Each row i in the result is the sum of feature vectors of i's neighbors,
    # weighted by the adjacency matrix.
    neighbor_avg = adjacency_matrix @ X_dense
    gc.collect()  # Suggest garbage collection
    elapsed = round((time.perf_counter() - start_time) / 60, 2)
    print(f"Computed neighbor average in {elapsed} mins")
    return neighbor_avg


# label transfer

# over correction score

# enrichment score

# enrichment score plot
