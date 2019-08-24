# Requires:
# Python, numpy, numba
#
# Can be installed with:
# pip install numpy numba
#
# Description:
# Short snippets for numpy functions parallelized with numba.

import numpy as np
import numba


@numba.njit(parallel=True)
def parallel_searchsorted(a, v, side="left"):
    indices = np.empty(v.shape, dtype=np.int64)
    for i in numba.prange(v.shape[0]):
        indices[i] = np.searchsorted(a[i], v[i], side=side)
    return indices


@numba.njit(parallel=True)
def parallel_knn_indices(X, n_neighbors, axis=-1, kind=None):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int64)
    for i in numba.prange(knn_indices.shape[0]):
        knn_indices[i] = X[i].argsort(axis=axis, kind=kind)[:n_neighbors]
    return knn_indices


@numba.njit(parallel=True)
def parallel_argsort(X, axis=-1, kind=None):
    index_array = np.empty(X.shape, dtype=np.int64)
    for i in numba.prange(index_array.shape[0]):
        index_array[i] = X[i].argsort(axis=axis, kind=kind)
    return index_array


@numba.njit(parallel=True)
def parallel_sort_by_argsort(X, argsort_indices):
    sorted_array = np.empty(X.shape, dtype=X.dtype)
    for i in numba.prange(sorted_array.shape[0]):
        sorted_array[i] = X[i][argsort_indices[i]]
    return sorted_array
