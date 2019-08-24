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
def parallel_searchsorted(a, v):
    indices = np.empty(v.shape, dtype=np.int64)
    for i in numba.prange(v.shape[0]):
        indices[i] = np.searchsorted(a[i], v[i])
    return indices


@numba.njit(parallel=True)
def parallel_searchsorted_left(a, v):
    indices = np.empty(v.shape, dtype=np.int64)
    for i in numba.prange(v.shape[0]):
        indices[i] = np.searchsorted(a[i], v[i], "left")
    return indices


@numba.njit(parallel=True)
def parallel_searchsorted_right(a, v):
    indices = np.empty(v.shape, dtype=np.int64)
    for i in numba.prange(v.shape[0]):
        indices[i] = np.searchsorted(a[i], v[i], "right")
    return indices


@numba.njit(parallel=True)
def parallel_knn_indices(X, n_neighbors):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int64)
    for i in numba.prange(knn_indices.shape[0]):
        knn_indices[i] = X[i].argsort()[:n_neighbors]
    return knn_indices


@numba.njit(parallel=True)
def parallel_knn_indices_quicksort(X, n_neighbors):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int64)
    for i in numba.prange(knn_indices.shape[0]):
        knn_indices[i] = X[i].argsort(kind="quicksort")[:n_neighbors]
    return knn_indices


@numba.njit(parallel=True)
def parallel_knn_indices_mergesort(X, n_neighbors):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int64)
    for i in numba.prange(knn_indices.shape[0]):
        knn_indices[i] = X[i].argsort(kind="mergesort")[:n_neighbors]
    return knn_indices


@numba.njit(parallel=True)
def parallel_argsort(X):
    index_array = np.empty(X.shape, dtype=np.int64)
    for i in numba.prange(index_array.shape[0]):
        index_array[i] = X[i].argsort()
    return index_array


@numba.njit(parallel=True)
def parallel_argsort_quicksort(X):
    index_array = np.empty(X.shape, dtype=np.int64)
    for i in numba.prange(index_array.shape[0]):
        index_array[i] = X[i].argsort(kind="quicksort")
    return index_array


@numba.njit(parallel=True)
def parallel_argsort_mergesort(X):
    index_array = np.empty(X.shape, dtype=np.int64)
    for i in numba.prange(index_array.shape[0]):
        index_array[i] = X[i].argsort(kind="mergsort")
    return index_array


@numba.njit(parallel=True)
def parallel_sort_by_argsort(X, argsort_indices):
    sorted_array = np.empty(X.shape, dtype=X.dtype)
    for i in numba.prange(sorted_array.shape[0]):
        sorted_array[i] = X[i][argsort_indices[i]]
    return sorted_array
