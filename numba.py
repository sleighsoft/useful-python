# Requires:
# Python, numpy, numba
#
# Can be installed with:
# pip install numpy numba
#
# Description:
# Short snippets for numpy functions parallelized with numba.
#
# Note:Be careful not to use parallelized functions within functions that
#      already have @numba.njit(parallel=True)!

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


@numba.njit(parallel=True)
def parallel_take_along_axis(X, indices):
    """Takes indices along axis=-1 from X. Only works on 2-D arrays.

    Both X and indices must be 2-D.

    Example: Can be used on argsort return value
    """
    taken = np.empty(indices.shape, dtype=X.dtype)
    for i in numba.prange(taken.shape[0]):
        row_indices = indices[i]
        row_X = X[i]
        for j in range(taken.shape[1]):
            taken[i][j] = row_X[row_indices[j]]
    return taken


@numba.njit(parallel=True)
def parallel_put_by_advanced_index(X, advanced_indices, values):
    """Takes a 2-tuple advanced index (x,y) and a 1-D array of values and puts
    these into the appropriate positions.
    """
    x_indices, y_indices = advanced_indices
    for i in numba.prange(x_indices.shape[0]):
        x = x_indices[i]
        y = y_indices[i]
        X[x, y] = values[i]
    return X


@numba.njit(parallel=True)
def parallel_insert(X, indices, value):
    """Insert `value` at all `indices` into X. This will shift all elements 
    starting at `indice` to the right. Elements whose index is > X.shape[1] will
    be dropped. X.shape[0] == indices.shape[0]
    """
    new = np.empty(X.shape, dtype=X.dtype)
    for i in numba.prange(new.shape[0]):
        pos = indices[i]
        new_row = new[i]
        row = X[i]
        if pos < X.shape[1]:
            new_row[:pos] = row[:pos]
            if pos != 0:
                new_row[pos] = value
                new_row[pos + 1 :] = row[pos + 1 :]
            else:
                new_row[pos + 1 :] = row[pos:-1]
                new_row[pos] = value
        else:
            new[i] = row
    return new
