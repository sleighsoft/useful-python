# Requires:
# Python, numpy, (scikit-learn [not actually need, just for validation])
#
# Can be installed with:
# pip install numpy scikit-learn
#
# Description:
# Implements pairwise euclidean distance computation.
#   pairwise_euclidean_distance()
# Also implements two types of batched pairwise euclidean distance functions.
#   batched_pairwise_euclidean_distance()
#   batched_pairwise_euclidean_distance_generator()

import numpy as np
from math import ceil
from sklearn.metrics.pairwise import euclidean_distances


r = np.random.uniform(size=[5, 3])
r2 = np.random.uniform(size=[10, 3])


def row_norm(X):
    return np.sum(np.square(X), axis=1)


def pairwise_euclidean_distance(
    X, Y, row_norms_X=None, row_norms_Y=None, squared=False, out=None
):
    if row_norms_X is None:
        row_norms_X = row_norm(X)
        row_norms_X = np.reshape(row_norms_X, [-1, 1])

    if row_norms_Y is None:
        row_norms_Y = row_norm(Y)
        row_norms_Y = np.reshape(row_norms_Y, [1, -1])

    d = -2 * np.dot(X, Y.T)

    if out is not None:
        out[:] = d
        d = out

    d += row_norms_X
    d += row_norms_Y

    np.maximum(d, 0, out=d)

    if not squared:
        np.sqrt(d, out=d)

    if X is Y:
        np.fill_diagonal(d, 0)
    return d, row_norms_X, row_norms_Y


def batched_pairwise_euclidean_distance(X, Y=None, batch_size=None):
    assert X.dtype == np.float64
    if Y is None:
        Y = X
    if X is not Y:
        assert X.dtype == Y.dtype
        assert X.shape[1] == Y.shape[1]

    if batch_size is None:
        batch_size = X.shape[0]

    d = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)

    for i in range(ceil(X.shape[0] / batch_size)):
        s = i * batch_size
        e = (i + 1) * batch_size

        pairwise_euclidean_distance(X[s:e], Y, out=d[s:e, :])

    return d


def batched_pairwise_euclidean_distance_generator(X, Y=None, batch_size=None):
    assert X.dtype == np.float64
    if Y is None:
        Y = X
    if X is not Y:
        assert X.dtype == Y.dtype
        assert X.shape[1] == Y.shape[1]

    if batch_size is None:
        batch_size = X.shape[0]

    for i in range(ceil(X.shape[0] / batch_size)):
        s = i * batch_size
        e = (i + 1) * batch_size

        yield pairwise_euclidean_distance(X[s:e], Y)[0]


sklearn = euclidean_distances(r, r2)

b1 = batched_pairwise_euclidean_distance(r, r2, r.shape[0])

print(np.isclose(b1, sklearn).all())

gen = batched_pairwise_euclidean_distance_generator(r, r2, r.shape[0] // 2)
b2 = np.vstack([d for d in gen])

print(np.isclose(b2, sklearn).all())
