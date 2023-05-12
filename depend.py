
import numpy as np
from numpy.linalg import norm

import scipy as sp
from scipy import inf
from scipy.helper import savemat
from scipy.sparse import kron, eye
import scipy.sparse.linalg

from polyhedron_tools.misc import polyhedron_to_Hrep, chebyshev_center, radius


from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens
from sage.modules.free_module_element import vector
from sage.functions.other import real_part, imag_part
from sage.functions.log import log, exp


def kron_prod(x, y):
    return [x[i] * y[j] for i in range(len(x)) for j in range(len(y))]


def get_key_from_index(i, j, n):
    x = polygens(QQ, ['x' + str(1 + k) for k in range(n)])
    x_power_j = kron_power(x, j)
    d = x_power_j[i].dict()
    return list(d.keys())[0]


def get_index_from_key(key, j, n):
    x = polygens(QQ, ['x' + str(1 + k) for k in range(n)])
    x_power_j = kron_power(x, j)
    for i, monomial in enumerate(x_power_j):
        if list(monomial.dict().keys())[0] == key:
            first_occurence = i
            break

    return first_occurence


def kron_power(x, i):
    if i > 2:
        return kron_prod(x, kron_power(x, i - 1))
    elif i == 2:
        return kron_prod(x, x)
    elif i == 1:
        return x
    else:
        raise ValueError('index i should be an integer >= 1')


def lift(x0, N):
    y0 = kron_power(x0, 1)
    for i in range(2, N + 1):
        y0 = [a + b for a, b in zip(y0, kron_power(x0, i))]
    return y0



def log_norm(A, p='inf'):
    # parse the input matrix
    if isinstance(A, scipy.sparse.spmatrix):
        # cast into numpy array (or ndarray)
        A = A.toarray()
        n = A.shape[0]
    elif isinstance(A, np.ndarray):
        n = A.shape[0]
    else:
        # assuming sage matrix
        n = A.nrows()

    # computation, depending on the chosen norm p
    if p == 'inf' or p == np.inf:
        z = max(np.real(A[i][i]) + np.sum(np.abs(A[i][j]) for j in range(n)) - np.abs(A[i][i]) for i in range(n))
        return z

    elif p == 1:
        return max(np.real(A[j][j]) + np.sum(np.abs(A[i][j]) for i in range(n)) - np.abs(A[j][j]) for j in range(n))

    elif p == 2:
        if not (A.base_ring() == RR or A.base_ring() == CC):
            return 1/2 * max((A + A.H).eigenvalues())
        else:
            z = 1/2 * max(np.linalg.eigvals(np.matrix(A + A.H, dtype=complex)))
            return np.real(z) if np.imag(z) == 0 else z

    else:
        raise NotImplementedError('value of p not understood or not implemented')

