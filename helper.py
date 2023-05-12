import numpy as np

import scipy as sp

from scipy import inf

import scipy.sparse.linalg

from scipy.sparse import dok_matrix


from scipy.sparse.linalg import expm_multiply
from sage.rings.integer import Integer
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens 
from sage.modules.free_module_element import vector
from carlin.trans import truncated_matrix
from scipy.sparse.linalg import expm_multiply

from carlin.trans import get_index_from_key
from carlin.utils import kron_power, lift

from scipy.sparse import coo_matrix
from sage.plot.plot import list_plot
from sage.structure.sage_object import load

def load_model(model_filename):
    model_data = load(model_filename)
    f, n = model_data

    k = max([fi.degree() for fi in f])

    return [f, n, k]



def get_Fj_from_model(model_filename=None, f=None, n=None, k=None):
    if model_filename is not None and f is None:
        got_model_by_filename = True
    elif model_filename is not None and f is not None and n is not None and k is None:
        k = n
        n = f
        f = model_filename
        got_model_by_filename = False
    else:
        raise ValueError("either the model name or the vector field (f, n, k) should be specified")

    if got_model_by_filename:
        [f, n, k] = load_model(model_filename)

    # create the collection of sparse matrices Fj
    F = [dok_matrix((n, n ** i), dtype=np.float64) for i in range(1, k + 1)]

    # read the powers appearing in each monomial
    try:
        dictionary_f = [fi.dict() for fi in f]

    except AttributeError:
        # if the polynomial has symbolic coefficients, we have to parse the
        # expression tree
        raise NotImplementedError("the coefficients of the polynomial should be numeric")

    for i, dictionary_f_i in enumerate(dictionary_f):
        for key in dictionary_f_i:
            row = i
            j = sum(key)
            column = get_index_from_key(list(key), j, n)
            F[j - 1][row, column] = dictionary_f_i.get(key)

    return F, n, k

def export_model_to_mat(model_filename, F=None, n=None, k=None, **kwargs):
    if '.sage' in model_filename:
        mat_model_filename = model_filename.replace(".sage", ".mat")
    elif '.mat' in model_filename:
        mat_model_filename = model_filename
    else:
        raise ValueError("Expected .sage or .mat file format in model filename.")

    if F is None:
        F, n, k = get_Fj_from_model(model_filename=model_filename)

    savemat(mat_model_filename, {'F': F, 'n': n, 'k': k})
    return


def solve_ode_exp(AN, x0, N, tini=0, T=1, NPOINTS=100):
    # transform to [x0, x0^[2], ..., x0^[N]]
    y0 = np.array([x0 ** (i + 1) for i in range(N)])

    # compute solution
    if isinstance(AN, sage.matrix.matrix.Matrix):
        t_dom = np.linspace(tini, T, num=NPOINTS)
        sol = [AN.exp() * np.exp(ti) * y0 for ti in t_dom]

    elif isinstance(AN, scipy.sparse.spmatrix):
        t_dom = np.linspace(tini, T, num=NPOINTS)
        sol = expm_multiply(AN, y0, start=tini, stop=T, num=NPOINTS, endpoint=True)

    else:
        raise ValueError("invalid matrix type")

    return sol


def plot_truncated(model, N, x0, tini, T, NPOINTS, xcoord=0, ycoord=1, **kwargs):


    f, n, k = model.funcs(), model.dim(), model.degree()
    F, _, _ = get_Fj_from_model(f=f, n=n, k=k)

    # Create the truncated matrix AN in COO format
    rows = []
    cols = []
    data = []

    for j in range(N):
        for i in range(N):
            rows.append(i)
            cols.append(j)
            data.append(F[j][i, j])

    AN = coo_matrix((data, (rows, cols)), shape=(N, N))

    # Solve the linear ODE using SciPy's sparse matrix solver
    t_dom = np.linspace(tini, T, num=NPOINTS)
    sol = expm_multiply(AN, x0, start=tini, stop=T, num=NPOINTS, endpoint=True)

    sol_x1 = sol[:, xcoord]
    sol_x2 = sol[:, ycoord]
    sol_zip = list(zip(sol_x1, sol_x2))
    
    return list_plot(sol_zip, plotjoined=True, **kwargs)
