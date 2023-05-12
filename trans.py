import numpy as np
from numpy.linalg import norm
from sage.symbolic.ring import SR
from sage.plot.graphics import Graphics
from sage.plot.plot import plot
from sage.plot.line import line
import scipy as sp
from scipy import inf
from scipy.helper import savemat
from scipy.sparse import kron, eye
import scipy.sparse.linalg

from sage.modules.free_module_element import vector

from carlin.helper import get_Fj_from_model

from carlin.solution import PolynomialODE

from polyhedron_tools.misc import polyhedron_to_Hrep, chebyshev_center, radius

from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens

from sage.modules.free_module_element import vector

from sage.functions.other import real_part, imag_part
from sage.functions.log import log, exp

from carlin.depend import *

from scipy.sparse import bmat, lil_matrix


def transfer_matrices(N, F, n, k):
    A = []

    # First row is trivial
    A.append(F)

    for i in range(1, N):
        newRow = []
        for j in range(k):
            L = kron(A[i-1][j], eye(n))
            R = kron(eye(n ** i), F[j])
            newRow.append(np.add(L, R))
        A.append(newRow)

    return A

def truncated_matrix(N, *args, **kwargs):
    if 'input_format' not in kwargs:
        input_format = 'model_filename'
    else:
        input_format = kwargs['input_format']

    if input_format == 'model_filename':
        model_filename = args[0]
        F, n, k = get_Fj_from_model(model_filename)
        A = transfer_matrices(N, F, n, k)
    elif input_format == 'transfer_matrices':
        A = args[0]
        n = args[1]
        k = args[2]
    elif input_format == 'Fj_matrices':
        F = args[0]
        n = args[1]
        k = args[2]
        A = transfer_matrices(N, F, n, k)
    else:
        raise ValueError('Invalid input format')

    AN_list = []

    for i in range(N):
        n3_i = max(N - i - k, 0)
        n2_i = N - i - n3_i

        newBlockRow = A[i][:n2_i]

        for _ in range(i):
            newBlockRow.insert(0, None)

        for _ in range(n3_i):
            newBlockRow.append(None)

        AN_list.append(newBlockRow)

    AN = bmat(AN_list)

    return AN



def quadratic_reduction(F, n, k):
    A = transfer_matrices(k-1, F, n, k)

    # LINEAR PART
    F1_tilde_list = []
    for i in range(k-1):
        newRow = A[i][:k-i-1]
        for _ in range(i):
            newRow.insert(0, None)
        F1_tilde_list.append(newRow)

    F1_tilde = bmat(F1_tilde_list)

    # QUADRATIC PART
    F2_tilde_list = []

    for i in range(k-1):
        newRow = []
        for h in range(k-1):

            for j in range(k-2):
                newRow.append(lil_matrix((n**(i+1), n**(h+j+2))))

            if h > i:
                newRow.append(lil_matrix((n**(i+1), n**(h+k))))
            else:
                newRow.append(A[i][k-i-1+h])

        F2_tilde_list.append(newRow)

    F2_tilde = bmat(F2_tilde_list)

    F_tilde = [F1_tilde, F2_tilde]

    kquad = 2

    nquad = F1_tilde.shape[0]

    return [F_tilde, nquad, kquad]




def error_function(model, N, x0):
    if isinstance(model, str):
        [F, n, k] = get_Fj_from_model(model)
    elif isinstance(model, PolynomialODE):
        [F, n, k] = get_Fj_from_model(model.funcs(), model.dim(), model.degree())

    [Fquad, nquad, kquad] = quadratic_reduction(F, n, k)

    ch = characteristics(Fquad, nquad, kquad)

    norm_F1_tilde, norm_F2_tilde = ch['norm_Fi_inf']

    x0_hat = [kron_power(x0, i+1) for i in range(k-1)]
    x0_hat = [item for sublist in x0_hat for item in sublist]

    norm_x0_hat = norm(x0_hat, ord=np.inf)
    beta0 = ch['beta0_const'] * norm_x0_hat
    Ts = 1/norm_F1_tilde * np.log(1 + 1/beta0)

    t = SR.var('t')
    error = norm_x0_hat * np.exp(norm_F1_tilde * t) / (1 + beta0 - beta0 * np.exp(norm_F1_tilde * t)) * (beta0 * (np.exp(norm_F1_tilde * t) - 1)) ** N
    return [Ts, error]

def plot_error_function(model_filename, N, x0, Tfrac=0.8):
    [Ts, eps] = error_function(model_filename, N, x0)
    P = Graphics()
    P += plot(eps, 0, Ts * Tfrac, axes_labels=["$t$", r"$\mathcal{E}(t)$"])
    P += line([[Ts, 0], [Ts, eps(t=Ts * Tfrac)]], linestyle='dotted', color='black')
    return P

def linearize(model, target_filename, N, x0, **kwargs):
    dic = dict()
    dic['model_name'] = model
    dic['N'] = N

    print('Obtaining the canonical representation...'),
    if isinstance(model, str):
        [F, n, k] = get_Fj_from_model(model)
    elif isinstance(model, PolynomialODE):
        [F, n, k] = get_Fj_from_model(model.funcs(), model.dim(), model.degree())
    print('done')

    dic['n'] = n
    dic['k'] = k

    print('Computing matrix AN...'),
    if isinstance(model, str):
        A_N = truncated_matrix(N, model)
    elif isinstance(model, PolynomialODE):
        A_N = truncated_matrix(N, F, n, k, input_format="Fj_matrices")
    print('done')

    dic['AN'] = A_N

    print('Computing the quadratic reduction...'),
    [Fquad, nquad, kquad] = quadratic_reduction(F, n, k)
    print('done')

    print('Computing the characteristics of the model...'),
    ch = characteristics(Fquad, nquad, kquad)
    print('done')

    norm_F1_tilde = ch['norm_Fi_inf'][0]
    norm_F2_tilde = ch['norm_Fi_inf'][1]

    dic['norm_F1_tilde'] = norm_F1_tilde
    dic['norm_F2_tilde'] = norm_F2_tilde

    dic['log_norm_F1_inf'] = ch['log_norm_F1_inf']

    if 'polyhedron' in str(type(x0)):
        from polyhedron_tools.misc import radius, polyhedron_to_Hrep
        [Fx0, gx0] = polyhedron_to_Hrep(x0)
        norm_initial_states = radius([Fx0, gx0])
        if norm_initial_states >= 1:
            norm_x0_hat = norm_initial_states**(k-1)
        elif norm_initial_states < 1:
            norm_x0_hat = norm_initial_states

        dic['norm_x0_tilde'] = norm_x0_hat

        beta0 = ch['beta0_const'] * norm_x0_hat
        dic['beta0'] = beta0

        Ts = 1 / norm_F1_tilde * np.log(1 + 1 / beta0)
        dic['Ts'] = Ts

        [F, g] = polyhedron_to_Hrep(x0)
        dic['x0'] = {'F': F, 'g': g.column()}

        cheby_center_X0 = chebyshev_center(x0)
        dic['x0_cc'] = vector(cheby_center_X0).column()

    else:  # assuming that x0 is a list object
        nx0 = np.linalg.norm(x0, ord=np.inf)
        norm_x0_hat = max([nx0 ** i for i in range(1, k)])

        dic['norm_x0_tilde'] = norm_x0_hat

        beta0 = ch['beta0_const'] * norm_x0_hat
        dic['beta0'] = beta0

        Ts = 1 / norm_F1_tilde * np.log(1 + 1 / beta0)
        dic['Ts'] = Ts

        dic['x0'] = vector(x0).column()

    if 'append' in kwargs.keys():
        extra_data = kwargs['append']

        if not isinstance(extra_data, list):
            dkey = extra_data.pop('name')
            dic[dkey] = extra_data
        else:
            for new_data in extra_data:
                new_key = new_data.pop('name')
                dic[new_key] = new_data

    print('Exporting to', target_filename, '...'),
    savemat(target_filename, dic)
    print('done')

    return

def characteristics(F, n, k, ord=np.inf):
    c = dict()

    c['norm_Fi_inf'] = [np.linalg.norm(F[i].toarray(), ord=ord) for i in range(k)]

    if ord == np.inf:
        c['log_norm_F1_inf'] = log_norm(F[0], p='inf')
    else:
        raise NotImplementedError("log norm error should be supremum (='inf')")

    if k > 1:
        if c['norm_Fi_inf'][0] != 0:
            c['beta0_const'] = c['norm_Fi_inf'][1] / c['norm_Fi_inf'][0]
        else:
            c['beta0_const'] = np.inf

    return c
