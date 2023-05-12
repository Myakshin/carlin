from sage.rings.integer import Integer
from sage.rings.all import RR, QQ
from sage.rings.real_double import RDF
from sage.rings.polynomial.polynomial_ring import polygens
from carlin.solution import PolynomialODE

def vanderpol(mu=1, omega=1):
    # dimension of state-space
    n = 2

    # define the vector of symbolic variables
    var_names = ['x' + str(i) for i in range(n)]
    x = polygens(QQ, var_names)

    # vector field (n-dimensional)
    f = [None] * n

    f[0] = x[1]
    f[1] = - omega ** 2 * x[0] + mu * (1 - x[0] ** 2) * x[1]

    # the order is k=3
    return PolynomialODE(f, n, k=3)