from sage.structure.sage_object import SageObject
from sage.plot.plot import list_plot
from sage.calculus.ode import ode_solver
from sage.misc.misc_c import prod

from sage.structure.sage_object import SageObject

class PolynomialODE(SageObject):
    def __init__(self, funcs=None, dim=None, degree=None):
        self._funcs = funcs
        self._dim = dim
        self._degree = degree


    def __repr__(self):
        return "A Polynomial ODE in n = {} variables".format(self._dim)


    def funcs(self):
        return self._funcs

    def dim(self):
        return self._dim

    def degree(self):
        return self._degree
    deg=degree


    def solve(self, x0=None, tini=0, T=1, NPOINTS=100):
        if x0 is None:
            raise ValueError("To solve, you should specify the initial condition")

        S = ode_solver()

        def funcs(t, x):
            f = []
            if self._dim == 1:
                fid = self._funcs[0].dict()
                f = sum(fid[fiex] * x[0] ** fiex for fiex in fid.keys())
                return [f]
            elif self._dim > 1:
                for i, fi in enumerate(self._funcs):
                    fid = fi.dict()
                    row_i = sum(fid[fiex] * prod(x[i] ** ai for i, ai in enumerate(fiex)) for fiex in fid.keys())
                    f.append(row_i)
                return f

        S.function = funcs

        def jac(t, x):
            n = self._dim
            # Jacobian
            dfi_dyj = [[self._funcs[i].derivative(x[j]) for j in range(n)] for i in range(n)]

            # We consider autonomous systems
            dfi_dt = [0] * n

            return dfi_dyj + dfi_dt

        S.jacobian = jac

        # Choose integration algorithm and solve
        S.algorithm = "rk4"
        S.ode_solve(y_0=x0, t_span=[tini, T], num_points=NPOINTS)
        return S

def plot_solution(self, x0=None, tini=0, T=1, NPOINTS=100, xcoord=0, ycoord=1, plotjoined=True, **kwargs):
        S = self.solve(x0=x0, tini=tini, T=T, NPOINTS=NPOINTS)
        sol_xy = [(S_ti[1][xcoord], S_ti[1][ycoord]) for S_ti in S.solution]
        return list_plot(sol_xy, plotjoined=plotjoined, **kwargs)