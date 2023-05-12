
from carlin.helper import solve_ode_exp
from carlin.trans import get_Fj_from_model, truncated_matrix

from sage.plot.graphics import Graphics
from sage.plot.plot import plot
from sage.plot.plot import list_plot

def solve_time_triggered(AN, N, x0, tini, T, NRESETS, NPOINTS):
    tdom = srange(tini, T, (T-tini)/(NPOINTS-1)*1., include_endpoint=True)

    CHUNK_SIZE = int(NPOINTS/(NRESETS+1))

    n = AN.shape[0]

    sol_tot = []
    tdom_k = tdom[:CHUNK_SIZE]
    x0_k = x0
    sol_chunk_k = solve_ode_exp(AN, x0=x0_k, N=N, tini=tdom_k[0], T=tdom_k[-1], NPOINTS=CHUNK_SIZE)
    sol_tot.append(sol_chunk_k[:, :n+1])

    for i in range(1, NRESETS+1):
        tdom_k = tdom[CHUNK_SIZE*i-1:CHUNK_SIZE*(i+1)]
        x0_k = list(sol_chunk_k[-1, :n+1])
        sol_chunk_k = solve_ode_exp(AN, x0=x0_k, N=N, tini=0, T=tdom_k[-1]-tdom_k[0], NPOINTS=CHUNK_SIZE+1)
        sol_tot.append(sol_chunk_k[:, :n+1])
    return sol_tot


def plot_time_triggered(model, N, x0, tini, T, NRESETS, NPOINTS, j, **kwargs):
    G = Graphics()

    color = kwargs.get('color', 'blue')

    Fj = get_Fj_from_model(model.funcs(), model.dim(), model.degree())
    AN = truncated_matrix(N, *Fj, input_format="Fj_matrices")
    solution = solve_time_triggered(AN, N, x0, tini, T, NRESETS, NPOINTS)

    CHUNK_SIZE = int(NPOINTS/(NRESETS+1))

    tdom = srange(tini, T, (T-tini)/(NPOINTS-1)*1., include_endpoint=True)
    tdom_k = tdom[:CHUNK_SIZE]

    G += list_plot(zip(tdom_k, solution[0][:, j]), plotjoined=True, linestyle="dashed", color=color, legend_label="$N="+str(N)+", r="+str(NRESETS)+"$")

    for i in range(1, NRESETS+1):
        G += point((tdom_k[0], x0_k[1]), size=25, marker='x', color=color)

        G += list_plot(zip(tdom_k, solution[i][:, j]), plotjoined=True, linestyle="dashed", color=color)

    S = model.solve(x0=x0, tini=tini, T=T, NPOINTS=NPOINTS)
    x_t = S.interpolate_solution(j)
    G += plot(x_t, tini, T, axes_labels=["$t$", "$x_{"+str(j)+"}$"], gridlines=True, color="black")

    return G
