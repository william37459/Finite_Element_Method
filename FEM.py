import ufl
import dolfinx
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
import pyvista as pv
import dolfinx.fem.petsc
from tabulate import tabulate
 
def u_ex(mod):
    return lambda x: mod.exp(x[0]+x[1])*mod.cos(x[0])*mod.sin(x[1])+x[0]

u_ufl = u_ex(ufl)
u_numpy = u_ex(np)

def solver(N=10, degree=2):
    ## Define domain and functionspace over it
    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1, -1), (1, 1)),
        n=(N, N))
    V = fem.functionspace(domain, ("Lagrange", degree))

    ## Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    ## Source term of the poisson equation
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(u_ufl(x)))
    
    ## Applying boundary conditions
    uD      = fem.Function(V)
    uD.interpolate(lambda x: np.exp(x[0]+x[1])*np.cos(x[0])*np.sin(x[1])+x[0])
    tdim    = domain.topology.dim
    fdim    = tdim-1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)
    
    # bilinear form
    a       = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L       = f * v * ufl.dx
    
    # set PETSc solver options
    sol_opts = {"ksp_type": "preonly", "pc_type": "lu"}
    # formulate the problem
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=sol_opts)
    return problem.solve(), u_ufl(x), domain, V, tdim

def error_L2(uh, u_ex, degree_raise=3):
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = fem.FunctionSpace(mesh, (family, degree + degree_raise))
    u_W = fem.Function(W)
    u_W.interpolate(uh)
    u_ex_W = fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def error_H10(uh, u_ex, degree_raise=3):
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = fem.FunctionSpace(mesh, (family, degree + degree_raise))
    u_W = fem.Function(W)
    u_W.interpolate(uh)
    u_ex_W = fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
    error_H10 = fem.form(ufl.dot(ufl.grad(e_W), ufl.grad(e_W)) * ufl.dx + ufl.dot(e_W,e_W) * ufl.dx)
    E_H10 = np.sqrt(mesh.comm.allreduce(fem.assemble_scalar(error_H10), MPI.SUM))
    return E_H10


def convergence_rate(error, Ns = [4, 8, 16, 32, 64], deg=1):
    result = []
    Es = np.zeros(len(Ns), dtype=default_scalar_type)
    hs = np.zeros(len(Ns), dtype=np.float64)
    for i, N in enumerate(Ns):
        uh, u_ex, domain, V, tdim = solver(N, deg)
        comm = uh.function_space.mesh.comm
        Es[i] = f"{error(uh, u_numpy, degree_raise=3):.2e}"
        hs[i] = f"{(1. / Ns[i]):.2e}"
    return [list(hs), list(Es)]

def tabulate_convergence_rate(error, start=1, end=11): 
    for i in range(start, end):
        if(i == start):
            data = convergence_rate(error=error,deg=i)
            data[0].insert(0, " ")
            data[1].insert(0, f"{i}")
        else:
            data.append(convergence_rate(error=error, deg=i)[1])
            data[i].insert(0, f"{i}")
    print(tabulate(data, tablefmt='latex_raw'))

#tabulate_convergence_rate(error_H10)