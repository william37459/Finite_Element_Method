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
#def u_ex(mod):
#    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])

u_ufl = u_ex(ufl)
u_numpy = u_ex(np)

def solver(N=10, degree=2):
    ## Define domain and functionspace over it
    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-10, -10), (10, 10)),
        n=(N, N)
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    ## Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    ## Source term of the poisson equation
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(u_ufl(x)))
    
    ## Applying boundary conditions
    uD      = fem.Function(V)
    #uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    uD.interpolate(lambda x: np.exp(x[0]+x[1])*np.cos(x[0])*np.sin(x[1])+x[0])
    tdim    = domain.topology.dim
    fdim    = tdim-1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
    #bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)
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
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = fem.FunctionSpace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def convergence_rate(Ns = [4, 8, 16, 32, 64], deg=1):
    result = []
    Es = np.zeros(len(Ns), dtype=default_scalar_type)
    hs = np.zeros(len(Ns), dtype=np.float64)
    for i, N in enumerate(Ns):
        uh, u_ex, domain, V, tdim = solver(N, deg)
        comm = uh.function_space.mesh.comm
        Es[i] = error_L2(uh, u_numpy)
        hs[i] = 1. / Ns[i]
        result.append([f"{hs[i]:.2e}", f"{Es[i]:.2e}"])
    return result

def tabulate_convergence_rate(start=1, end=11): 
    for i in range(start, end):
        data = convergence_rate(deg=i)
        print(tabulate(data, tablefmt='latex_raw'))

tabulate_convergence_rate()