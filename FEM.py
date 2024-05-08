import ufl
import dolfinx
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
import pyvista as pv
import dolfinx.fem.petsc
 
def u_ex(mod):
    return lambda x: mod.exp(x[0]+x[1])*mod.cos(x[0])*mod.sin(x[1])+x[0]

u_ufl = u_ex(ufl)
u_numpy = u_ex(np)

def solver(N=10, degree=2):
    ## Define domain and functionspace over it
    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0, -1.0), (1.0, 1.0)),
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
    uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
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

uh, u_ex, domain, V, tdim = solver()
 
#Ns = [4, 8, 16, 32, 64]
#Es = np.zeros(len(Ns), dtype=default_scalar_type)
#hs = np.zeros(len(Ns), dtype=np.float64)
#for i, N in enumerate(Ns):
#    uh, u_ex = solve_poisson(N, degree=1)
#    comm = uh.function_space.mesh.comm
#    # One can send in either u_numpy or u_ex
    # For L2 error estimations it is reccommended to send in u_numpy
    # as no JIT compilation is required
#    Es[i] = error_L2(uh, u_numpy)
#    hs[i] = 1. / Ns[i]
#    if comm.rank == 0:
#        print(f"h: {hs[i]:.2e} Error: {Es[i]:.2e}")

 