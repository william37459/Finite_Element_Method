import ufl
import dolfinx
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
import pyvista as pv
import dolfinx.fem.petsc
 
 
## Define domain and functionspace over it
NElem   = 16
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(NElem, NElem)
)
V = fem.functionspace(domain, ("Lagrange", 2))

## Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
 
#E Source term of the poisson equation
x = ufl.SpatialCoordinate(domain)
f = ufl.exp(x[0]+x[1])*ufl.classes.Cos(x[0])*ufl.classes.Sin(x[1]*x[0])+x[1]
 
## Applying boundary conditions
uD      = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
tdim    = domain.topology.dim
fdim    = tdim-1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)
bc      = fem.dirichletbc(uD, boundary_dofs)
 
# bilinear form
a       = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L       = f * v * ufl.dx
 
# set PETSc solver options
sol_opts = {"ksp_type": "preonly", "pc_type": "lu"}
# formulate the problem
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=sol_opts)
# solve the problem
uh = problem.solve()
 
 
# 2D contour plots of the mesh and result using pyvista
pv.off_screen = True
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
 
plotter = pv.Plotter()
plotter.add_mesh(grid,show_edges=True)
plotter.view_xy()
#plotter.save_graphic('mesh.pdf')
plotter.show()
 
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pv.Plotter()
u_plotter.add_mesh(u_grid,show_edges=True)
u_plotter.view_xy()
#u_plotter.save_graphic('contour.pdf')
u_plotter.show()

warped = u_grid.warp_by_scalar()
plotter2 = pv.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
plotter2.show()


