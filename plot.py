
import ufl
import dolfinx
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
import pyvista as pv
import dolfinx.fem.petsc
import FEM

uh, u_ex, domain, V, tdim = FEM.solver()


pv.off_screen = True
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
 
plotter = pv.Plotter()
plotter.add_mesh(grid,show_edges=True)
plotter.view_xy()
#plotter.save_graphic('mesh.svg')
 
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pv.Plotter()
u_plotter.add_mesh(u_grid,show_edges=True)
u_plotter.view_xy()
#u_plotter.save_graphic('contour.svg')

warped = u_grid.warp_by_scalar()
plotter2 = pv.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
plotter2.show(cpos="xy", screenshot='./screenshot.png')