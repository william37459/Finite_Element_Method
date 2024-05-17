#import ufl
#import dolfinx
#from dolfinx import mesh, fem, plot, default_scalar_type
#from mpi4py import MPI
import pyvista as pv
#import dolfinx.fem.petsc
import FEM
import matplotlib.pyplot as plt
import numpy as np


h_and_error = FEM.convergence_rate()
print(h_and_error[0], h_and_error[1])
print(np.log(h_and_error[0]), np.log(h_and_error[1]))

chart = pv.Chart2D()
chart.scatter(np.log(h_and_error[0]), np.log(h_and_error[1]), style="o")
chart.show()
