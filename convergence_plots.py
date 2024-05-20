#import ufl
#import dolfinx
#from dolfinx import mesh, fem, plot, default_scalar_type
#from mpi4py import MPI
import pyvista as pv
#import dolfinx.fem.petsc
import FEM
import numpy as np
import random
from matplotlib import pyplot as plt

def tabulate_convergence_rate(start=1, end=4): 
    chart = pv.Chart2D()
    # for error in [FEM.error_H10, FEM.error_L2 ]:
    for i in range(start, end):
        h_and_error = FEM.convergence_rate(FEM.error_H10,deg=i)
        # chart.scatter(np.log(h_and_error[0]), np.log(h_and_error[1]), style="o", color=[random.randrange(1, 255), random.randrange(1, 255), random.randrange(1, 255)], label=f"Degree {i}")
        #find line of best fit
        x = np.log(h_and_error[0])
        y = np.log(h_and_error[1])
        a, b = np.polyfit(x, y, 1)

        color = [random.randrange(1, 255), random.randrange(1, 255), random.randrange(1, 255)]

        styles = [ "x","+","s","o","d"]

        chart.line([np.log(0.0156), np.log(0.25)], [a*np.log(0.0156)+b, a*np.log(0.25)+b], color=color)
        chart.scatter(x, y, style=styles[i % len(styles)], color=color, label=f"Degree {i}")
        
        # Increase tick label size
        chart.x_axis.tick_label_size = 20
        chart.y_axis.tick_label_size = 20
        
        # Increase axis label size
        chart.x_axis.label_size = 20
        chart.y_axis.label_size = 20
        
        # Set marker size
        chart.marker_size = 20

tabulate_convergence_rate()