import numpy as np 
from MatricialFunction import CreateM, CreateSystem, CreateConstants, CreateQ 
from scipy.optimize import least_squares
from SystemSim import simulate_z, simulate_z0

""" SYSTEM PARAMETERS ADJUSTMENT
In this section we'll adjust the main system's parameters, that is:

    - magnitud of the lattice functions
    - noise level to simulate 
    - number of quadrupoles to simulate 
    - grade of the solver 
    - other methods
"""

# GENERAL SYSTEM PARAMETERS 
No_qp = 8          # Number of quadrupoles 
approx_order = 8    # Precision of approximation (= No_qp for exact solution)


# BETA FUNCTION PARAMETERS
Bx_magnitud = 500.0          # Magnitud of the beta function 
Bx_sigma = 200.0             # Sigma to simulate our beta's in a normal distribution

By_magnitud = 500.0          
By_sigma = 200.0            

# PHI FUNCTION PARAMETERS 
Px_initial = np.random.normal(np.pi, 1.0)        # Magnitud of the initial PHI  -> different values of Phi shall be similar
Px_sigma = 0.05                                  # Sigma value to model Phi

Py_initial = np.random.normal(np.pi, 1.0)        
Py_sigma = 0.05 

# SIMULATED ERRORS 
ERR_sigma = 5.0e-5      # Magnitud of the errors to simulate (sigma of a normal distribution)

# SIMULATED NOISE   
Noise_level = 5.0         # Noise level in percentage

# INITIAL GUESS FOR OPTIMIZER
x0 = np.zeros(No_qp)

""" SIMULATING OUR SYSTEM
In this step, we'll simulate the quadrupole chain (that is, the beta's, phi's and errors we'll place to each quadrupole)
"""

# -----------------------------
#   CREATION OF THE SYSTEM 
# -----------------------------

# For the x axis
Bx = np.abs(np.random.normal(Bx_magnitud, Bx_sigma, No_qp))
Px = np.random.normal(Px_initial, Px_sigma, No_qp)   

# For the y axis
By = np.abs(np.random.normal(By_magnitud, By_sigma, No_qp))
Py = np.random.normal(Py_initial, Py_sigma, No_qp)  

# Error simulation
ERR = np.random.normal(0.0, ERR_sigma, No_qp)

# CREATION OF THE MATRICIAL SYSTEM 
Qx, ux, vx = CreateSystem(Bx, Px, ERR, p=1.0, grad = No_qp)      # The p value expresses the nature of the error, that is
Qy, uy, vy = CreateSystem(By, Py, ERR, p=-1.0, grad = No_qp)     # in which plane the error is present 
Mx = CreateM(Bx, Px)
My = CreateM(By, Py)

# We create the real constant 
realSolution_x = CreateConstants(Qx, ux, vx)
realSolution_y = CreateConstants(Qy, uy, vy)

# And now we create the noisy variables we'll work with
noisySolution_x = realSolution_x * (1.0 + np.random.normal(0.0, Noise_level / 100.0, len(realSolution_x)))
noisySolution_y = realSolution_y * (1.0 + np.random.normal(0.0, Noise_level / 100.0, len(realSolution_y)))

# We put these vectors together 
noisySolution = np.concatenate([noisySolution_x, noisySolution_y])


""" OPTIMIZER
"""


# def residual(errors):
#     _Qx = CreateQ(Mx, errors, p = 1.0, grad = approx_order)
#     _Qy = CreateQ(My, errors, p = -1.0, grad = approx_order)
#
#     solution = np.concatenate([CreateConstants(_Qx, ux, vx), CreateConstants(_Qy, uy, vy)])
#
#     return solution - noisySolution
#
#
# result = least_squares(
#     residual,  # residual function
#     x0,
#     method='lm',             # Levenberg-Marquardt
#     jac='2-point',           # Numerical Jacobian (finite differences)
#     ftol=1e-8,               # tolerance for residual
#     xtol=1e-8,               # tolerance for x
#     gtol=1e-8,               # tolerance for gradient
#     max_nfev=1000            # max iterations
# )



""" PRINTING OF THE RESULTS
# """

# # PRINTING OF SYSTEM DATA
# print("SYSTEM SIMULATION\n", "_"*30,"\n")
#
# print("Real errors")
# print(ERR)
# print() 
#
# print("Real constants generated")
# print(np.concatenate([realSolution_x, realSolution_y]))
# print()
#
# print("Noisy constants generated")
# print(np.concatenate([noisySolution_x, noisySolution_y]))
# print()-1
#
# print("Level of noise")
# noise = 1.0 - np.abs(np.concatenate([noisySolution_x, noisySolution_y]) / np.concatenate([realSolution_x, realSolution_y]))
# print(noise, " = ", np.mean(noise))
#
# print()
#
#
#
# print("_"*30, "\n", "SYSTEM RESULTS\n", "_"*30,"\n")
#
# F_ERR = result.x 
#
# F_Qx, F_ux, F_vx = CreateSystem(Bx, Px, F_ERR, p =  1.0, grad=approx_order) 
# F_Qy, F_uy, F_vy = CreateSystem(By, Py, F_ERR, p =  -1.0, grad=approx_order) 
#
# F_Cx = CreateConstants(F_Qx, F_ux, F_vx)
# F_Cy = CreateConstants(F_Qy, F_uy, F_vy)
#
# print("Fitted errors")
# print(result.x)
# print("Real errors")
# print(ERR)
# print()
#
# print("Fitted constants generated")
# print(np.concatenate([F_Cx, F_Cy]))
# print("Real constants generated")
# print(np.concatenate([realSolution_x, realSolution_y]))
# print("Noisy constants generated")
# print(np.concatenate([noisySolution_x, noisySolution_y]))


""" CHECKING MATRICIAL FORMULATION
"""

# z0x = simulate_z0(Bx, Px) 
# zx_measured, Dzx = simulate_z(Bx, Px, ERR, p=-1.0)
#
# print("CLASSICAL FORMULATION \n", "_"*30, "\n")
# print("\Delta z")
# print(Dzx)
#
# print()
#
#
# Dzx_mat = linalg.inv( np.eye(No_qp) + 1.0 * Mx @ np.diag(ERR) ) @ z0x - z0x 
# print("MATRICIAL FORMULATION \n", "_"*30, "\n")
# print("\Delta z")
# print(Dzx_mat)

