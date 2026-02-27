import numpy as np 
from MatricialFunction import CreateM, CreateSystem  
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
approx_order = 2    # Precision of approximation (= No_qp for exact solution)


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
ERR[-1] = 0.0

# CREATION OF THE MATRICIAL SYSTEM 
Mx = CreateM(Bx, Px)
My = CreateM(By, Py)
hatK = np.diag(ERR)


#-------------------------------------
#   CREATE THE OBSERVABLES
#-------------------------------------

zx, z0x, Dzx = simulate_z(Bx, Px, ERR, p = 1.0)
zy, z0y, Dzy = simulate_z(By, Py, ERR, p = -1.0)

# ADD NOISE
zx_measured = zx * (1.0 + np.random.normal(0.0, Noise_level / 100.0, len(zx)))
zy_measured = zy * (1.0 + np.random.normal(0.0, Noise_level / 100.0, len(zy)))

# IMPORTANT: well simulate that we get the \Delta z from simply substracting the measure with the perfect orbit
Dzx_measured = zx_measured - z0x
Dzy_measured = zy_measured - z0y

#-------------------------------------
#       SOLVE THE PROBLEM
#-------------------------------------

Dz = np.concatenate((Dzx_measured, Dzy_measured))
z = np.concatenate((zx_measured, zy_measured))


Ax = Mx @ np.diag(zx_measured)
Ay = -1.0 * My @ np.diag(zy_measured)

A = np.vstack((Ax, Ay))

# # solve
# y, residuals, rank, s = np.linalg.lstsq(A, Dz, rcond=None)


# Here we apply SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"SVD singular values = {S}")
tol = 50
S_inv = np.zeros_like(S)
S_inv[S > tol] = 1.0 / S[S > tol]

y = Vt.T @ np.diag(S_inv) @ U.T @ Dz


print("solution")
print(y)

print("REAL ERRORS")
print(ERR)

print("\n\n", "_"*30, "\n CORROBORATION \n", "_"*30)

zx_fit, _, _= simulate_z(Bx, Px, y, p = 1.0)
zy_fit, _, _ = simulate_z(By, Py, y, p = -1.0)
