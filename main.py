import numpy as np 
from MatricialFunction import CreateM, CreateSystem, CreateConstants, createFirstOrderMatrix 
from Optimizer import *

""" SYSTEM PARAMETERS ADJUSTMENT
In this section we'll adjust the main system's parameters, that is:

    - magnitud of the lattice functions
    - noise level to simulate 
    - number of quadrupoles to simulate 
    - grade of the solver 
    - other methods
"""

# GENERAL SYSTEM PARAMETERS 
No_qp = 4          # Number of quadrupoles 
approx_order = 5    # Precision of approximation (= No_qp for exact solution)


# BETA FUNCTION PARAMETERS
Bx_magnitud = 500.0          # Magnitud of the beta function 
Bx_sigma = 200.0             # Sigma to simulate our beta's in a normal distribution

By_magnitud = 500.0          
By_sigma = 200.0            


# PHI FUNCTION PARAMETERS 
Px_initial = np.random.normal(np.pi/2.0, 1.0)        # Magnitud of the initial PHI  -> different values of Phi shall be similar
Px_sigma = 0.05                                  # Sigma value to model Phi

Py_initial = np.random.normal(np.pi/2.0, 1.0)        
Py_sigma = 0.05 


# SIMULATED ERRORS 
ERR_sigma = 5.0e-5      # Magnitud of the errors to simulate (sigma of a normal distribution)


# SIMULATED NOISE   
Noise_level = 0.0         # Noise level in percentage


# INITIAL GUESS FOR OPTIMIZER
x0 = np.zeros(No_qp)


# NORMALIZATION FACTOR
nf = 1e5                    # A scale applied to our errors 


# ORIGINAL LATICE PARAMETERS 
J0 = 1.0 
delta0 = 0.0 


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
# ERR = ERR_ns * nf



# CREATION OF THE MATRICIAL SYSTEM
# IMPORTANT: we create the system considering the exact solution, as it is closer to what we'll measure

# This ones if we want closer to real world measurements
Qx, ux, vx = CreateSystem(Bx, Px, ERR, p=1.0, grad = No_qp)      # The p value expresses the nature of the error, that is
Qy, uy, vy = CreateSystem(By, Py, ERR, p=-1.0, grad = No_qp)     # in which plane the error is present 

# # This ones to check linearity
# Qx, ux, vx = CreateSystem(Bx, Px, ERR, p=1.0, grad = approx_order)      # The p value expresses the nature of the error, that is
# Qy, uy, vy = CreateSystem(By, Py, ERR, p=-1.0, grad = approx_order)     # in which plane the error is present 

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




# PRINTING OF THE RESULTS


# PRINTING OF SYSTEM DATA
print("SYSTEM SIMULATION\n", "_"*30,"\n")


print("Real errors")
print(ERR)
print() 

print("Real constants generated")
print(np.concatenate([realSolution_x, realSolution_y]))
print()

print("Noisy constants generated")
print(np.concatenate([noisySolution_x, noisySolution_y]))
print()

print("Level of noise")
noise = 1.0 - np.abs(np.concatenate([noisySolution_x, noisySolution_y]) / np.concatenate([realSolution_x, realSolution_y]))
print(noise, " = ", np.mean(np.abs(noise)))

print()

M = createFirstOrderMatrix(ux, vx, uy, vy)

print(Qx / ERR)

print("_"*30, "\n", "SYSTEM RESULTS\n", "_"*30,"\n")

# K_reconstructed = solve_with_relative_noise(M, noisySolution, relative_noise=0.05)
K_reconstructed = solve_simple(M, noisySolution)

print(f"True K:          {ERR}")
print(f"Reconstructed K: {K_reconstructed}")
print(f"Condition number: {np.linalg.cond(M)}")
print()
print(f"Relative error:  {np.linalg.norm(K_reconstructed - ERR) / np.linalg.norm(ERR):.2%}")
print(f"Relative error:  {((K_reconstructed - ERR) / ERR)*100}")


print("\n"*2)

F_Qx, F_ux, F_vx = CreateSystem(Bx, Px, K_reconstructed, p =  1.0, grad=approx_order) 
F_Qy, F_uy, F_vy = CreateSystem(By, Py, K_reconstructed, p =  -1.0, grad=approx_order) 

F_Cx = CreateConstants(F_Qx, F_ux, F_vx)
F_Cy = CreateConstants(F_Qy, F_uy, F_vy)
print("Fitted constants generated")
print(np.concatenate([F_Cx, F_Cy]))
print("Real constants generated")
print(np.concatenate([realSolution_x, realSolution_y]))
print("Noisy constants generated")
print(np.concatenate([noisySolution_x, noisySolution_y]))





print("_"*30, "\n", "SYSTEM RESULTS TRUNCATING SVD SPACE\n", "_"*30,"\n")

M_better, k = truncate_matrix(M, max_condition=100)
K_reconstructed = solve_simple(M_better, noisySolution)

print()
print(f"True K:          {ERR}")
print(f"Reconstructed K: {K_reconstructed}")
print()
print(f"Relative error:  {np.linalg.norm(K_reconstructed - ERR) / np.linalg.norm(ERR):.2%}")
print(f"Relative error %:  {((K_reconstructed - ERR) / ERR)*100}")
print()
print()

F_Qx, F_ux, F_vx = CreateSystem(Bx, Px, K_reconstructed, p =  1.0, grad=approx_order) 
F_Qy, F_uy, F_vy = CreateSystem(By, Py, K_reconstructed, p =  -1.0, grad=approx_order) 

F_Cx = CreateConstants(F_Qx, F_ux, F_vx)
F_Cy = CreateConstants(F_Qy, F_uy, F_vy)
print("Fitted constants generated")
print(np.concatenate([F_Cx, F_Cy]))
print("Real constants generated")
print(np.concatenate([realSolution_x, realSolution_y]))
print("Noisy constants generated")
print(np.concatenate([noisySolution_x, noisySolution_y]))


