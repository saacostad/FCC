import numpy as np
from scipy.optimize import minimize
import random

N = 4       # Number of variables in the system

def choose_lambda(a0_star, x_prior, noise_level):
    """Adaptive lambda selection"""
    data_scale = np.var(a0_star)
    prior_scale = np.var(x_prior) if np.var(x_prior) > 0 else 1.0
    noise_scale = (noise_level * np.linalg.norm(a0_star))**2
    
    # Balance terms: data_term ~ lambda * prior_term
    lambda_adaptive = noise_scale / (prior_scale + 1e-8)
    return np.clip(lambda_adaptive, 1e-3, 1e3)




def objective(x, a_observed, x_prior, lambda_reg = 1.0):
    """ Objective function for the minimization: 
    vec hat x = argmin_{vec x}( | F(vec x) - a_0^* |^2 ) + lambda | vec x - vec x_{prior} | """

    data_term = np.sum((F(x) - a_observed)**2)      # Actual minimization problem
    prior_term = np.sum(abs(x - x_prior))           # Add some information to my problem (Order of magnitude of \vec x)

    return data_term + lambda_reg * prior_term




def multiple_restart_minimization(a0_star, x_prior, lambda_reg, n_restarts=10):
    best_x = None
    best_obj = np.inf
    
    for _ in range(n_restarts):
        # Try different initial guesses around reasonable range
        if best_x is None:
            # First try: use prior as initial guess
            x0 = x_prior.copy()
        else:
            # Subsequent tries: random perturbations
            x0 = np.random.uniform(-10, 10, size=N)
        
        # Add bounds to keep solution reasonable
        bounds = [(-20, 20) for _ in range(N)]
        
        result = minimize(objective, x0, args=(a0_star, x_prior, lambda_reg), 
                         method='L-BFGS-B', bounds=bounds)
        
        if result.success and result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    
    return best_x, best_obj



def F(x):
    """System of equations"""
    return np.array([
        coefficients[0,0]*x[0] + coefficients[0,1]*x[1]*x[0] + coefficients[0,2]*x[2]*x[0] + coefficients[0,3]*x[3]*x[0] + coefficients[0,4],
        coefficients[1,0]*x[1] + coefficients[1,1]*x[0]*x[1] + coefficients[1,2]*x[2]*x[1] + coefficients[1,3]*x[3]*x[1] + coefficients[1,4],
        coefficients[2,0]*x[2] + coefficients[2,1]*x[1]*x[2] + coefficients[2,2]*x[0]*x[2] + coefficients[2,3]*x[3]*x[2] + coefficients[2,4],
        coefficients[3,0]*x[3] + coefficients[3,1]*x[1]*x[3] + coefficients[3,2]*x[2]*x[3] + coefficients[3,3]*x[3]*x[0] + coefficients[3,4]
    ])



for i in range(5):

    coefficients = np.random.uniform(-10, 10, size=(N, N+1))

    x_real = np.random.uniform(low=-1.0, high = 1.0, size=N)
    x_prior = np.random.uniform(low = -1.0, high = 1.0, size = N)

    x0 = np.zeros(N)

    noise_level = np.random.uniform(0.01, 0.10)
    noise = np.random.normal(0.0, noise_level * np.abs(F(x_real)))
    a0_star = F(x_real) + noise                                    # Actual output vector measured


    noise_est = a0_star * 0.05                                  # Asuming noise is 5% of the output    
    lambda_reg = choose_lambda(a0_star, x_prior, noise_level)   # Lambda has information of both the noise levels on the system, 
                                                                # and the order of magnitude in x_real


    # Use multiple restarts
    x_estimated, final_obj = multiple_restart_minimization(a0_star, x_prior, lambda_reg)


    print("\n================================================")
    print("Added noise: ", noise / a0_star * 100, "\n")
    print("Stimated x: ", x_estimated)
    print("Real x: ", x_real)
    print("\nCorrectness: ", abs(x_estimated - x_real))
    print("Residual: ", a0_star - F(x_estimated))
    print("Res + Noise: ", a0_star - F(x_estimated) - noise)
    print("================================================\n")
