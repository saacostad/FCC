from scipy.optimize import least_squares

def F(x):  # Your polynomial operator: R^n -> R^m
    # Example: F(x) = [x[0]**2 + x[1], x[0]*x[1] - 1, ...]
    return your_polynomial_function(x)

# Noisy data
a_star = ...  # your noisy measurement vector

# Initial guess
x0 = ...  # e.g., np.zeros(n) or random

# Solve
result = least_squares(
    lambda x: F(x) - a_star,  # residual function
    x0,
    method='lm',             # Levenberg-Marquardt
    jac='2-point',           # Numerical Jacobian (finite differences)
    ftol=1e-8,               # tolerance for residual
    xtol=1e-8,               # tolerance for x
    gtol=1e-8,               # tolerance for gradient
    max_nfev=1000            # max iterations
)

x_est = result.x


# Esto es para normalizar tanto \vec x como \vec a, sirve para el LS    
result = least_squares(..., x_scale='jac')  # or provide array of scales
