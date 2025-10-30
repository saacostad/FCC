import numpy as np
from scipy.optimize import minimize

# Set parameters
N = 4
np.random.seed(42)

# Generate fixed random coefficients for F
coefficients = np.random.uniform(-10, 10, size=(N, N+1))

def F(x):
    """System of equations with FIXED coefficients"""
    return np.array([
        coefficients[0,0]*x[0] + coefficients[0,1]*x[1]*x[0] + coefficients[0,2]*x[2]*x[0] + coefficients[0,3]*x[3]*x[0] + coefficients[0,4],
        coefficients[1,0]*x[1] + coefficients[1,1]*x[0]*x[1] + coefficients[1,2]*x[2]*x[1] + coefficients[1,3]*x[3]*x[1] + coefficients[1,4],
        coefficients[2,0]*x[2] + coefficients[2,1]*x[1]*x[2] + coefficients[2,2]*x[0]*x[2] + coefficients[2,3]*x[3]*x[2] + coefficients[2,4],
        coefficients[3,0]*x[3] + coefficients[3,1]*x[1]*x[3] + coefficients[3,2]*x[2]*x[3] + coefficients[3,3]*x[3]*x[0] + coefficients[3,4]
    ])

def compute_jacobian(F, x, epsilon=1e-8):
    """Numerical Jacobian computation"""
    n = len(x)
    m = len(F(x))
    J = np.zeros((m, n))
    
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += epsilon
        J[:, i] = (F(x_perturbed) - F(x)) / epsilon
        
    return J

def analyze_system_condition(F, x):
    """Use SVD to analyze system conditioning at point x"""
    J = compute_jacobian(F, x)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    
    condition_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
    effective_rank = np.sum(s > s[0] * 1e-6)
    
    analysis = {
        'condition_number': condition_number,
        'singular_values': s,
        'rank': np.sum(s > 1e-10),
        'effective_rank': effective_rank,
        'ill_conditioned': condition_number > 1e6,
        'jacobian': J
    }
    
    return analysis, U, s, Vt

def identify_well_determined_parameters(F, x_prior, n_samples=50):
    """Use SVD to identify which parameters are well-determined"""
    Jacobians = []
    
    # Sample Jacobians around prior
    for _ in range(n_samples):
        x_perturbed = x_prior + np.random.normal(0, 0.1, len(x_prior))
        try:
            J = compute_jacobian(F, x_perturbed)
            Jacobians.append(J)
        except:
            continue
    
    if len(Jacobians) == 0:
        # Fallback: use prior point only
        J = compute_jacobian(F, x_prior)
        Jacobians = [J]
    
    # Average Jacobian
    J_avg = np.mean(Jacobians, axis=0)
    
    # SVD analysis
    U, s, Vt = np.linalg.svd(J_avg, full_matrices=False)
    
    # Parameter importance (how much each parameter affects output)
    # Use the right singular vectors weighted by singular values
    effective_rank = np.sum(s > s[0] * 1e-6)
    if effective_rank > 0:
        parameter_importance = np.sum((Vt[:effective_rank].T * s[:effective_rank])**2, axis=1)
    else:
        parameter_importance = np.ones(len(x_prior))
    
    # Normalize importance
    parameter_importance = parameter_importance / np.max(parameter_importance)
    
    return parameter_importance, Vt

def choose_lambda_svd(F, x_prior, a0_star, noise_level):
    """Lambda selection using SVD analysis"""
    try:
        # Compute Jacobian at prior
        J = compute_jacobian(F, x_prior)
        
        # SVD analysis
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        condition_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
        
        # Base lambda from noise and prior scales
        noise_scale = (noise_level * np.linalg.norm(a0_star))**2
        prior_scale = np.sum(x_prior**2) / len(x_prior)  # Average squared magnitude
        
        lambda_base = noise_scale / (prior_scale + 1e-8)
        
        # Adjust lambda based on condition number
        if condition_number > 1e6:
            lambda_adj = lambda_base * min(np.sqrt(condition_number) / 1e3, 100)
        elif condition_number < 10:
            lambda_adj = lambda_base * 0.1  # Less regularization for well-conditioned
        else:
            lambda_adj = lambda_base
            
        return np.clip(lambda_adj, 1e-4, 1e2), condition_number
        
    except:
        # Fallback if SVD fails
        noise_scale = (noise_level * np.linalg.norm(a0_star))**2
        prior_scale = np.sum(x_prior**2) / len(x_prior)
        lambda_base = noise_scale / (prior_scale + 1e-8)
        return np.clip(lambda_base, 1e-3, 1e1), 1.0

def objective_with_svd_weights(x, a_observed, x_prior, lambda_reg, param_importance):
    """Objective function with SVD-based parameter weighting"""
    data_term = np.sum((F(x) - a_observed)**2)
    
    # Weight prior term by parameter importance
    # Important parameters get less regularization (smaller weight in prior term)
    weighted_prior_term = np.sum((1.0 / (param_importance + 1e-8)) * (x - x_prior)**2)
    
    return data_term + lambda_reg * weighted_prior_term

def multiple_restart_optimization(F, a0_star, x_prior, lambda_reg, param_importance, n_restarts=15):
    """Multiple restart optimization with SVD-informed initialization"""
    best_x = None
    best_obj = np.inf
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    for restart in range(n_restarts):
        if restart == 0:
            # First try: use prior
            x0 = x_prior.copy()
        elif restart == 1:
            # Second try: zeros
            x0 = np.zeros(N)
        elif restart < 8:
            # Next tries: small perturbations around prior
            x0 = x_prior + np.random.normal(0, 0.2, N)
            x0 = np.clip(x0, -1.0, 1.0)
        else:
            # Final tries: completely random in [-1, 1]
            x0 = np.random.uniform(-1.0, 1.0, N)
        
        try:
            result = minimize(
                objective_with_svd_weights, 
                x0, 
                args=(a0_star, x_prior, lambda_reg, param_importance),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success and result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except:
            continue
    
    return best_x, best_obj

# Main execution loop
print("SVD-Enhanced Parameter Estimation")
print("=" * 60)
print(f"System dimension: {N}")
print(f"Fixed coefficients shape: {coefficients.shape}")
print()

for i in range(5):
    print(f"\nTrial {i+1}:")
    print("-" * 40)
    
    # Generate true parameters and prior in [-1, 1] range
    x_real = np.random.uniform(low=-1.0, high=1.0, size=N)
    x_prior = np.random.uniform(low=-1.0, high=1.0, size=N)
    
    # Generate noisy observations
    noise_level = np.random.uniform(0.01, 0.10)
    a0_real = F(x_real)
    noise = np.random.normal(0.0, noise_level * np.abs(a0_real))
    a0_star = a0_real + noise
   
    known_noise_level = np.ones(N) * 0.05

    # SVD-based analysis and lambda selection
    lambda_reg, condition_number = choose_lambda_svd(F, x_prior, a0_star, 0.05)
    param_importance, Vt = identify_well_determined_parameters(F, x_prior)
    
    print(f"True x:        {x_real}")
    print(f"Prior x:       {x_prior}")
    print(f"Noise level:   {noise_level*100:.1f}%")
    print(f"Condition num: {condition_number:.2e}")
    print(f"Lambda:        {lambda_reg:.6f}")
    print(f"Param importance: {param_importance}")
    
    # Multiple restart optimization
    x_estimated, final_obj = multiple_restart_optimization(F, a0_star, x_prior, lambda_reg, param_importance)
    
    if x_estimated is not None:
        # Final SVD analysis at solution
        analysis_final, U_final, s_final, Vt_final = analyze_system_condition(F, x_estimated)
        
        error = np.linalg.norm(x_estimated - x_real)
        relative_error = error / (np.linalg.norm(x_real) + 1e-8)
        
        print(f"Estimated x:   {x_estimated}")
        print(f"Error norm:    {error:.6f}")
        print(f"Relative error: {relative_error*100:.2f}%")
        print(f"Final objective: {final_obj:.6f}")
        print(f"Final condition: {analysis_final['condition_number']:.2e}")
        
        # Check if any estimated parameters are outside desired range
        if np.any(np.abs(x_estimated) > 1.0):
            print("*** WARNING: Solution outside [-1, 1] range ***")
        
        # Check for parameter swapping
        from itertools import permutations
        min_perm_error = np.inf
        best_perm = None
        for perm in permutations(range(N)):
            perm_error = np.linalg.norm(x_estimated - x_real[list(perm)])
            if perm_error < min_perm_error:
                min_perm_error = perm_error
                best_perm = list(perm)
        
        if min_perm_error < error * 0.8:  # If permutation gives much better error
            print(f"*** Possible parameter swapping detected ***")
            print(f"Best permutation {best_perm} gives error: {min_perm_error:.6f}")
            
    else:
        print("Optimization failed for all restarts")
    
    print()

# Display system information
print("\n" + "=" * 60)
print("SYSTEM ANALYSIS SUMMARY")
print("=" * 60)
print("Fixed coefficients matrix:")
print(coefficients)

# Test the system conditioning at a typical point
test_point = np.random.uniform(-1, 1, N)
analysis_test, U_test, s_test, Vt_test = analyze_system_condition(F, test_point)
print(f"\nSystem conditioning at random point:")
print(f"Singular values: {s_test}")
print(f"Condition number: {analysis_test['condition_number']:.2e}")
print(f"Effective rank: {analysis_test['effective_rank']}/{N}")
print(f"Ill-conditioned: {analysis_test['ill_conditioned']}")
