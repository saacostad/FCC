import numpy as np 

def simulate_z0(B, P, J = 1.0, d0 = 0.0):
    """
    This function simulates a perfectr orbit of the accelerator according to 

    z(s) = \sqrt(2 J_0 \beta(s)) \sin(\psi(s) - \delta_0)
    """

    return np.sqrt(2.0 * J * B) * np.sin( P - d0 )

def simulate_Dz(B, P, Err, z, Bs, Ps, p = 1.0):
    """
    FUNCTION PARAMETERS:
    B -> The beta function \beta(s_i) for i<j [np.ndarray]
    P -> The psi function \psi(s_i) for i<j [np.ndarray]
    Err -> The errors \Delta K for i<j [np.ndarray]
    z -> The already calculated positions for i<j [np.ndarray]
    Bs, Ps -> The \beta(s) and \psi(s) functions on the point to calculate
    p -> The sign of the error (+1.0 for x axis, -1.0 for y axis)
    -----------------------------------------------------------------------------------------------------
    This function simulates the \Delta z component of a measurement given the lattice functions 
    and the already calculated z positions tweaked because of the previos quadrupole errors contributions

    \Delta z(s_j) = p*\sum_{i<j} \Delta K_i z(s_i) \sqrt(\beta(s_j) \beta(s_i)) \sin(\psi(s_j) - \psi(s_i))
    -----------------------------------------------------------------------------------------------------
    """

    return p*np.sum(np.array([Err * z * np.sqrt(B * Bs) * np.sin(Ps - P)]))

def simulate_z(B, P, Err, p=1.0, J = 1.0, d0 = 0.0):
    """
    FUNCTION PARAMETERS
    B -> \beta functions on the quadrupoles
    P -> \psi functions on the quadrupoles 
    Err -> \Delta K on the quadrupoles 
    p -> Axis constant (+1.0 for x axis / -1.0 for y axis)
    J and d0 -> Action and phase constants
    --------------------------------------------------------------------------------------------------------
    This function simulates the z measurements for a chain of N magnetic quadrupoles errors
    given by 

    z(s_j) = z_0(s_j) + \Delta z(s_j)

    for j<N 
    ---------------------------------------------------------------------------------------------------------
    Returns a tuple with the z and \Delta z measurements
    """
    
    # We first simulate the perfect measurements 
    z0 = simulate_z0(B, P, J, d0)

    # Our first measurement hasn't feel any quadrupole error kick
    z = [z0[0]]
    Dz = [0.0]

    for i in range(1, len(B)):

        Dz_s = simulate_Dz(B[:i], P[:i], Err[:i], z, B[i], P[i], p)

        z.append( z0[i] + Dz_s)
        Dz.append( Dz_s )
    
    return (z, Dz)

    


