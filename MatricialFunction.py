import numpy as np 


def NeummanInverse(T, grad = 8):
    """
    This function creates the Neuman Truncated Inverse of

    (1 - p\hat M \hat {\Delta K})

    according to the formula

    (1 - p\hat M \hat {\Delta K})^{-1} = sum_{k = 1}^{grad} (p \hat M \hat \Delta K)^k 
    """
    n = T.shape[0]
    I = np.eye(n, dtype=T.dtype)
    
    S = I.copy()
    Tk = I.copy()
    
    for _ in range(1, grad):
        Tk = Tk @ T
        S += Tk
        
    return S


def CreateQ(B, P, K, p, grad):
    """
    This function creates the Q matrix given that 

    \hat Q = \hat{\Delta K} ( 1 - p \hat{M} \hat{\Delta K} )^{-1}

    with 

    \hat{\Delta K} = \Diag(\Delta K_i)
    M_{ij} = \sin(\psi_i - \psi_j) \sqrt(\beta_j \beta_i) for j < i else 0

    where grad is the "order of the inverse" of the term in parenthesis, given by truncating 
    the Neumman series of p \hat{M} \hat{\Delta K} when 

    ( 1 - p \hat{M} \hat{\Delta K} )^{-1} = \sum_{k = 1}^{grad} (p \hat{M} \hat{\Delta K} )^k
    """

    # Matrix form of \Delta K
    hatK = np.diag(K)
    

    # We create the matrix M 
    #-------------------------------

    n = len(P)  # Number of Quadrupoles
    
    hatM = np.zeros((n, n))    # Create a square matrix first

    i, j = np.tril_indices(n, k=-1) # Take only the indexes of the lower triangular section
    
    hatM[i, j] = np.sin(P[i] - P[j]) * np.sqrt(B[i] * B[j])    # Fill the gaps with what we need 


    # Now we create the whole matrix in parenthesis and we invert it 
    #---------------------------------------------------------------------------
    
    if grad == 8:
        Q = hatK @ np.linalg.inv(np.eye(n) - p * hatM @ hatK)
    else:
        print("calculating inverse with Neumman")
        Q = hatK @ NeummanInverse(p * hatM @ hatK, grad)

    return Q

    

def CreateSystem(B, P, K, p = 1, grad = 8):
    """
    This function takes the system's \beta's, \phi's and quadrupole magnetic errors \Delta K
    and returns a tuple with the matrix \hat Q, and the vectors \vec u and \vec v following that

    \vec u = {\sqrt(\beta_i) \sin(\psi_i)}
    \vec v = {\sqrt(\beta_i) \cos(\psi_i)}

    \hat Q = \hat{\Delta K} ( 1 - p \hat{M} \hat{\Delta K} )^{-1}
    """

    u = np.sqrt(B) * np.sin(P)
    v = np.sqrt(B) * np.cos(P)

    Q = CreateQ(B, P, K, p, grad)

    return Q, u, v


def CreateConstants(Q, u, v):
    """
    This function takes the system vectors/matrices and calculates the coefficients 
    of the expansion we care about. Returns a tuple with 

    \vec v^T \hat Q \vec u
    \vec v^T \hat Q \vec v
    \vec v^u \hat Q \vec u
    \vec v^u \hat Q \vec v
    """

    return(v @ Q @ u, v @ Q @ v, u @ Q @ u, u @ Q @ v)
