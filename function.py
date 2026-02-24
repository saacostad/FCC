import numpy as np 
from MatricialFunction import CreateSystem, CreateConstants

def general_equation(Err, B, P, f1 = np.sin, f2 = np.cos, _n = 8):
    """ 

    This function constains the general structure of the equations of the SOE. 

    Variables
    ------------------------------------------------------------------------------------------------
    Err: the quadrupole errors to simulate 
    B and P: respectevily, are the \beta and \psi values of the quadrupoles 
    f1: the first trigonometric function to model
    f2: the second trignometric function to model 
    n: the order of the equation (default = 8)
    ------------------------------------------------------------------------------------------------
    In general, the equation returned is the following (taking indexes starting from 1)

    $$          \sum_{i=1}^{n} Err[i] * B[i] f1(P[i]) * f2([p1])  
                + \sum_{i=2}^{n} \sum{j=1}^{i-1} f1(P[i]) * f2(P[i]) 
                * \prod_{k=j}^{i} [ Err[k] B[k] ] * \prod_{m=j}^{i-1} [ np.sin(P[m+1] - P[m]) ] $$

    """

    linear_term = np.sum( np.array([ Err[i] * B[i] * f1(P[i]) * f2(P[i]) for i in range(_n)]) )

    non_linear_term = 0.0
    
    if _n > 1: 
        for i in range(1, _n):
            for j in range(i):

                first_product = np.prod( np.array([ Err[k] * B[k] for k in range(j, i + 1)]) )
                second_product = np.prod( np.array([ np.sin( P[m+1] - P[m] ) for m in range(j, i) ]) )


                non_linear_term += first_product * second_product * ( f1(P[j]) * f2(P[i]) )

        return linear_term + non_linear_term
    
    return linear_term



def general_equation_ALT(Err, B, P, f1 = np.sin, f2 = np.cos, _n = 8):
    
    value = 0.0 

    for i in range(_n):

        value += Err[i] * B[i] * f1(P[i]) * f2(P[i])
        
        if i != 0:

            non_linear_term = 0.0 

            for j in range(i):
                non_linear_term += f1(P[j]) * np.sin(P[i] - P[j]) * Err[i] * B[i]

            value += f2(P[i]) * Err[i] * B[i] * non_linear_term
    
    return value


def system_of_equations_ALT(ERR, B, P, n = 8):

    s = np.sin 
    c = np.cos 
    

    return np.array([
            general_equation_ALT(ERR, B, P, f1 = s, f2 = c, _n = n),
            general_equation_ALT(ERR, B, P, f1 = s, f2 = s, _n = n),
            general_equation_ALT(ERR, B, P, f1 = c, f2 = c, _n = n),
            general_equation_ALT(ERR, B, P, f1 = c, f2 = s, _n = n)
        ])

def system_of_equations(ERR, B, P, n = 8):
    """
    This fuction creates 4 entries of the system of equations following 
    eq1: general_equation(..., f1 = np.sin, f2 = np.cos)
    eq2: general_equation(..., f1 = np.sin, f2 = np.sin)
    eq3: general_equation(..., f1 = np.cos, f2 = np.cos)
    eq4: general_equation(..., f1 = np.cos, f2 = np.sin)
    ----------------------------------------------------------------------

    Variables
    ----------------------------------------------------------------------
    Err: the quadrupole errors to simulate 
    B and P: respectevily, are the \beta and \psi values of the quadrupoles 
    n: the order of the equation (default = 8)
    """
    s = np.sin 
    c = np.cos 
    

    return np.array([
            general_equation(ERR, B, P, f1 = s, f2 = c, _n = n),
            general_equation(ERR, B, P, f1 = s, f2 = s, _n = n),
            general_equation(ERR, B, P, f1 = c, f2 = c, _n = n),
            general_equation(ERR, B, P, f1 = c, f2 = s, _n = n)
        ])




BETA = np.abs(np.random.normal(300.0, 100.0, 8))
PHI = np.random.normal( np.random.normal(np.pi, 1.0), 0.05, 8)
ERR = np.random.normal( 0.0, 5.0e-5, 8 )



print(f"BETA: {BETA}")
print(f"PHI: {PHI}")
print(f"ERR: {ERR}")


print()
print("n=1 OR")
print(system_of_equations(ERR, BETA, PHI, n=1))
print("n=1 ALT")
print(system_of_equations_ALT(ERR, BETA, PHI, n=1))
print("n=1 MATRICIAL")
Q, u, v = CreateSystem(BETA, PHI, ERR, grad = 1)
print(CreateConstants(Q, u, v))

print()
print("n=2 OR")
print(system_of_equations(ERR, BETA, PHI, n=2))
print("n=2 ALT")
print(system_of_equations_ALT(ERR, BETA, PHI, n=2))
Q, u, v = CreateSystem(BETA, PHI, ERR, grad = 2)
print("n=2 MATRICIAL")
print(CreateConstants(Q, u, v))

print()
print("n=8 OR")
print(system_of_equations(ERR, BETA, PHI, n=8))
print("n=8 ALT")
print(system_of_equations_ALT(ERR, BETA, PHI, n=8))
Q, u, v = CreateSystem(BETA, PHI, ERR)
print("n=8 MATRICIAL")
print(CreateConstants(Q, u, v))
