"""
This script takes the best quadrupoles on each one of the interaction regions and performs
the error corrections accoring to the calculated theory (or something like that)
"""
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

from math import cos, sin
from numba import njit, prange
import numpy as np
import pandas as pd
from scipy.optimize import root
from tfs import constants


from QPphysicalParam import (
    getQP as QPphysicalParam,
)  # From here, we get a pandas DataFrame with the physical parameters of the quadrupoles
from QPSelector import (
    MainFunction as selQP,
)  # From here, we et the best quadrupoles for each one of the Interaction Regions



# We import the datframes from the other scripts
physParams = QPphysicalParam()

regs = [0, 1, 3, 5]

selectedQP = selQP().iloc[
    regs,
]  # As the quadrupoles are repeated in the datafiles, we only select the first 4


"""
        GLOBAL PARAMETERS
"""


# If the system of equations, aftereach iteration, don't change in more than
# this quantity, then we're satisfied
TRESHOLD = 0.8e-5


# This is the error we're gonna try to recreate
ERRs = np.array(list(range(1, 7))) * 10 ** (-5)


"""
    DATA MODIFICATION
In this part, we modify the original quadrupoles dataframes SO they contain the new column KL, which will be used
in order to calculate the magnetic errors of the quadrupoles
"""


def createRowKL(df):
    """This function takes a pandas dataFrame df, creates a new column KL and, according to the name of its
    quadrupoles, appends the value of the magnetic kick of the quadrupoles"""

    df["KL"] = np.nan  # We create the new row in the dataframe

    for index, row in df.iterrows():
        qpName = row["NAME"].split(".")[0]
        df.at[index, "KL"] = physParams.loc[physParams["NAME"] == qpName, "KL"].values[
            0
        ] * 10 ** (-4)


for index, row in selectedQP.iterrows():
    createRowKL(selectedQP.loc[index, "leftX"])
    createRowKL(selectedQP.loc[index, "leftY"])
    createRowKL(selectedQP.loc[index, "rightX"])
    createRowKL(selectedQP.loc[index, "rightY"])


"""
    MATRIX CREATION
In this section, all the functions defining the matrices and vectors are created.
The First Order Matrix M is the one that satisfies M b = e, that is, the one we use 
to solve the linear system of equations. The Second Oreder Vector is the vector that 
has the information of all the Non-linear (up to the second term) terms of the equation.
"""

""" APPROACH USING SIMPLER EQUATIONS WITH EVERYTHING INCLUDED """

N = 6


def getBet(qps, plane):
    return np.array(qps[f"BET{plane}"])

def getMu(qps, plane):
    return np.array(qps[f"MU{plane}"])


@njit   
def equation(errs, betas, mus, eqNo = 1):

    sum = 0.0

    match eqNo:
        case 1: 
            sectrig = lambda x,y: np.sin(x) * np.cos(y)
        case 2: 
            sectrig = lambda x,y: np.sin(x) * np.sin(y)
        case 3: 
            sectrig = lambda x,y: np.cos(x) * np.cos(y)
        case 4: 
            sectrig = lambda x,y: np.cos(x) * np.sin(y)


    for i in range(N):
        sum += errs[i] * betas[i] * sectrig(mus[i], mus[i])


    for i in range(1, N):
        for j in range(0, i-1):

            sot = sectrig(mus[j], mus[i])
            
            for k in range(j, i+1):
                sot *= errs[k] * betas[k]

            for m in range(j, i):
                sot *= np.sin(mus[m + 1] - mus[m])

            sum += sot

    return sum * (-1)**(eqNo not in [2, 3])



@njit   
def equationGrad(errs, xj, betas, mus, eqNo = 1):

    match eqNo:
        case 1: 
            sectrig = lambda x,y: np.sin(x) * np.cos(y)
        case 2: 
            sectrig = lambda x,y: np.sin(x) * np.sin(y)
        case 3: 
            sectrig = lambda x,y: np.cos(x) * np.cos(y)
        case 4: 
            sectrig = lambda x,y: np.cos(x) * np.sin(y)


    sum = betas[xj] * sectrig(mus[xj], mus[xj])


    for i in range(1, N):
        for j in range(0, i-1):

            if xj not in range(j, i+1):
                continue

            sot = sectrig(mus[j], mus[i])
            
            for k in range(j, i+1):
                sot *= errs[k] * betas[k]

            for m in range(j, i):
                sot *= np.sin(mus[m + 1] - mus[m])
        
            sum += sot / errs[xj]


    return sum * (-1)**(eqNo not in [2, 3])






"""
        EQUATION SYSTEM
"""


def ErrorSimulation(quadrupoles, errors):
    """Takes a vector of errors that will be simulated on the 8 quadrupoles and
    outputs the value of the 2nd order equation for error"""
    
    return np.array([ equation(errors, getBet(quadrupoles, "X"), getMu(quadrupoles, "X"), eqNo = i+1) for i in range(int(N / 2)) ] + 
                    [ equation(errors, getBet(quadrupoles, "Y"), getMu(quadrupoles, "Y"), eqNo = i+1) for i in range(int(N / 2)) ])



def noiseWeights(noise, lambdas, weights = np.full(N, 0.1)):
    return 2 * weights * noise + lambdas


def functionConstrain(errors, quadrupoles, noise, RSC):
    return ErrorSimulation(quadrupoles, errors) + noise - RSC


def gradientConstrain(lambs, errors, quadrupoles):
    
    betasY = getBet(quadrupoles, "Y")
    musY = getBet(quadrupoles, "Y")
    betasX = getBet(quadrupoles, "X")
    musX = getBet(quadrupoles, "X")

    def getGradient(xj):
        return np.array([equationGrad(errors, xj, betasX, musX, i) for i in range(1, int(N / 2 + 1))] + 
                        [equationGrad(errors, xj, betasY, musY, i) for i in range(1, int(N / 2 + 1))] )

    return np.array( [np.dot(lambs, getGradient(j)) for j in range(N)] )


def residuals(params, quadrupoles, goal):

    errs = params[0:N]
    nois = params[N:2*N]
    lambs = params[2*N:]

    eqs = []

    eqs.extend(functionConstrain(errs, quadrupoles, nois, goal))
    eqs.extend(gradientConstrain(lambs, errs, quadrupoles))
    eqs.extend(noiseWeights(nois, lambs))

    return np.array(eqs)


def findSystemSolution(reg, errors=ERRs, treshold=TRESHOLD):
    """This function runs the iterative method for finding a solution of the system"""
    #
    # print(f"Interaction region NO: {reg}")
    # print(f"Goal errors: {errors}")

    left = selectedQP.loc[reg - 1, "leftX"]
    right = selectedQP.loc[reg - 1, "leftY"]

    left["BETX"] = left["BETX"]
    left["BETY"] = left["BETY"]
    right["BETX"] = right["BETX"]
    right["BETY"] = right["BETY"]

    quadrupoles = pd.concat([left, right])
    
    errors = np.random.normal(loc = 0.0, scale = 1e-4, size =N)
    noise = np.random.normal(loc=0.0, scale=1.0e-2, size=N)     # Noise added to our signal to simulate BPMs' noise
    rightSideConstants = ErrorSimulation(quadrupoles, errors)   # The systems expected right-side constants without noise
    noisy_constants = rightSideConstants + noise                           # The goal right side of the equation

    print("Added noise: \n", 100 * noise / noisy_constants)

    for x0 in [np.random.normal(loc=0.0, scale=1.0e-5, size=3*N) for i in range(100000)]:
    
        # print(f"Noise added: {np.mean(np.abs(noise / goal * 100))}")

        # x0 = np.random.normal(loc = 0.0, scale = 1e-3, size = 3*N)
        sol = root(residuals, x0, args=(quadrupoles, noisy_constants), method="hybr", tol=1e-15)
        
        if np.all(np.array(sol.x[0:N]) - errors < 1e-5):
            print("\n \n ============================= \n ============================ \n\n")
            print("Converges? ", sol.success)
            print()

            print("Solution Vector: \n", sol.x[0:N])
            print("Simulated errors: \n", errors)

            print()

            print("Found noise: \n", sol.x[N:2*N])
            print("Simulated noise: \n", noise)

            print()

            print("Lambdas: \n", sol.x[2*N:])
        
            print()

            print("Convergence in errors: \n", noisy_constants - ErrorSimulation(quadrupoles, sol.x[0:N]) - sol.x[N:2*N])
            
            


#
ERRs = np.random.normal(loc=0.0, scale=1.0e-4, size=6)
findSystemSolution(1, errors=ERRs)
