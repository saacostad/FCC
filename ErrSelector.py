"""
This script takes the best quadrupoles on each one of the interaction regions and performs
the error corrections accoring to the calculated theory (or something like that)
"""

from math import cos, sin

import numpy as np
import pandas as pd

from QPphysicalParam import (
    getQP as QPphysParam,
)  # From here, we get a pandas DataFrame with the physical parameters of the quadrupoles
from QPSelector import (
    MainFunction as selQP,
)  # From here, we get the best quadrupoles for each one of the Interaction Regions

# We import the dataframes from the other scripts
physParams = QPphysParam()
selectedQP = selQP().iloc[
    [0, 1, 3, 5],
]  # As the quadrupoles are repeated in the datafiles, we only select the first 4


"""
        GLOBAL PARAMETERS 
"""


# If the system of equations, aftereach iteration, don't change in more than
# this quantity, then we're satisfied
TRESHOLD = 0.8e-5


# This is the error we're gonna try to recreate
ERRs = np.array(list(range(1, 9))) * 10 ** (-7)

"""
    ACTION AND PHASE FOR EACH IP
The nominal action and phase are pretty much constants on each side of IP, so, we bring them here manually
NOTE: at first, this doesn't matter, but who knows if in the future they will
"""

HJ0s = [1.57318230159e-10, 1.59040817156e-10, 1.93292075956e-11, 4.33351930746e-11]
VJ0s = [1.09287769985e-13, 1.07739134429e-13, 9.28858562765e-10, 9.45894676299e-10]
HP0s = [0.587739114054, 0.588883553621, -2.15738114587, 2.479568333]
VP0s = [1.58772647432, 1.59059853389, 1.20002337969, -0.520781188437]

"""
    FUNCTION DEFINITIONS
In this part, we define different functions to make the script more readable
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


"""
    DATA MODIFICATION
In this part, we modify the original quadrupoles dataframes SO they contain the new column KL, which will be used 
in order to calculate the magnetic errors of the quadrupoles
"""

for index, row in selectedQP.iterrows():
    createRowKL(selectedQP.loc[index, "leftX"])
    createRowKL(selectedQP.loc[index, "leftY"])
    createRowKL(selectedQP.loc[index, "rightX"])
    createRowKL(selectedQP.loc[index, "rightY"])


"""
    MATRIX CREATION
"""


def firstOrderMatrix(quadrupoles):
    """This function takes una region number and creates the matrix from which one can get the linear terms"""

    # left = selectedQP.loc[reg - 1, "leftX"]
    # right = selectedQP.loc[reg - 1, "rightX"]
    #
    # quadrupoles = pd.concat([left, right])

    # We create the matrix with only the linear terms, which are \beta_i \sin(\psi_i) \cos(\psi_i)
    return np.diag(
        quadrupoles.apply(
            lambda qp: qp["BETX"] * sin(qp["MUX"]) * cos(qp["MUX"]), axis=1
        )  # This is the function that creates the matrix elements
    )


def secondOrderVector(qp, err=ERRs):
    """Similar to the previous one, this function creates a vector where each element corresponds to the second order terms
    of the equation, given an errors array err"""

    # For this first part, we'll create a vector array where it's first element is 0, and all the other elements are \cos(\psi_i) \beta_i B_ilinVector
    linVector = np.array(
        [0]
        + [qp.iloc[i]["BETX"] * cos(qp.iloc[i]["MUX"]) * err[i] for i in range(1, 8)]
    )

    # To create the next term, which takes into account the no linear terms, we have to create a strict lower triangular matrix whose i, j elements are
    # T_{i, j} = \beta_i sin(\psi_i) sin(\psi_j - \psi_i). Matrix T which multuplied by the errors vector, will give the non linear vector VJ

    def nonLinearMatrixTerms(vi, vj):
        """Function that, given an i, j entry of the matrix, returns the corresponding element"""

        i = int(vj)  # We have to cast the integer or else it explodes
        j = int(vi)

        return (
            qp.iloc[i]["BETX"]
            * sin(qp.iloc[i]["MUX"])
            * sin(qp.iloc[j]["MUX"] - qp.iloc[i]["MUX"])
        )

    # We create the corresponding matrix given the element-wise rule
    nonLinearMatrix = np.fromfunction(
        np.vectorize(nonLinearMatrixTerms), (8, 8), dtype=np.double
    )

    # We're only interested in the strictly lower half of the matrix
    nonLinearMatrix = np.tril(nonLinearMatrix, k=-1)

    # Now, we multiply the matrix by the errors values
    secondOrderVector = nonLinearMatrix @ err

    # Finally, we have to return the convolution of the linear and the second order vectors
    return linVector * secondOrderVector


"""
        EQUATION SYSTEM 
"""


def ErrorSimulation(quadrupoles, errors=ERRs):
    """Takes a vector of errors that will be simulated on the 8 quadrupoles and
    outputs the value of the 2nd order equation for error"""

    firstOrderTerm = firstOrderMatrix(quadrupoles) @ errors

    secondOrderTerm = secondOrderVector(quadrupoles, errors)

    return firstOrderTerm + secondOrderTerm


def findLinearSolutions(quadrupoles, solution):
    """Function that, given an interaction region _reg_, performs all the code to find the first approximations to
    the solutions of the errors, that is, solves the system of linear non-cross equations"""

    A = firstOrderMatrix(quadrupoles)

    Corrections = np.linalg.solve(A, solution)

    return Corrections


def findSystemSolution(reg, errors=ERRs, treshold=TRESHOLD):
    """This function runs the iterative method for finding a solution of the system"""

    print(f"Simulated Errors: {errors}")

    left = selectedQP.loc[reg - 1, "leftX"]
    right = selectedQP.loc[reg - 1, "rightX"]

    quadrupoles = pd.concat([left, right])

    rightSideConstants = ErrorSimulation(quadrupoles, errors)

    corrections = findLinearSolutions(quadrupoles, rightSideConstants)

    print("First order corrections: ")
    print(corrections)
    # secondOrderConstantTerm = secondOrderVector(quadrupoles, errors)

    i = 1
    while True:
        secondOrderTerm = secondOrderVector(quadrupoles, errors - corrections)
        rightSide = rightSideConstants - secondOrderTerm

        corrections = findLinearSolutions(quadrupoles, rightSide)

        print(f"Iteration {i}")
        print(corrections)

        i += 1


findSystemSolution(1, errors=ERRs)
