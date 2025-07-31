"""
This script takes the best quadrupoles on each one of the interaction regions and performs
the error corrections accoring to the calculated theory (or something like that)
"""

from math import cos, sin

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from QPphysicalParam import (
    getQP as QPphysParam,
)  # From here, we get a pandas DataFrame with the physical parameters of the quadrupoles
from QPSelector import (
    MainFunction as selQP,
)  # From here, we get the best quadrupoles for each one of the Interaction Regions

# We import the dataframes from the other scripts
physParams = QPphysParam()

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
ERRs = np.array(list(range(1, 9))) * 10 ** (-5)


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


#   FIRST ORDER EQUATIONS LOGIC
# -----------------------------------------------------------------------------------------


def firstOrderMatrixRows(quadrupoles, term, plane):
    """Creates a row for the First Order Matrix given the quadrupoles dataframe,
    the term number (the row for the X plane) and the plane"""

    match term:
        case 2:
            return np.array(
                [
                    quadrupoles.iloc[i][f"BET{plane}"]
                    * sin(quadrupoles.iloc[i][f"MU{plane}"]) ** 2
                    for i in range(0, 8)
                ]
            )

        case 3:
            return np.array(
                [
                    quadrupoles.iloc[i][f"BET{plane}"]
                    * cos(quadrupoles.iloc[i][f"MU{plane}"]) ** 2
                    for i in range(0, 8)
                ]
            )

        case 1:
            return np.array(
                [
                    -quadrupoles.iloc[i][f"BET{plane}"]
                    * sin(quadrupoles.iloc[i][f"MU{plane}"])
                    * cos(quadrupoles.iloc[i][f"MU{plane}"])
                    for i in range(0, 8)
                ]
            )
        case 4:
            return np.array(
                [
                    -quadrupoles.iloc[i][f"BET{plane}"]
                    * sin(quadrupoles.iloc[i][f"MU{plane}"])
                    * cos(quadrupoles.iloc[i][f"MU{plane}"])
                    for i in range(0, 8)
                ]
            )


def FirstOrderMatrix(quadrupoles):
    """This function creates the matrix for the first order terms of the problem,
    given the dataframe of the quadrupoles to analyse"""

    return np.array(
        [
            firstOrderMatrixRows(quadrupoles, 1, "X"),
            firstOrderMatrixRows(quadrupoles, 2, "X"),
            firstOrderMatrixRows(quadrupoles, 3, "X"),
            firstOrderMatrixRows(quadrupoles, 4, "X"),
            firstOrderMatrixRows(quadrupoles, 1, "Y"),
            firstOrderMatrixRows(quadrupoles, 2, "Y"),
            firstOrderMatrixRows(quadrupoles, 3, "Y"),
            firstOrderMatrixRows(quadrupoles, 4, "Y"),
        ]
    )


#   SECOND ORDER EQUATION LOGIC
# --------------------------------------------------------------------------------------


def secondOrderVector1(qp, err=ERRs, f=lambda qp: (qp["BETX"], qp["MUX"])):
    linVector = np.array(
        [0] + [f(qp.iloc[i])[0] * cos(f(qp.iloc[i])[1]) * err[i] for i in range(1, 8)]
    )

    def nonLinearMatrixTerms1(vi, vj):
        i, j = int(vj), int(vi)
        return (
            f(qp.iloc[i])[0]
            * sin(f(qp.iloc[i])[1])
            * sin(f(qp.iloc[j])[1] - f(qp.iloc[i])[1])
        )

    nonLinearMatrix = np.tril(
        np.fromfunction(np.vectorize(nonLinearMatrixTerms1), (8, 8), dtype=np.double),
        k=-1,
    )
    return -1 * linVector * (nonLinearMatrix @ err)


def secondOrderVector2(qp, err=ERRs, f=lambda qp: (qp["BETX"], qp["MUX"])):
    linVector = np.array(
        [0] + [f(qp.iloc[i])[0] * sin(f(qp.iloc[i])[1]) * err[i] for i in range(1, 8)]
    )

    def nonLinearMatrixTerms2(vi, vj):
        i, j = int(vj), int(vi)
        return (
            f(qp.iloc[i])[0]
            * sin(f(qp.iloc[i])[1])
            * sin(f(qp.iloc[j])[1] - f(qp.iloc[i])[1])
        )

    nonLinearMatrix = np.tril(
        np.fromfunction(np.vectorize(nonLinearMatrixTerms2), (8, 8), dtype=np.double),
        k=-1,
    )
    return linVector * (nonLinearMatrix @ err)


def secondOrderVector3(qp, err=ERRs, f=lambda qp: (qp["BETX"], qp["MUX"])):
    linVector = np.array(
        [0] + [f(qp.iloc[i])[0] * cos(f(qp.iloc[i])[1]) * err[i] for i in range(1, 8)]
    )

    def nonLinearMatrixTerms3(vi, vj):
        i, j = int(vj), int(vi)
        return (
            f(qp.iloc[i])[0]
            * cos(f(qp.iloc[i])[1])
            * sin(f(qp.iloc[j])[1] - f(qp.iloc[i])[1])
        )

    nonLinearMatrix = np.tril(
        np.fromfunction(np.vectorize(nonLinearMatrixTerms3), (8, 8), dtype=np.double),
        k=-1,
    )
    return linVector * (nonLinearMatrix @ err)


def secondOrderVector4(qp, err=ERRs, f=lambda qp: (qp["BETX"], qp["MUX"])):
    linVector = np.array(
        [0] + [f(qp.iloc[i])[0] * sin(f(qp.iloc[i])[1]) * err[i] for i in range(1, 8)]
    )

    def nonLinearMatrixTerms4(vi, vj):
        i, j = int(vj), int(vi)
        return (
            f(qp.iloc[i])[0]
            * cos(f(qp.iloc[i])[1])
            * sin(f(qp.iloc[j])[1] - f(qp.iloc[i])[1])
        )

    nonLinearMatrix = np.tril(
        np.fromfunction(np.vectorize(nonLinearMatrixTerms4), (8, 8), dtype=np.double),
        k=-1,
    )
    return -1 * linVector * (nonLinearMatrix @ err)


def secondOrderGeneralVector(quadrupoles, err=ERRs):
    """This function creates the vector for the second order terms"""

    return np.array(
        [
            np.sum(secondOrderVector1(quadrupoles, err)),
            np.sum(secondOrderVector2(quadrupoles, err)),
            np.sum(secondOrderVector3(quadrupoles, err)),
            np.sum(secondOrderVector4(quadrupoles, err)),
            np.sum(
                secondOrderVector1(
                    quadrupoles, err, f=lambda qp: (qp["BETY"], qp["MUY"])
                )
            ),
            np.sum(
                secondOrderVector2(
                    quadrupoles, err, f=lambda qp: (qp["BETY"], qp["MUY"])
                )
            ),
            np.sum(
                secondOrderVector3(
                    quadrupoles, err, f=lambda qp: (qp["BETY"], qp["MUY"])
                )
            ),
            np.sum(
                secondOrderVector4(
                    quadrupoles, err, f=lambda qp: (qp["BETY"], qp["MUY"])
                )
            ),
        ]
    )


reg = regs[2]
leftX = selectedQP.loc[reg, "leftX"]
rightX = selectedQP.loc[reg, "rightX"]
quadrupolesX = pd.concat([leftX, rightX])

leftY = selectedQP.loc[reg, "leftY"]
rightY = selectedQP.loc[reg, "rightY"]
quadrupolesY = pd.concat([leftY, rightY])

quadrupolesLEFT = pd.concat([leftX, leftY])

# print(quadrupolesLEFT)
# print(secondOrderGeneralVector(quadrupolesLEFT, ERRs))
#


def GetConstants(quadrupoles, errors=ERRs):
    Vectorsx = []
    Vectorsy = []
    totalVectors = []
    f_y = lambda qp: (qp["BETY"], qp["MUY"])

    for i in range(1, 5):
        # Obtener las funciones dinámicamente por nombre
        firstOrderFunc = globals()[f"firstOrderMatrix{i}"]
        secondOrderFunc = globals()[f"secondOrderVector{i}"]

        # Calcular contribuciones
        firstOrder = firstOrderFunc(quadrupoles) @ errors
        secondOrder = secondOrderFunc(quadrupoles, errors)

        totalVector = firstOrder + secondOrder
        Vectorsx.append(np.sum(totalVector))
    for i in range(1, 5):
        # Obtener las funciones dinámicamente por nombre
        firstOrderFunc = globals()[f"firstOrderMatrix{i}"]
        secondOrderFunc = globals()[f"secondOrderVector{i}"]

        # Calcular contribuciones
        firstOrder = firstOrderFunc(quadrupoles, f=f_y) @ errors
        secondOrder = secondOrderFunc(quadrupoles, errors, f=f_y)

        totalVector = firstOrder + secondOrder
        Vectorsy.append(np.sum(totalVector))

    totalVectors = Vectorsx + Vectorsy
    return totalVectors


# print(firstOrderMatrix1(quadrupolesX, "Y"))

# firstOrder = firstOrderMatrix(quadrupoles) @ ERRs
# secondOrder = secondOrderVector(quadrupoles, ERRs)
# C1 = GetConstants(quadrupoles, ERRs)
# print(C1)


"""
        EQUATION SYSTEM
"""


def ErrorSimulation(quadrupoles, errors=ERRs):
    """Takes a vector of errors that will be simulated on the 8 quadrupoles and
    outputs the value of the 2nd order equation for error"""
    firstOrderTerm = FirstOrderMatrix(quadrupoles) @ errors
    secondOrderTerm = secondOrderGeneralVector(quadrupoles, errors)
    return firstOrderTerm + secondOrderTerm


def findLinearSolutions(quadrupoles, solution):
    """Function that, given an interaction region _reg_, performs all the code to find the first approximations to
    the solutions of the errors, that is, solves the system of linear non-cross equations"""

    A = FirstOrderMatrix(quadrupoles)

    Errors, t1, t2, t3 = np.linalg.lstsq(A, solution)
    return Errors


def residuals(params, quadrupoles, goal):
    return (
        FirstOrderMatrix(quadrupoles) @ params
        + secondOrderGeneralVector(quadrupoles, params)
        - goal
    )


def chi2(e1, e2, e3, e4, e5, e6, e7, e8, goal, quadrupoles):
    params = [e1, e2, e3, e4, e5, e6, e7, e8]
    return np.sum((goal - (ErrorSimulation(quadrupoles, params))) ** 2)


def findSystemSolution(reg, errors=ERRs, treshold=TRESHOLD):
    """This function runs the iterative method for finding a solution of the system"""

    print(f"Interaction region NO: {reg}")
    print(f"Goal errors: {errors}")

    left = selectedQP.loc[reg - 1, "leftX"]
    right = selectedQP.loc[reg - 1, "leftY"]

    left["BETX"] = left["BETX"]
    left["BETY"] = left["BETY"]
    right["BETX"] = right["BETX"]
    right["BETY"] = right["BETY"]

    quadrupoles = pd.concat([left, right])

    rightSideConstants = ErrorSimulation(quadrupoles, errors)

    # corrections = findLinearSolutions(quadrupoles, rightSideConstants)
    #
    # print("First order corrections: ")
    # print(corrections)
    # secondOrderConstantTerm = secondOrderVector(quadrupoles, errors)

    # test = FirstOrderMatrix(quadrupoles) @ corrections - rightSideConstants
    # print(f": {test}")

    # i = 1
    # while i < 8:
    #     secondOrderTerm = secondOrderGeneralVector(quadrupoles, corrections)
    #
    #     rightSide = rightSideConstants + secondOrderTerm
    #
    #     corrections = findLinearSolutions(quadrupoles, rightSide)
    #
    #     print(f"Iteration {i}")
    #     print(corrections)
    #
    #     i += 1
    #
    # test = FirstOrderMatrix(quadrupoles) @ corrections - rightSideConstants
    # print(f"second test: {test}")

    initial_guess = np.array([0.0] * 8)
    print(f"Goal constants: {rightSideConstants}")
    print("===============================================")
    errorsList = list()

    for noise in [np.random.normal(loc=0.0, scale=1.0e-3, size=8) for i in range(5)]:
        goal = rightSideConstants + noise
        print(f"Noise added: {noise}")

        min = 1000
        minErrArr = None
        for initial_guess in [
            np.random.normal(loc=0.0, scale=1.0e-5, size=8) for i in range(5)
        ]:
            res = least_squares(
                residuals,
                initial_guess,
                args=(quadrupoles, goal),
                # bounds=(np.full(8, -9e-4), np.full(8, 9e-4)),
                # x_scale="jac",
                gtol=1e-10,
            )

            if np.sum(np.abs(ErrorSimulation(quadrupoles, res.x) - goal)) < min:
                min = np.sum(np.abs(ErrorSimulation(quadrupoles, res.x) - goal))
                minErrArr = res.x

        print(
            f"Total sum. of errors respect to addded noise : {np.sum(np.abs(ErrorSimulation(quadrupoles, minErrArr) - goal))}"
        )

        print(f"Found errors: {minErrArr}")
        # print(f"test: {ErrorSimulation(quadrupoles, res.x) - rightSideConstants}")
        print(
            f"Total sum. of test: {np.sum(np.abs(ErrorSimulation(quadrupoles, minErrArr) - rightSideConstants))}"
        )
        # print(f"Time taken for scipy: {end - start:.4f} seconds")
        print("===============================================")

        errorsList.append(minErrArr)

    print("\n\n")
    print(f"Goal errors: {errors}")
    print(f"Average error found: {np.mean(errorsList, axis=0)}")
    print(
        f"General loss of average: {np.sum(np.abs(rightSideConstants - ErrorSimulation(quadrupoles, np.mean(errorsList, axis=0))))}"
    )


ERRs = np.random.normal(loc=0.0, scale=1.0e-4, size=8)
findSystemSolution(1, errors=ERRs)
# print(f"Simulated Errors: {ERRs}")
# newerrors=findLinearSolutions(quadrupoles, C1)
# print(newerrors)
