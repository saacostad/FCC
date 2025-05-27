"""
This script takes the best quadrupoles on each one of the interaction regions and performs
the error corrections accoring to the calculated theory (or something like that)
"""

from math import cos, sin

import numpy as np

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
    ACTION AND PHASE FOR EACH IP
The nominal action and phase are pretty much constants on each side of IP, so, we bring them here manually
"""

HJ0s = [1.57318230159e-10, 1.59040817156e-10, 1.93292075956e-11, 4.33351930746e-11]
VJ0s = [1.09287769985e-13, 1.07739134429e-13, 9.28858562765e-10, 9.45894676299e-10]
HP0s = [0.587739114054, 0.588883553621, -2.15738114587, 2.479568333]
VP0s = [1.58772647432, 1.59059853389, 1.20002337969, -0.520781188437]


# This is the error we're gonna try to recreate
ERR = [0.000003] * 8


# A value for the lattice function to choose
phi_s = 400.0


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


def diagBetaMatrix(reg):
    """Creates a diagonal matrix with the beta functions of each quadrupole of the IR. It is used
    as a factor in the equation, taking into account the beta_i can be factorized out"""

    left = selectedQP.loc[reg - 1, "leftX"]
    right = selectedQP.loc[reg - 1, "rightX"]

    return np.diag(list(left.loc[:, "BETX"]) + list(right.loc[:, "BETX"]))


def diagWeightedMatrix(reg):
    """Same thing as previous function but with the weights of each quadrupole for the linear expressions"""

    left = list(selectedQP.loc[reg - 1, "leftX"].loc[:, "MUX"])
    right = list(selectedQP.loc[reg - 1, "rightX"].loc[:, "MUX"])

    MUs = left + right
    DELTA0 = HP0s[reg - 1]

    firstMatrix = (
        np.diag([cos(MUs[i]) * sin(MUs[i]) for i in range(0, 8)])
        * cos(DELTA0)
        * sin(phi_s)
    )
    secondMatrix = (
        np.diag([-sin(MUs[i]) * sin(MUs[i]) for i in range(0, 8)])
        * cos(DELTA0)
        * cos(phi_s)
    )
    thirdMatrix = (
        np.diag([-cos(MUs[i]) * cos(MUs[i]) for i in range(0, 8)])
        * sin(DELTA0)
        * sin(phi_s)
    )
    fourthMatrix = (
        np.diag([sin(MUs[i]) * cos(MUs[i]) for i in range(0, 8)])
        * sin(DELTA0)
        * cos(phi_s)
    )

    return firstMatrix + secondMatrix + thirdMatrix + fourthMatrix


def linearMatrix(reg):
    """This function returns a matrix that corresponds to the one used to solve the linear terms,
    ignoring the cross terms"""
    betaMatrix = diagBetaMatrix(reg)
    weightMatrix = diagWeightedMatrix(reg)

    return betaMatrix * weightMatrix


# Solution vector
def findLinearSolutions(reg):
    """Function that, given an interaction region _reg_, performs all the code to find the first approximations to
    the solutions of the errors, that is, solves the system of linear non-cross equations"""

    A = linearMatrix(reg)
    Sol = [sin(phi_s - HP0s[reg - 1])] * 8

    Corrections = np.linalg.solve(A, Sol)

    return Corrections


print(findLinearSolutions(1))
