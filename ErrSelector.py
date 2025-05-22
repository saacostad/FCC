"""
This script takes the best quadrupoles on each one of the interaction regions and performs
the error corrections accoring to the calculated theory (or something like that)
"""

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
        ]


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


# At this point, all dataFrame may have their corresponding magnetic error
