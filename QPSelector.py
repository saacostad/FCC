"""Script made to select the best quadrupoles on each side of the Interaction Points
according to chosen parameters"""

from os import listdir  # Library used to read the datafile names

import numpy as np  # Will be helpful specially for array manipulation
import pandas as pd  # Even though tfs imports it, I need to use special functions of pandas
import tfs  # CERN package to read .tsf files. It already imports pandas and other useful libraries

# From other script, we created a DataFrame that contains the information of K and L of the quadrupoles

"""
        ADJUSTABLE VARIABLES
Variables and global parameters so the script performs what the user wants
"""

FILE_NO = 1  # Take the file #n on the list of twiss__.tfs files on the /data/ folder

# What columns to save from the twiss filer
PARAMETERS = ["NAME", "KEYWORD", "S", "MUX", "MUY", "BETX", "BETY"]

# Select the keywords for the elements to analyse
KEYWORDS = ["MARKER", "QUADRUPOLE"]

# Number of quadrupoles next to each interaction region to check.
NEAR_IP_QP_WINDOW = 20

# How many quadrupoles to select on each side of IPs (has to be an even number)
SELECTED_QP = 8


"""
            READING FILE DATA
Just reading the data using the tfs module, which will store the read table in a pandas dataFrame
"""

# Get the datafiles names of the Twiss files just to select
# which one we'll do the analysis with
data_files = [f for f in listdir("data") if f.endswith(".tfs")]

# Save the name of the file chosen
file_path = "data/" + data_files[FILE_NO - 1]

# Save all the data from the .tsf file
rawData = pd.DataFrame(tfs.read(file_path))

"""
            MODIFY DATA 
As we're interested purely on the phases of the quadrupoles, specifically those closer to Int. Regions,,
we actually only need certain data: that of the quadrupoles, whose KEYWORD is QUADRUPOLE, and we only 
need information about the phase MU(X/Y) [but I will save other info as the position S and the name]; 
and the same data for the interaction regions, named IP.n, where for now, I'll save the same information.
"""

columnData = rawData[PARAMETERS]  # DataFrame with only the columns in PARAMETERS

# We create the dataFrames we need
for key in KEYWORDS:
    """ we create a dataframe with only the rows with the selected keyword """
    exec(f"df{key} = columnData.loc[columnData['KEYWORD'] == '{key}']")

    # I'm saving all this data in dynamic variables so they are easier to control. In this case, I'll create
    # 2 dataframes, dfQUADRUPOLE and dfMARKER that can be accesible as a normal variable.


""" We create a dataFrame where all the IPs are stored. From here, and for each IP, we'll look for the best Quadrupoles """
dfIP = dfMARKER.loc[dfMARKER["NAME"].str.contains(r"^IP\.\d+$", na=False)]
# If you get an error here, it is because the variable which the dataFrame reads from is created dynamically,
# so the text editor doesn't find it.


""" 
        LOOK FOR THE BEST QUADRUPOLES
In this section, we'll create a function that takes one IP and will return 16 total quadrupoles, 8 on each side (left and right)
and from each side, 4 with the highest BETX and 4 with the highest BETY. The number of quadrupoles to search on each side is 
NEAR_IP_QP_WINDOW. 
"""


def Check_rows(df, Selection=int(SELECTED_QP / 2), Columns=["BETX", "BETY"]):
    """function that takes a dataframe of quadrupoles, selects the ones with the bests beta functions, checks there are no repeated quadrupoles
    on the selection for both planes, and returns 2 dataframes with only the best quadrupoles"""

    # We sort the dataFrames from their beta values
    dfSorted_bX = df.sort_values(by=f"{Columns[0]}", ascending=False)
    dfSorted_bY = df.sort_values(by=f"{Columns[1]}", ascending=False)

    # We start by choosing the bests 4
    dfSelected_bX = dfSorted_bX.head(Selection)
    dfSelected_bY = dfSorted_bY.head(Selection)

    counter = 0  # Counter in case we need to replace repeating QPs
    while True:
        # We check is there are any repeated quadrupoles in both sides
        merged = pd.merge(dfSelected_bX, dfSelected_bY, how="left", indicator=True)
        common_rows = pd.DataFrame(merged[merged["_merge"] == "both"])

        if common_rows.empty:
            # If there are none, just continue
            break
        else:
            # If there are, change the the repeated quadrupole for the next highest-beta-Y value
            for row in common_rows.itertuples(index=False):
                dfSelected_bY.iloc[dfSelected_bY["NAME"] == f"{row[0]}"] = (
                    dfSorted_bY.iloc[[Selection + counter]]
                )

                dfSelected_bY = dfSelected_bY.sort_values(
                    by=f"{Columns[1]}", ascending=False
                )

                # Increase the counter just in case we have to replace the QP again
                counter += 1
                # TODO: by now, the only betas that are being replaced are the betaY, but there are cases where betaX is a better choice to be replaced. That is, add logic to check wheter to change BETAX or BETAY

    # Return the selected quadrupoles
    return (dfSelected_bX, dfSelected_bY)


def Select_Quadrupoles(
    IPnumber, window=NEAR_IP_QP_WINDOW, selection=int(SELECTED_QP / 2)
):
    """Given the number of the IP to work with, selects the best 8 quadrupoles on a given window on each side.
    The criteria to select the quadrupoles is they have the highest BETA functions, chosing 4 and 4 for each plane
    (or SELECTED_QP / 2)"""

    # We take the index of the IP
    IP_index = dfIP.index[dfIP["NAME"] == f"IP.{IPnumber}"][0]

    # We create an array of all indexes of the QUADRUPOLES
    QP_indexes = dfQUADRUPOLE.index

    # Look for the closest quadrupole to the IP using np.searchsorted
    Closest_QP_index = np.searchsorted(QP_indexes, IP_index)

    df = dfQUADRUPOLE.copy()
    df = pd.concat([df.tail(NEAR_IP_QP_WINDOW), df])
    df = pd.concat([df, dfQUADRUPOLE.head(NEAR_IP_QP_WINDOW)])

    Closest_QP_index += NEAR_IP_QP_WINDOW
    Closest_QP = df.index[Closest_QP_index]

    # Create dataFrames for each side of the IP
    if Closest_QP > IP_index:
        rightSide_QP = df.iloc[Closest_QP_index : Closest_QP_index + NEAR_IP_QP_WINDOW]
        leftSide_QP = df.iloc[Closest_QP_index - NEAR_IP_QP_WINDOW : Closest_QP_index]
    else:
        rightSide_QP = df.iloc[Closest_QP_index : Closest_QP_index + NEAR_IP_QP_WINDOW]
        leftSide_QP = df.iloc[Closest_QP_index - NEAR_IP_QP_WINDOW : Closest_QP_index]

    # Check there are no repeated quadrupoles in both sides
    leftSide_selected_X, leftSide_selected_Y = Check_rows(leftSide_QP)
    rightSide_selected_X, rightSide_selected_Y = Check_rows(rightSide_QP)

    # Return the selected quadrupoles
    return (
        leftSide_selected_X,
        leftSide_selected_Y,
        rightSide_selected_X,
        rightSide_selected_Y,
    )


def PrintQuadrupoles():
    """If called, it will print out all the quadrupoles selected"""
    for i in [1, 8]:
        print(f"ITERACION {i}")
        QP = Select_Quadrupoles(i)
        print(f" FOR INTERACTION REGION NO {i}:")
        print()
        print()

        print("--------------------------")
        print("Left side quadrupoles:")
        print("BETA X:")
        print(QP[0])

        print()

        print("BETA Y:")
        print(QP[1])

        print("--------------------------")
        print("Right side quadrupoles:")
        print("BETA X:")
        print(QP[2])

        print()

        print("BETA Y:")
        print(QP[3])


"""         MAIN EXECUTION      """

# The main dictionary to be created, were the data of the QPs is saved
QPdict = None


def MainFunction():
    keys = ["leftX", "leftY", "rightX", "rightY"]

    data = [Select_Quadrupoles(i) for i in range(1, 9)]

    QPdict = pd.DataFrame(data, columns=keys)

    return QPdict


if __name__ == "__main__":
    df = MainFunction()
    # Qprint(df.iloc[0, 2])
