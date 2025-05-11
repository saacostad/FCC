from os import listdir  # Library used to read the datafile names

import numpy as np  # Will be helpful specially for array manipulation
import tfs  # CERN package to read .tsf files. It already imports pandas and other useful libraries

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
data_files = listdir("data")

# Save the name of the file chosen
file_path = "data/" + data_files[FILE_NO - 1]

# Save all the data from the .tsf file
rawData = tfs.read(file_path)


"""
            MODIFY DATA 
As we're interested purely on the phases of the quadrupoles, specifically those closer to Int. Regions,,
we actually only need certain data: that of the quadrupoles, whose KEYWORD is QUADRUPOLE, and we only 
need information about the phase MU(X/Y) [but I will save other info as the position S and the name]; 
and the same data for the interaction regions, named IP.n, where for now, I'll save the same information.
"""

columnData = rawData[PARAMETERS]  # DataFrame with only the columns in PARAMETERS


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
    Closest_QP = QP_indexes[Closest_QP_index]

    # Create dataFrames for each side of the IP
    if Closest_QP > IP_index:
        rightSide_QP = dfQUADRUPOLE.iloc[
            Closest_QP_index : Closest_QP_index + NEAR_IP_QP_WINDOW
        ]
        leftSide_QP = dfQUADRUPOLE.iloc[
            Closest_QP_index - NEAR_IP_QP_WINDOW : Closest_QP_index
        ]
    else:
        rightSide_QP = dfQUADRUPOLE.iloc[
            Closest_QP_index - 1 - NEAR_IP_QP_WINDOW : Closest_QP_index
        ]
        leftSide_QP = dfQUADRUPOLE.iloc[
            Closest_QP_index : Closest_QP_index + NEAR_IP_QP_WINDOW - 1
        ]

    print(leftSide_QP)
    print(dfIP[dfIP["NAME"] == f"IP.{IPnumber}"])
    print(rightSide_QP)


Select_Quadrupoles(6)
