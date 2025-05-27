"""
This script is in charge of creating a pandas Dataframe with the physical properties of the quadrupoles.
"""

import pandas as pd

# Parameters to filter
MAGNET_PARAMETERS = ["K1", "L"]


# Path to the file with the physical properties of the quadrupoles
file_path = "data/fccee_t.seq"


# Create a dictionary from which we create then a pandas DataFrame df
inLines = dict.fromkeys(["NAME"] + MAGNET_PARAMETERS)
global QPparams


nameFlag = True  # This flag is used to keep track if we've already chosen the names of the quadrupoles

for param in MAGNET_PARAMETERS:
    with open(file_path, "r") as file:
        """ For each one of the chosen parameters, we open the .seq file and choose only the lines
            that start with the parameter"""
        lines = [line for line in file if line.startswith(f"{param + 'Q'}")]

        names = list()
        values = list()

        # We iterate over the lines and saveup the data
        for line in lines:
            name, value = line[len(param) :].split(" = ")
            value = value.replace(";\n", "")
            names.append(name)
            values.append(float(value))

        # If we haven't already saved the names of the QPs, we save them
        if nameFlag:
            inLines["NAME"] = names
            nameFlag = False

        # We add the data to the respective key of the dictionary
        inLines[param] = values

# We create a dictionary using the chosen data.
QPparams = pd.DataFrame(inLines)


# As wwe're actually interested in the product of K1*L, we create a new column with this data
QPparams["KL"] = QPparams["K1"] * QPparams["L"]


def getQP():
    return QPparams
