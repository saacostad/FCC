"""
A more general but simpler implementation of APJ's main code.

What it does:
    
    -> Takes the tbt file (trackone) and creates a file that saves all the data for each one of the elements 

    -> Computes the avermax trajectory
            
            Av_trajectory = (z_max + z_min) / 2.0

    -> From the avermax trajectory, it will look into the nominal twiss file and get the Psi's for each element. 
"""

import numpy as np 
import tfs as tfs
import pandas as pd
from tools.utils_ActPhase10 import doaccionyfase


# By how much scale the lectures of BPMs on each axis
z_scale = 1.e4


def read_madx_track(filepath):
    """
    This is in charge of reading the madx trackone file and dealing with it's weird formatting.

    Input: filepath of the trackone from the simulations 

    Output: dictionary with the values of s and lists for each turn of X and Y per element 

        dictionary[ELEMENT] = {"S": s, "X": [xs], "Y": [ys]}
    """
    
    # Open the file
    with open(filepath) as file:
        """
        We iterate over each line:
            -> Those that have @ $ * do not matter to us
            -> From the #segment ones, we'll only care about the last column (element name)
            -> From the one behind it, we'll only save the 3rd and 5th element (X and Y)
        """ 
        
        # We'll save our current element for now 
        current_element = None 

        elements_dict = {}              # Dictionary where wi'll save our data: dict[ELEMENT] = {s, [X], [Y]}
        
        SKIP ={'@', '*', '$'}           # The characters to skip the comment lines

        for line in file:
            
            # Get rid of the comment lines
            if line[0] in SKIP: 
                continue

            # The segment lines tells us the element
            if "#segment" in line:
                current_element = line.split()[-1]

                # If first time with this element, we create the data we need 
                if current_element not in elements_dict:
                    elements_dict[current_element] = {"S": None, "X": [], "Y": [], "amX": None, "amY": None} 

            else: 
                # Now we add the data from the lines that tell us the actual tbt  
                s_line = line.split()
                elements_dict[current_element]['X'].append(float(s_line[2])*z_scale)
                elements_dict[current_element]['Y'].append(float(s_line[4])*z_scale)
                elements_dict[current_element]['S'] = s_line[-2]
        
        # Return our database
        return elements_dict


def avermax(data):
    """
    This will calculate the avermax for each element.

    avermax(Z) = (Z.min() + Z.max()) / 2.0

    Input: the dataframe from our previous func 
    Output: the same dataframe but adding columns for the avermax
    """

    for row in data.values():
        
        Xs = row["X"]
        row["amX"] = (min(Xs) + max(Xs)) / 2.0

        Ys = row["Y"]
        row["amY"] = (min(Ys) + max(Ys)) / 2.0

    print(len(data.values()))


def save_avermax(data, output):
    """
    This function will take our avermax dataframe and save it's data 
    to a file with the same format as the normal APJ

    Inputs: data [dataframe], output [path]
    """
    
    # Open the output file
    with open(output, 'w') as file:
    
        # Iterate over the dictionary 
        for element, row in data.items():
            
            S = row["S"]
            X = row["amX"]
            Y = row["amY"]

            print(f"{element} {S} {X} {Y}", file = file)


def prepare_zPsi(twissFile, averFile):
    """
    This function will prepare the twissfile so it has the same order as our 
    avermax file so the action and phase function does not explodes.

    Input: twissFile and averFile are the paths to these files 
    Output: 4 lists ready to be passed to the APJ function xred, psix, yred, psiy
    """

    twissDF = tfs.read_tfs(twissFile)
    averDF = pd.read_csv(averFile, sep = " ", names=["NAME", "S", "X", "Y"])
    
    df = pd.merge(averDF[['NAME', 'X', 'Y']], twissDF[['NAME', 'MUX', 'MUY']], on='NAME')
    
    return df['X'].tolist(), df['MUX'].tolist(), df['Y'].tolist(), df['MUY'].tolist(),


# dfs = read_madx_track("fcc_ee_test_simulation/tbt/trackone")
# avermax(dfs)
# save_avermax(dfs, "avermax.csv")

## sqrt(beta)

'''
CONVENCIONES AVERMAX PROFESOR

   $1==1?$4 

'''

x, mux, y, muy = prepare_zPsi("fcc_ee_test_simulation/twiss.dat", "avermax.csv")

xaction, xphase = doaccionyfase(x, mux)
yaction, yphase = doaccionyfase(y, muy)


print(xphase)
