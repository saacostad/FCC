import numpy as np 
import matplotlib.pyplot as plt 
import tools.FCC_matricial_system as FCC
import pandas as pd

# Working now with IP.2
IP = 2

# Path of the output files we'll be dealing with 
MUXpath = "APJresults/HmaxAPJaction_nofilt.sdds"
MUYpath = "APJresults/VmaxAPJaction_nofilt.sdds"
PHASEXpath = "APJresults/HmaxAPJphase_nofilt.sdds"
PHASEYpath = "APJresults/VmaxAPJphase_nofilt.sdds"

# Path of the integrals file from where to get the lattice functions
integrals_path = "fcc_ee_test_simulation/integrals.dat"


# Arcs regions to calculate previous and posterior APJ values
leftArc = (15000, 20000)
rightArc = (23500, 30000)


# TODO: check the best quadrupole selection

# Names of the quadrupoles we'll use to make the corrections
QUADRUPOLES_SELECTION = ["QC3L.2", "QC0L.2", "QC4.3", "QC0.3"]
# ATM I've chosen them so they have the most similar possible beta values for x and y


# We'll create a function to calculate the APJ parameters easily
def get_APJ_parameter(path, axis, left_arc, right_arc):
    """ Given a sdds file with the APJ calculations, it formats it on the region of interest, filters the axis to deal with, 
    and calculates the average of this parameter on the arc. 

    Returns the average value of the important parameter """
    
    # Which columns we'll be looking for
    filter = 0 if axis == 'X' else 1
    
    # Values list
    left_values = []
    right_values = []

    with open(path, 'r') as file:

        # Iterate the file
        lines = file.readlines()

        for line in lines:
            
            # Separate each column
            data = line.split()
            
            # Get the data from the line
            axis_val = int(data[0])
            value = float(data[-1])
            s = float(data[2])

            # Check if it belongs to any of the interest regions/filters
            if axis_val == filter and left_arc[0] <= s <= left_arc[1]:
                left_values.append(value)
            elif axis_val == filter and right_arc[0] <= s <= right_arc[1]: 
                right_values.append(value)
        
    
    # Return the mean of the values encountered
    return np.mean(left_values), np.mean(right_values)




def get_observed_system(mxp, myp, pxp, pyp):
    """ Given the 4 APJ .sdds paths, this function gets the values of each one of the APJ variables and,
    according to Santiago's theory, created the right hand side vector to be solved by the system of equations """

    # Obtenemos acciones y fases para eje x
    J0x, J1x = get_APJ_parameter(mxp, 'X', leftArc, rightArc)
    P0x, P1x = get_APJ_parameter(pxp, 'X', leftArc, rightArc)

    # Igualmente para el eje y
    J0y, J1y = get_APJ_parameter(myp, 'Y', leftArc, rightArc)
    P0y, P1y = get_APJ_parameter(pyp, 'Y', leftArc, rightArc)

    def calculate_S_contribution(J0, J1, P0, P1):
        """ Calculates the \sin(\psi_s) contribution according to the system of equations """
        return np.sqrt(J1/J0)*np.cos(P1) - np.cos(P0)

    def calculate_C_contribution(J0, J1, P0, P1):
        """ Calculates the \cos(\psi_s) contribution according to the system of equations """
        return -np.sqrt(J1/J0)*np.sin(P1) + np.sin(P0)
    
    # We calculate the RHS constants
    SinCont_X = calculate_S_contribution(J0x, J1x, P0x, P1x)
    cosCont_X = calculate_C_contribution(J0x, J1x, P0x, P1x)
    SinCont_Y = calculate_S_contribution(J0y, J1y, P0y, P1y)
    cosCont_Y = calculate_C_contribution(J0y, J1y, P0y, P1y)

    # Return the RHS vector
    return np.array([SinCont_X, cosCont_X, SinCont_Y, cosCont_Y])



def get_quadrupoles_lattice_functions(path, QPlist):
    """ Function that reads the integrals file generated and filters out the beta and phi functions
    for the quadrupoles that are being used at the moment """

    betx = list()
    bety = list()
    mux = list()
    muy = list()
    names = list()      # Just in case so we do not get the names scrambled up

    with open(path, 'r') as file:
        
        # Read the file
        lines = file.readlines()
        
        # Iterate over the liens
        for line in lines:
    
            data = line.split()
            
            # data[3] contains the name of the quadrupole without the "" so there's no need to format
            if data[3] in QPlist:

                names.append(data[3])

                betx.append(float(data[1]))
                bety.append(float(data[2]))

                # Do not forget to add the 2\pi to the phase
                mux.append(2.*np.pi*float(data[4]))
                muy.append(2.*np.pi*float(data[5]))
   
    # We create a dataframe for easy access to these values
    df = pd.DataFrame({
            'ELEMENT': names,
            'BETX': betx,
            'BETY': bety,
            'MUX': mux,
            'MUY': muy,
        })

    # Return given dataframe
    return df








print(get_quadrupoles_lattice_functions(integrals_path, QUADRUPOLES_SELECTION))
