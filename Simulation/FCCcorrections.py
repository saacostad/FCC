import numpy as np 
from tools.FCC_matricial_system import createSystem_base2 as createSystem   # This is the left hand side of the equation
import pandas as pd
from scipy.optimize import least_squares

# TODO: meterle ruido a las tbt [ arcos = 0.1mm, IR = 0.2mm ]
# TODO: revisar con el beta-beating


# Working now with IP.2
IP = 2

# Path of the output files we'll be dealing with
out_path = "fcc_ee_test_simulation/tbt_simu/sim_tbt"
MUXpath = out_path + "/HmaxAPJaction_nofilt.sdds"
MUYpath = out_path + "/VmaxAPJaction_nofilt.sdds"
PHASEXpath = out_path + "/HmaxAPJphase_nofilt.sdds"
PHASEYpath = out_path + "/VmaxAPJphase_nofilt.sdds"

# Path of the integrals file from where to get the lattice functions
integrals_path = "fcc_ee_test_simulation/integrals.dat"


# Arcs regions to calculate previous and posterior APJ values
# TODO: check the quadrupoles for IP 1 and 4
def get_arc(IP):
    leftArc = None 
    rightArc = None 
    QUADRUPOLES_SELECTION = None

    if IP == 2:
        leftArc = (15000, 20000)
        rightArc = (23500, 30000)
        # QUADRUPOLES_SELECTION = ["QC4L.1", "QC3.2"]
        QUADRUPOLES_SELECTION = ["QC4L.1", "QC3L.1", "QC0.2", "QC3.2"]
    elif IP == 5 or IP == 3:
        leftArc = (32000, 41900)
        rightArc = (47000, 60000)
        QUADRUPOLES_SELECTION = ["QC4L.2", "QC3L.2", "QC0.3", "QC3.3"]
    elif IP == 7 or IP == 4:
        leftArc = (60000, 66000)
        rightArc = (69000, 80000)
        QUADRUPOLES_SELECTION = ["QC4L.3", "QC3L.3", "QC0.4", "QC3.4"]
    elif IP == 8 or IP == 1:
        leftArc = (80000, 88000)
        rightArc = (1000, 14000)
        QUADRUPOLES_SELECTION = ["QC4L.1", "QC3L.1", "QC0.2", "QC3.2"]
    
    return leftArc, rightArc, QUADRUPOLES_SELECTION

# TODO: check the best quadrupole selection
# Names of the quadrupoles we'll use to make the corrections
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
    according to Santiago's theory, created the right hand side vector to be solved by the system of equations 

    OUTPUT: np.array with the constants of RHS  |   value of delta_0x   |   value of delta_0y"""

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
    return np.array([SinCont_X, cosCont_X, SinCont_Y, cosCont_Y]), P0x, P0y



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



"""
=================================================================
        MAIN EXECUTION OF THE SCRIPT
=================================================================
"""

if __name__ == '__main__':

    leftArc, rightArc, QUADRUPOLES_SELECTION = get_arc(IP)

    # Initial guess for the errors
    ERR_init = np.zeros(len(QUADRUPOLES_SELECTION))

    # First, we get the right hand side vector of the system 
    RHS, delta0_x, delta0_y = get_observed_system(MUXpath, MUYpath, PHASEXpath, PHASEYpath)

    # In order to create the left hand side, we need to retreive the lattice functions of the quadrupoles of interest
    latticeDF = get_quadrupoles_lattice_functions(integrals_path, QUADRUPOLES_SELECTION)

    # We will now create simple lists of the lattice functions for easier access
    BETX = latticeDF['BETX'].to_numpy()
    BETY = latticeDF['BETY'].to_numpy()
    MUX = latticeDF['MUX'].to_numpy()
    MUY = latticeDF['MUY'].to_numpy()


    # We'll create the residual function to use with Least_Squares()
    def residual(K):

        # We create the constants for both axis
        Sx, Cx = createSystem(K, BETX, MUX, delta0_x, axis = 'X')
        Sy, Cy = createSystem(K, BETY, MUY, delta0_y, axis = 'Y')
        
        # Return the residual
        return np.array([Sx, Cx, -Sy, -Cy]) - RHS


    """ CALCULATE THE ERRORS STIMATIONS """
    ERR_estimations = least_squares(residual, ERR_init, ftol = 1e-12).x
    

    print("Errors estimation:")
    print(QUADRUPOLES_SELECTION)
    print(ERR_estimations)

    print("\nWith a residue of: ", residual(ERR_estimations))


    # Write to the file
    # Build lookup dictionary
    err_dict = dict(zip(QUADRUPOLES_SELECTION, ERR_estimations))

    # Read and modify file
    with open("IR_errors+corrections.madx", "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        # Get the quadrupole name
        # TODO: this only works when I add errors, as if there are none, the -> will never appear
        quad_name = line.split("->")[0].strip()

        if quad_name in err_dict:
            err = err_dict[quad_name]

            # Remove trailing semicolon/newline, append new term, add semicolon back
            if err < 0:
                line = line.rstrip(";\n") + f"-{abs(err)};\n"
            else:
                line = line.rstrip(";\n") + f"+{abs(err)};\n"

        new_lines.append(line)

    # Write back
    with open("IR_errors+corrections.madx", "w") as f:
        f.writelines(new_lines)


    print("Finished writing to file")
