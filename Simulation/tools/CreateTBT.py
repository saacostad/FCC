"""
This python script is made in order to create the TBT_from_PTC3.madx file for an arbitrary accelerator
"""

import argparse
import tfs as tfs 
import numpy as np


# We start by parsing the inputs of the command like
parser = argparse.ArgumentParser()
parser.add_argument(
        "-if", "--input_file",
        help="Input file of twiss file of the given accelerator. NOTE: it has to have a keyword column",
        required=True,
        dest="input_file"
    )
parser.add_argument(
        "-ip", "--interaction_point",
        help="The interaction point IP we'll work around",
        required=True,
        dest="IP"
    )
parser.add_argument(
        "-cf", "--corrections_file",
        help="If there exists corrections, this is the file to them",
        dest="corr_file"
    )
parser.add_argument(
        "-w", "--window",
        help="Over which region [m] around the IP we'll select the quadrupoles",
        required=True,
        dest="window"
    )
parser.add_argument(
        "-re", "--random_errors",
        help="Flag that tells if we want to add random errors to the selected quadrupoles",
        action="store_true",
        dest="random_flag"
    )
parser.add_argument(
        "-rs", "--random_sigma",
        help="The average dispersion of the errors (sigma of a normal)",
        dest="random_sigma"
    )
args = parser.parse_args()


def get_elements_around_ip(df, ip_name, window):
    """ This function will get the quadrupoles around a given IP """

    circumference = df["S"].iloc[-1]    # First, check the position of the last element 
    ip_s = df.loc[df["NAME"] == ip_name, "S"].values[0]     # Check the s position of the given IP 
    
    # We create a new column on our dataframe that accounts for the lattice centered at our IP
    # Basically maps all s around [-C/2, C/2]
    df["S_SHIFTED"] = (df["S"] - ip_s + circumference / 2) % circumference - circumference / 2
    
    # We select only those quadrupoles around the IP
    mask = (df["KEYWORD"] == "QUADRUPOLE") & (df["S_SHIFTED"].abs() <= window)

    return df[mask].sort_values("S_SHIFTED")

def get_qp(df):
    """ This function will get the quadrupoles around a given IP """

    mask = (df["KEYWORD"] == "QUADRUPOLE")

    return df[mask].sort_values("S")


# We will read the lines of our .seq file
input_file_path = args.input_file

# We read the twiss file using tfs 
df = tfs.read(input_file_path)

# We'll select only the quadrupoles and we'll check for the IPs 
filtered_df = get_elements_around_ip(df, args.IP, float(args.window))


# Now, we'll write the errors and corrections files 
e_f = open("IR_errors.madx",'w')
c_f = open("IR_errors+corrections.madx", 'w')

if args.random_flag:
    for _, QP in filtered_df.iterrows():

        name = QP["NAME"] # We get the name of the QP  

        # We'll write the errors for each of the quadrupoles 
        err = np.random.normal(0.0, float(args.random_sigma))
        print(f"{name}->K1 = {name}->K1 + {err};", file = e_f)
        print(f"{name}->K1 = {name}->K1 + {err};", file = c_f)
else:
    for _, QP in filtered_df.iterrows():
        
        name = QP["NAME"] # We get the name of the QP  

        # We'll write the errors for each of the quadrupoles 
        print(f"{name}, K1 := K1{name.split('.')[0]};", file = e_f)

e_f.close()


# q_f = open("QP_creator.madx", "w")
# qps_df = get_qp(df)
#
# for _, QP in qps_df.iterrows():
#
#     name = QP["NAME"]
#
#     print(f'PRINTF, TEXT="{name} length strength: %f, %f",VALUE= {name}->L, {name}->K1 ;', file = q_f) 

