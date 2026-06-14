import tfs as tfs 
import numpy as np
import matplotlib.pyplot as plt

"""
This script will calculate the beta-beating of the lattice before and
after applying the corrections.
"""


nominal = "my_model"
apj = "my_model_err"


before_path = "fcc_ee_test_simulation/tbt/"
after_path = "fcc_ee_corrections/tbt/"


errors_path_nominal = before_path + nominal
errors_path_apj = before_path + apj 

corrections_path_nominal = after_path + nominal
corrections_path_apj = after_path + apj 


def calc_beta_beating(nom_path, apj_path):
    """ This function takes the nominal and experimental twiss files and calculates the beta beating """

    nom = tfs.read_tfs(nom_path)
    
    nom_s = nom["S"]
    nom_beta_x = nom["BETX"] 
    nom_beta_y = nom["BETY"]



    apj = tfs.read_tfs(apj_path)
    
    apj_s = apj["S"]
    apj_beta_x = apj["BETX"] 
    apj_beta_y = apj["BETY"]


    beta_beating_x = (apj_beta_x - nom_beta_x) / nom_beta_x
    beta_beating_y = (apj_beta_y - nom_beta_y) / nom_beta_y

    return beta_beating_x, beta_beating_y, nom_s


bx_bef, by_bef, sb = calc_beta_beating(errors_path_nominal, errors_path_apj)
bx_aft, by_aft, sa = calc_beta_beating(corrections_path_nominal, corrections_path_apj)


plt.title(r"$\beta$-Beating en eje X")
plt.plot(sb, bx_bef, label = "Before corrections")
plt.plot(sa, bx_aft, label = "After corrections")
plt.grid()
plt.legend()
plt.xlabel("s [m]")
plt.ylabel(r"$\beta$-beating")
plt.show()


plt.title(r"$\beta$-Beating en eje Y")
plt.plot(sb, by_bef, label = "Before corrections")
plt.plot(sa, by_aft, label = "After corrections")
plt.grid()
plt.legend()
plt.xlabel("s [m]")
plt.ylabel(r"$\beta$-beating")
plt.show()
