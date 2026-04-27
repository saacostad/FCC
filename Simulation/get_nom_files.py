import sys
import os
import math
from scipy.integrate import quad
from subprocess import call
import argparse
import datetime

import tfs 
import pandas

version="Santiago's version 0.9"

# Version = "Santiago's version basically cleans and documents existing code + adds fcc_ee functionalities"]
#               but 0.9 just comments and tries to understand what is happening here

print("get_nom_files Version",version)

def beta_integral(L, alpha0, beta0,  KK, plane, beam):
    """calc beta av in foc magnet
    b ... beta at waist
    L ...  L*
    KK ... Quadrupole Gradient
    K ... Sqrt of Quadrupole Gradient
    KL ... Sqrt of Quadrupole Gradient times Quadrupole length
    """
    K = math.sqrt(abs(KK))
    KL = K*L
    b = beta0 / (1 + alpha0**2)
    if (((KK > 0) and (plane == '-x') and (beam == '1')) or ((KK < 0) and (plane == '-y') and (beam == '1')) or ((KK > 0) and (plane == '-y') and (beam == '2')) or ((KK < 0) and (plane == '-x') and (beam == '2'))):
        sin2KL = ((math.sin(2 * KL)) / (2 * KL))
        beta_av = 0.5 * beta0 * (1 + sin2KL) + alpha0 * ((math.sin(KL) ** 2) / (KL * K)) + (1 - sin2KL) / (2 * b * K ** 2)
    else:
        sinh2KL = ((math.sinh(2 * KL)) / (2 * KL))
        beta_av = 0.5 * beta0 * (1 + sinh2KL) + alpha0 * ((math.sinh(KL) ** 2) / (KL * K)) + (sinh2KL - 1) / (2 * b * K ** 2)
    beta_int = beta_av*L
    return beta_int

def beta_f(x, alpha0, beta0,  KK, plane, beam):
    """ 
    KK ... Quadrupole Gradient
    K ... Sqrt of Quadrupole Gradient
    """
    K = math.sqrt(abs(KK))
    gamma =  (1 + alpha0**2)/beta0
    if (((KK > 0) and (plane == '-x') and (beam == '1')) or ((KK < 0) and (plane == '-y') and (beam == '1')) or ((KK > 0) and (plane == '-y') and (beam == '2')) or ((KK < 0) and (plane == '-x') and (beam == '2'))):
        SS = (math.sin(K*x))/K
        CC = math.cos(K*x)
    else:
        SS = (math.sinh(K*x))/K
        CC = math.cosh(K*x)        
    return beta0*CC**2 +2*CC*SS*alpha0 + gamma*SS**2

def beta_sk(x,alphax, betax,alphay, betay):
    return math.sqrt((-2*alphax*x + betax)*(-2*alphay*x + betay))
    
def beta_integral2(L, alpha0, beta0,  KK, plane, beam):
    def beta_aux(x):
        return beta_f(x, alpha0, beta0,  KK, plane, beam)
    res, err = quad(beta_aux, 0, L)
    return res

def beta_int_skew(L, alphax, betax,alphay, betay):
    def beta_sk_aux(x):
        return beta_sk(x,alphax, betax,alphay, betay)
    res, err = quad(beta_sk_aux, 0, L)
    return res        

def betaxbetay_int(L, alphax, betax,alphay, betay, KK, beam):
    def betxbety_aux(x):
        return math.sqrt((beta_f(x, alphax, betax,  KK, '-x', beam))*(beta_f(x, alphay, betay,  KK, '-y', beam)))
    res, err = quad(betxbety_aux, 0, L)
    return res           

def rename_mag(magnet_name, accel = "fcc_ee"):
    mag = magnet_name.split('.')

    if accel == "fcc_ee":
        # So far, we do not need new names for our quadrupoles, so we just don't change them when working with the fcc
        return magnet_name
    
    newname = None
    if ((mag[0] =='MQXA' and mag[1][0] =='3') or (mag[0] =='MQXFA' and mag[1][1] =='3')): newname = 'Q3'+ mag[1][-2]+mag[1][-1]
    if ((mag[0] =='MQXA' and mag[1][0] =='1') or (mag[0] =='MQXFA' and mag[1][1] =='1')): newname = 'Q1'+ mag[1][-2]+mag[1][-1]
    if (mag[0] =='MQXB' or mag[0] =='MQXFB' ):newname = 'Q2'+ mag[1][-2]+mag[1][-1]
    if (mag[0] =='MQSX'): newname = 'MQSX'+'.'+mag[1]
    if (mag[0] == 'MQY' and mag[1][0] =='4'):newname = 'Q4' + mag[1][-2]+mag[1][-1]
    if (mag[0] == 'MQY' and mag[1][0] =='5'): newname = 'Q5' + mag[1][-2]+mag[1][-1]
    if (mag[0] == 'MQML' and mag[1][0] =='5'): newname = 'Q5' + mag[1][-2]+mag[1][-1]
    if (mag[0] == 'MQML' and mag[1][0] =='6'): newname = 'Q6' + mag[1][-2]+mag[1][-1]   
    return newname

parser = argparse.ArgumentParser()
parser.add_argument(
        "-b", "--beam",
        help="beam can be 1 or 2",
        choices=['1', '2'],
        required=True,
        dest="beam"
    )
parser.add_argument(
        "-id", "--input_dir",
        help="Directory where the original job.twiss.madx is",
        required=True,
        dest="input_dir"
    )
parser.add_argument(
        "-od", "--out_dir",
        help="Directory where all output files will be written",
        required=True,
        dest="out_dir"
    )
parser.add_argument(
        "-mx", "--mad_program",
        help="Location of madx executable",
        dest="mad_program",
        #default='/Users/jfcp/bin/madx507'
        default='/afs/cern.ch/user/m/mad/madx/releases/5.07.00/madx-linux64-gnu'
    )
parser.add_argument(
        "-a", "--accel",
        help="accelerator name",
        choices=['lhc','hl_lhc', 'fcc_ee'],
        dest="accel",
        default='lhc'
    )
parser.add_argument(
   "-nc", "--no_cern_afs",
    help="To be able to run job.twiss.madx from any system different to cern afs. Macros and modifiers are called from apj/lhc/madx_macros/ and apj/lhc/madx_modifiers/",
    action="store_true",
    dest="no_cern_afs"
)
parser.add_argument(
        "-sq", "--sequence",
        help="Name of the sequence to use",
        dest="sequence_name",
        default='LHCB'
    )
args = parser.parse_args()


apj_files_dir = None        # The directory of the given accelerator
if (args.accel == 'lhc'): apj_files_dir = os.path.dirname(sys.argv[0]) + '/lhc/'
if (args.accel == 'hl_lhc'):apj_files_dir= os.path.dirname(sys.argv[0]) + '/hl_lhc/'    
if (args.accel == 'fcc_ee'):apj_files_dir= os.path.dirname(sys.argv[0]) + '/fcc_ee/'    



input_dir = args.input_dir       # Where the data is alocated (the .seq with the data)
out_dir = args.out_dir                    # Where to output everything
accel_name = args.accel

# So far, we won't need it
# beam = args.beam                        # Beam to work with 


call(["mkdir","-p", out_dir])           # We create a new folder for the output 

dirty_twiss_name = None                 # The .madx file from where we'll start the simulation

# The madx file from where we'll get the simulation for the twiss files done
# match accel_name:
#     case "lhc":
#         # dirty_twiss_name = "job.twiss.madx"                       # In the case we want to work with the original
#         dirty_twiss_name = "get_real_twiss_data.madx"
#     case "hl_lhc":
#         # dirty_twiss_name = "job.twiss.madx"                       # In the case we want to work with the original
#         dirty_twiss_name = "get_real_twiss_data.madx"
#     case "fcc_ee":
#         dirty_twiss_name = "get_real_twiss_data.madx"               # We'll just try to leave everything in a default value 


if accel_name in ["lhc", "hl_lhc"]:
    dirty_twiss_name = "get_real_twiss_data.madx"
else:
    dirty_twiss_name = "get_real_twiss_data.madx"


# The original twiss file 
job_orig_file = input_dir + "/" + dirty_twiss_name                  



# This seems to be useless too 
# --------------------------------------------------------------------------------------------------------
# # Where we'll save the data
# acc_models_lhc=out_dir + f'acc-models-{accel_name}'
#
# # We'll create a symlink from one thing to another                                          <-
# call(["ln", "-s", original_model_dir + f"acc-models-{accel_name}", acc_models_lhc])

## Probably where we'll input our data up
# twiss_orig=original_model_dir + "/" + "twiss.dat"                                           <- get rid of this because twiss_orig is never used again

# --------------------------------------------------------------------------------------------------------

beam = args.beam
nominal_file= out_dir  + "nominal_lattice.madx"

# In the case we are using the fcc data, we already have most of it formatted and really do not need further more
if accel_name != "fcc_ee":
    
    # We'll copy the get_twiss_data.madx to the output directory 
    call(["cp",job_orig_file,out_dir])

    # We copy the .madx file to get the twiss back into the output dir                          
    job_file =  out_dir + dirty_twiss_name

    # From here we'll create the nominal lattice simulation
    template_file= apj_files_dir + 'b' + beam +'/' + 'nom.madx'
    nominal_file= out_dir  + "nominal_lattice.madx"

    # Probably where we'll save out data up                                                     <- 
    twiss_file = '"' + out_dir  + "twiss.dat" + '"'

    # Basic command to perform our twiss meditions
    twiss_command= f'exec, do_twiss_monitors({args.sequence_name}'+ beam + ',' +  twiss_file + ', 0.0);'


    # Opening files to read and write 
    jobf = open(job_file,'r')               # Our dirty twiss 
    templ=open(template_file,'r')           # Our nominal twiss

    nomf=open(nominal_file,'w')             # Where we'll write to



    # We'll check first our dirty twiss files 
    for line in jobf:

        # This is the condition that actually matters to us (running locally)
        # What this block does is to rewritte the dirty twiss.madx file but with our local directions 
        if (args.no_cern_afs):
                if ('madx_macros' in line):
                    # Here we'll simply make all the callables local to our machine: from jfcp -> local machine
                    lines = line.split('"')
                    macro_path = lines[1]
                    new_path = apj_files_dir +  'madx_macros/' + os.path.basename(macro_path)
                    print('call , file = "' + new_path + '";', file=nomf)

                    """
                        IMPORTANT: here, they call a lot of macros, but the only function they use from them is the 
                        `define_nominal_beams()` function on lhc.macros.madx, which basically defines the beams with protons, 
                        the given energy and bunches...
                        `cycle_sequences()` which tells the program to use a periodic sequence 
                        `set_default_crossing_scheme()` which, I think only tells the simulator we're on a no-experiments run
                    """

                elif ('madx_modifiers' in line):
                    # Same thing here
                    lines = line.split('"')
                    macro_path = lines[1]
                    new_path = apj_files_dir +  'madx_modifiers/' + os.path.basename(macro_path)
                    print('call , file = "' + new_path + '";', file=nomf)    

                    """ 
                        IMPORTANT: they do the same here, but this time, every modifier does something, which is basically modify or add 
                        information about the run (I think)
                    """

                elif ('twiss_ac' in line or 'twiss_adt' in line or 'twiss_elements' in line or 'twiss_monitors' in line):
                    if ('twiss_monitors' in line):
                        print(twiss_command, file=nomf)
                    else:
                        print('!', end=' ', file=nomf)
                        print(line.strip(), file=nomf)       
                else: print(line.strip(), file=nomf)
        else:
                if ('twiss_ac' in line or 'twiss_adt' in line or 'twiss_elements' in line or 'twiss_monitors' in line):
                    # Same thing but 
                    if ('twiss_monitors' in line):
                        print(twiss_command, file=nomf)
                    else:
                        print('!', end=' ', file=nomf)
                        print(line.strip(), file=nomf)       
                else: print(line.strip(), file=nomf)        

    jobf.close()
    nomf.close()


    # We copy the file we just created into get_real_twiss_data.madx
    call(['cp',nominal_file,out_dir + 'get_real_twiss_data.madx'])



    nomf=open(nominal_file,'a')

    for line in templ:
        # I think it basically rewrittes the information of the nominal.madx 
        # back into the new .madx we're building in the output directory. Apparently no changes at all 
        if ('twiss.optics' in line):
            new_file = out_dir + 'twiss.optics'
            newline=line.replace("twiss.optics",new_file)
        elif ('twiss_c.optics' in line):
            new_file = out_dir + 'twiss_c.optics'
            newline=line.replace("twiss_c.optics",new_file)
        elif ('twiss_shifted.dat' in line):
            new_file = out_dir  + 'twiss_shifted.dat'
            newline=line.replace("twiss_shifted.dat",new_file)
        elif ('Quad_KyL.txt' in line):
            new_file = out_dir  + 'Quad_KyL.txt'
            newline=line.replace("Quad_KyL.txt",new_file)
        elif ('my_model' in line):
            new_file = out_dir  + 'my_model'
            newline=line.replace("my_model",new_file)
        # elif ('fccee_integral.seq' in line):
        #     new_file = out_dir  + 'fccee_integral.seq'
        #     newline=line.replace("fccee_integral.seq",new_file)
        else:
            newline = line

        print(newline.strip(), file=nomf)           # It writes to the nomf file: nominal_lattice.madx

    templ.close()    
    nomf.close()

    
    print(nominal_file, " was created, running madx...")

else:
    # TODO: change automatically the CALL, FILE = "fccee_t.seq" line in the .madx file so it has no problems when doing this
    """ If we're working on the fcc, then we'll just copy the twiss file """
        
    print("Using FCC_ee configs")

    nominal_file= out_dir  + "nominal_lattice.madx"

    call(['cp', f'{input_dir}/b{beam}/{dirty_twiss_name}', f'{nominal_file}'])

    # Open the original for reading and a temp file for writing
    with open(nominal_file, 'r') as f_in, open(f"{nominal_file}.tmp", 'w') as f_out:
        for line in f_in:
            newline = line  # Start with the original line
            
            # Apply the changes
            if "fccee_t.seq" in line:
                newline = line.replace("fccee_t.seq", f'{input_dir}/b{beam}/fccee_t.seq')
            if "twiss.dat" in line:
                newline = line.replace("twiss.dat", f'{out_dir}twiss.dat')
            if "twiss_c.dat" in line:
                newline = line.replace("twiss_c.dat", f'{out_dir}twiss_c.dat')
            if "twiss.optics" in line:
                newline = line.replace("twiss.optics", f'{out_dir}twiss.optics')
            if "twiss_c.optics" in line:
                newline = line.replace("twiss_c.optics", f'{out_dir}twiss_c.optics')
            if "QP_creator.madx" in line:
                newline = line.replace("QP_creator.madx", f'{input_dir}/b{beam}/QP_creator.madx')
            

            # Write the processed line to the new file
            f_out.write(newline)

        # Now, we'll recreate the "my_model" file, which is just the measurements at observation points, this case, all the quadrupoles
        f_out.write("\n\n\n !MY_MODEL TWISS: creating the twiss at the observation points \n\n\n")
        f_out.write(f'call, file="{input_dir}/b{beam}/quad.obs_ptc.madx";')

        command = f'''
        select, flag=twiss, clear;
        select, flag=twiss,pattern="^Q.*",column=name,s,betx,mux,bety,muy,x,y,alfx,alfy;
        select, flag=twiss,pattern="^IP*",column=name,s,betx,mux,bety,muy,x,y,alfx,alfy;
        twiss, file="my_model";
        '''
        
        f_out.write(command)
    # Replace the old file with the new one
    os.replace(f"{nominal_file}.tmp", nominal_file)
    




# Get the madx executable
madx_exec=args.mad_program


# Dir stuff to know where we are
current_dir = os.getcwd()
os.chdir(out_dir)
# os.chdir(input_dir + f"/b{beam}/")
# Call madx on the nominal file we just created
print("Calling madx")
with open(out_dir + "madx.out", "wb") as madxout:  call([madx_exec, nominal_file], stdout=madxout)
print("madx ended")
os.chdir(current_dir)


""" 
    nominal_lattice.madx EXECUTION TREE 

    1. Dirty twiss 
        
        ->  Calls the macros and modifiers. The macros add callable functions, from where we obtain the beam, sequence and crossing schemes off 
        ->  Mathces tunes
        ->  Saves the twiss output data in `twiss.dat`

    2. Nominal twiss 
        
        -> It selects the sequence we'll work with 
        -> Changes the starting position 
        -> Turns off chromacity correctors 
        -> Saves these results on `twiss_shifted.dat`


    ==> The main differences between these last twiss outputs is that the the nominal twiss shiftes the start of the accelerator 

    3. Manual twiss 
        
        -> They create twiss.optics and twiss_shifted.optics, which are basically the same shifts but off all the elements and all params
        -> After that, they create the twiss output that actually matter to us: my model. Selects elements as monitors, quadrupoles and IPs, 
            and only the params worth of saving for APJ 
"""


# We'll select the output of the twiss files we just simulated
twissfilename= out_dir + "my_model"
latticefilename=out_dir + "lattice.asc"
twissfile = open(twissfilename,'r')
latticefile = open(latticefilename,'w')


# In this section, the my_model output (twiss files we're interested in) are cleaned: delete the header and all the tfs info

# Basically, it rewrittes the simulation's output in a format that the code can later read
for line in twissfile:
    sline = line.split(None)

    # Here we'll selecting like actual data, not just names and headers
    if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
        if ('*' in sline):#column names
            colname = sline.index('NAME')-1
            cols = sline.index('S')-1
            colbetx = sline.index('BETX')-1
            colbety = sline.index('BETY')-1
            colmux = sline.index('MUX')-1
            colmuy = sline.index('MUY')-1
            colalfx = sline.index('ALFX')-1
            colalfy = sline.index('ALFY')-1             
        else:
            # TODO little risky as it basically expect that the prior if block had been already run but it works so fuck it
            print(sline[colname], sline[cols],sline[colbetx],'0',sline[colbety],sline[colalfx],sline[colalfy],'0','0','0',sline[colmux],sline[colmuy], file=latticefile)
twissfile.close()
latticefile.close()


# Basically, `lattice.asc` is the file that we'll actually format, while `twiss.*` are not formatted


# Message
print("Archivo",latticefilename," was created")





"""
        INTEGRALS OF THE QUADRUPOLES
I don't really think this is going to be used for now for the FCC

-> It reads the quadrupoles physical parameters in Quad_KyL.txt (length and strength) of the quadrupoles 
    that were manually selected during the `nominal_lattice.madx` execution at the end
-> Get's the lattice parameters of the accelerator's elements 
-> Performs the integrals defined on top

"""

# TODO: I'll have to check if we want to do this for all the quadrupoles of the accelerator or what
QLyKf_file=out_dir  + "Quad_KyL.txt"                # Quadrupole integrals
integral_out=out_dir  + "integrals.dat"             # Quadrupole integrals already calcualted
QLyKf=open(QLyKf_file,'r')
fout = open(integral_out,'w')
# Variables to calculate everything
name=[]
length=[]
strength=[]



# In this part, we save the physical parameters of the quadrupoles 
for i in QLyKf:
    il = i.split(None)
    if (len(il) > 0):
        if ('MQ' in il[0]) or ('Q' in il[0]):
            name.append(il[0])
            length.append(float(il[3].strip(',')))
            strength.append(float(il[4]))

twiss_optics=out_dir  + "twiss.dat"              # twiss.optics
twiss=open(twiss_optics,'r')



# And here, we save the lattice parameters of ALL the elements of the accelerator
namet=[]
ss=[]
betx=[]
mux=[]
bety=[]
muy=[]
alfx=[]
alfy=[]
flag=0

for j in twiss:
    jl=j.split(None)
    if (flag):
        namet.append(jl[0].strip('"').split(".")[0])
        ss.append(float(jl[1]))
        betx.append(float(jl[2]))
        alfx.append(float(jl[8]))
        bety.append(float(jl[4]))
        alfy.append(float(jl[9]))
        mux.append(float(jl[3]))
        muy.append(float(jl[5]))

    # This conditional will basically position us on the actual data
    if ('$' in jl[0]): flag = 1

twiss.close()


# Here we perform different calculations that, so far, I think are not that needed                                                      <-
for k in name:
    # k = k.split(".")[0] 
    # TODO this seems to be specifically for the LHC's formatting
    if ('MQSX' in k ):
        beta_intx = beta_int_skew(length[name.index(k)], alfx[namet.index(k)], betx[namet.index(k)], alfy[namet.index(k)], bety[namet.index(k)])
        beta_inty = beta_intx
        betxbety = beta_intx
    else:
        #print k, strength[name.index(k)]
        if (strength[name.index(k)] == 0):
            beta_intx = 0
            beta_inty = 0
            betxbety = 0
        else:            
            beta_intx = beta_integral2(length[name.index(k)], alfx[namet.index(k.split(".")[0])], betx[namet.index(k.split(".")[0])], strength[name.index(k)], '-x', beam)
            beta_inty = beta_integral2(length[name.index(k)], alfy[namet.index(k.split(".")[0])], bety[namet.index(k.split(".")[0])], strength[name.index(k)], '-y', beam)
            betxbety =  betaxbetay_int(length[name.index(k)], alfx[namet.index(k.split(".")[0])], betx[namet.index(k.split(".")[0])], alfy[namet.index(k.split(".")[0])], bety[namet.index(k.split(".")[0])],strength[name.index(k)], beam)
    k2 = '"'+k+'"'
    print(k2,  beta_intx, beta_inty, rename_mag(k), mux[namet.index(k.split(".")[0])], muy[namet.index(k.split(".")[0])], betxbety, file=fout)


QLyKf.close()
fout.close()
print(integral_out, ' was created')


"""
        OPTICS TO MAKE CUTE GRAPHS
Here, we'll get the lenght (in and out positions) of different elements 
so we can create the rectangle graphs we usually see 
"""


twiss_optics=out_dir  + "twiss.optics"
twiss_optics_c=out_dir  + "twiss_c.optics"
twiss=open(twiss_optics,'r')
twiss_c=open(twiss_optics_c,'r')

optics_file= out_dir  + 'optics.out'
opt=open(optics_file,'w')

for line in twiss:

    if ('@' in line) or ('*' in line) or ('$' in line) : 
        line_c = twiss_c.readline()
        continue 
   
    line_c = twiss_c.readline()
    lines=line.split()
    line_cs=line_c.split()

    if ('MB' in line) or ('B' in line):
        s_out = float(lines[1])
        s_c = float(line_cs[1])
        s_in = s_c - (s_out -s_c)
        print(lines[0], s_in, 0, file=opt)
        print(lines[0], s_in, 1, file=opt)
        print(lines[0], s_out, 1, file=opt)
        print(lines[0], s_out, 0, file=opt)

    if ('MQ' in line) or ('Q' in line):
        s_out = float(lines[1])
        s_c = float(line_cs[1])
        s_in = s_c - (s_out -s_c)
        print(lines[0], s_in, 0, file=opt)
        print(lines[0], s_in, 2, file=opt)
        print(lines[0], s_out, 2, file=opt)
        print(lines[0], s_out, 0, file=opt)

    if ('BPM' in line):
        s_out = float(lines[1])
        s_c = float(line_cs[1])
        s_in = s_c - (s_out -s_c)
        print(lines[0], s_in, 0, file=opt)
        print(lines[0], s_in, 0.25, file=opt)
        print(lines[0], s_out, 0.25, file=opt)
        print(lines[0], s_out, 0, file=opt)

twiss.close()
twiss_c.close()

print(optics_file, ' was created \n')

version_file = open(out_dir + 'version_get_nom_files.dat','a')
print("Version", version, "timestamp", datetime.datetime.now(), file=version_file)
version_file.close()



# TODO mymodel should have the bending magnets too
