import sys
import os
from subprocess import call
import argparse
from scipy.optimize import fsolve
import numpy as np
from random import gauss
from tools.utils_ActPhase10 import *
import datetime


version= "Santiago's version 0.9"

print("tbt_simulator Version", version)

def multiturn2sddsnew(in_file,out_file):
    """ This function takes an input tbt simulation file from madx and 
    converts it to a format that can be worked with in this script """

    in_tracks = open(in_file,'r')

    # We create some tables to save up data
    bpms = []
    turn = []
    x = []
    y = []
    s = []
    
    # We define these variable to avoid having errors on the IDE 
    colturn = colx = coly = cols = None 

    for line in in_tracks:
        # We start to read the file 
        sline = line.split(None)

        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
        # Skip the headers 

            if '*' in sline:
                # Save the indexes of each one of the data of interest
                colturn = sline.index("TURN")-1
                colx = sline.index("X")-1
                coly = sline.index("Y")-1
                cols = sline.index("S")-1

            else:
                if ("#segment" in sline):
                    # Here, we save the bpm names for later use 
                    bpms.append(sline[5].upper())
                else:
                    # Or simply add the data of interest of each measurement
                    turn.append(int(sline[colturn]))
                    x.append(float(sline[colx])*1000.0)
                    y.append(float(sline[coly])*1000.0)
                    s.append(sline[cols])

    
    # Eliminate the start of the measurements as it is just no-needed
    bpms.pop(0)
    turn.pop(0)
    x.pop(0)
    y.pop(0)
    s.pop(0)

    print(len(bpms),"lines read from", in_file)     # Debug

    # Now, for each one of the bpms, we'll save their names and s position
    # and later, their x measurements for each turn
    setS = []
    salida = []
    
    # Here, we're basically saving up the unique BPM data (convert from a list to a set)
    for i in range(len(s)):
        if not(s[i] in setS):
            setS.append(s[i])
            salida.append(['0',bpms[i],s[i]])

    # Now we append the x measurement of each turn do a "dictionary" of BPMs
    for i in range(len(s)):
        indice = setS.index(s[i])
        salida[indice].insert(turn[i]+3,x[i])

    # Clean undesired data and save everything on another data structure
    salida.pop()
    salidax = salida[:]



    # Here we'll do exactly the same, but this time for the y axis
    setS = []
    salida = []
    for i in range(len(s)):
        if not(s[i] in setS):
            setS.append(s[i])
            salida.append(['1',bpms[i],s[i]])

    for i in range(len(s)):
        indice = setS.index(s[i])
        salida[indice].insert(turn[i]+3,y[i])

    salida.pop()
    saliday = salida[:]
    
    # Create a new structure with all the data we just created
    salidatot=salidax+saliday

    # We'll write this information on a file with a custom format
    out_tracks = open(out_file,'w')
    print("# title", file=out_tracks)
    for linea in salidatot:
        for slin in linea:
            print(slin, end=' ', file=out_tracks)
        print(file=out_tracks) # End of line 

    print("File",out_file ,"created")
    in_tracks.close()
    out_tracks.close()   

    # We return having already written to the new file
    return


parser = argparse.ArgumentParser()
parser.add_argument(
        "-b", "--beam",
        help="beam can be 1 or 2",
        choices=['1', '2'],
        required=True,
        dest="beam"
    )
parser.add_argument(
        "-ip", "--ip",
        help="ip can be 1, 2, 5 or 8",
        choices=['1', '2', '5', '8'],
        required=True,
        dest="ip"
    )
parser.add_argument(
        "-md", "--model_dir",
        help="Directory where nominal model is",
        required=True,
        dest="model_dir"
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
        "-n", "--number_turns",
        help="Number of turns for the simulated tbt. If -td is activated, the number of turns is one",
        dest="number_turns",
        default='100'
    )
parser.add_argument(
        "-ef", "--IR_errors",
        help="File with IR magnetic errors",
        required=True,
        dest="IR_errors",
    )
#parser.add_argument(
#       "-sh", "--shift_tbt",
#        help="generate TBT with each turn starting at ac-dipole location. Avermax always starts at ac-dipole location regardless of this flag",
#        action="store_true",
#        dest="shift_tbt"
#    )
parser.add_argument(
       "-tw", "--twiss",
        help="If twis is True, twiss files with errors are generated",
        action="store_true",
        dest="twiss"
    )
parser.add_argument(
        "-td", "--tbt_dir",
        help="Directory where experimental tbt to be compared with simu can be found",
        dest="tbt_dir"
    )
parser.add_argument(
        "-a", "--accel",
        help="accelerator name",
        choices=['lhc','hl_lhc', "fcc_ee"],
        dest="accel",
        default='fcc_ee'
    )

args = parser.parse_args()

acc_name = args.accel
apj_files_dir = None        # Where the model files are located
if (args.accel == 'lhc'): apj_files_dir = os.path.dirname(sys.argv[0]) + '/lhc/'
if (args.accel == 'hl_lhc'):apj_files_dir= os.path.dirname(sys.argv[0]) + '/hl_lhc/'    
if (args.accel == 'fcc_ee'):apj_files_dir= os.path.dirname(sys.argv[0]) + '/fcc_ee/'    

model_dir=args.model_dir
beam = args.beam
ip = args.ip

out_dir=args.out_dir
madx_exec=args.mad_program
IR_errors=args.IR_errors


# These are the lenght of the closest part of the closests QP to the IP
# TODO check this for the FCC_EE
if (args.accel == 'lhc'):
    lipl1 = 21.579
    lipr1 = 21.564
    lipl5 = 21.564
    lipr5 = 21.564
if (args.accel == 'hl_lhc'):
    lipl1 = 21.853
    lipr1 = 21.853 
    lipl5 = 21.853
    lipr5 = 21.853    
if (args.accel == 'fcc_ee'):
    # We need to check this later
    lipl1 = 21.853
    lipr1 = 21.853 
    lipl5 = 21.853
    lipr5 = 21.853    

lipl2 = 21.595
lipr2 = 21.595

lipl8 = 21.595
lipr8 = 21.595

if (ip == '5'):
    lipl = lipl5
    lipr = lipr5
if (ip == '1'):
    lipl = lipl1
    lipr = lipr1    
if (ip == '2'):
    lipl = lipl2
    lipr = lipr2    
if (ip == '8'):
    lipl = lipl8
    lipr = lipr8   



# These are equations and the solver to find the action and phase on the IP 
# given the betas on the QP and the waist
def equations(p,betl,betr):
    bwv, wv = p
    return (bwv-betl+((lipl+wv)**2)/bwv,bwv-betr+((lipr-wv)**2)/bwv)

def ipvalues_exact(phi_l,betl,betr):
    bw,w = fsolve(equations, (bw_guess, 0.01), args=(betl,betr))
    return bw, w




# We create the output folder
call(["mkdir","-p", out_dir])


# We'll save up the model we used, which for each accel, will be different 
job_file = None 
if acc_name == "fcc_ee":
    job_file=model_dir + "nominal_lattice.madx"
else:
    job_file=model_dir + "/" + "job.twiss_out.madx"


# This is the file that is in charge of doing the tbt simulation
# TODO Somehow we have to make the errors thing for the FCC_ee
template_file = apj_files_dir + 'b'+ beam +'/' + 'TBT_from_PTC3.madx'
simu_file = out_dir+ "/" + "tbt_simulator.madx"

# Create a folder for the tbt simulations
if(args.tbt_dir != None):
    ic_dir_i=args.tbt_dir
    dirs_and_names=ic_dir_i.split('/')
    name_dir=dirs_and_names[-2]
    sim_tbt_dir =  out_dir+ name_dir + "/"
else:
    sim_tbt_dir =  out_dir+ "sim_tbt/"

# Symbolic link just because idk
call(["ln","-s",model_dir + "/" + 'acc-models-lhc',out_dir + 'acc-models-lhc'])


# Create the folder with the very raw tbt data
call(['mkdir', '-p', sim_tbt_dir])

jobf = open(job_file,'r')       # The fixed twiss simulator
templ=open(template_file,'r')   # The original tbt simulator 
simuf=open(simu_file,'w')       # Where we'll write


# We're copying the twiss file to the simulator output
for line in jobf:
    if ('twiss_ac' in line or 'twiss_adt' in line or 'twiss_elements' in line or 'twiss_monitors' in line):
        # Basically, deactivating the macros
        print('!', end=' ', file=simuf)
        print(line.strip(), file=simuf)       
    else:
        print(line.strip(), file=simuf)
jobf.close()



for line in templ:

    # We check for the lines where we place the observers at the BPMs
    if ('bpm.obs_ptc.madx' in line):
        new_file = apj_files_dir + '/b'+beam+ '/' + 'bpm.obs_ptc.madx'
        bpm_obs_file= new_file
        newline=line.replace("bpm.obs_ptc.madx",new_file)
    if ('quad.obs_ptc.madx' in line):
        # In the case we're using the quadrupoles
        new_file = apj_files_dir + '/b'+beam+ '/' + 'quad.obs_ptc.madx'
        bpm_obs_file= new_file
        newline = line.replace("quad.obs_ptc.madx",new_file)

    elif ('IR_errors.madx' in line):
        # The file where we'll place the errors
        new_file = IR_errors 
        newline=line.replace("IR_errors.madx",new_file)

    # TODO no idea what this is
    elif ('icx.madx' in line):
        if (args.tbt_dir != None):
            old_file = args.tbt_dir + 'icx.madx'
            new_file =  out_dir + '/'  + 'icx.madx'
            call(['cp','-f',old_file,new_file])
            newline=line.replace('icx.madx',new_file)
        else:
            newline = ''
    elif ('icy.madx' in line):
        if (args.tbt_dir != None):
            old_file = args.tbt_dir + 'icy.madx'
            new_file =  out_dir + '/'  + 'icy.madx'
            call(['cp','-f',old_file,new_file])
            newline=line.replace('icy.madx',new_file)
        else:
            newline = ''

    elif ('onetable' in line):
        # Change formatting of the original tbt simulator
        if (args.tbt_dir != None):
           new_file = "onetable, extension=H"
           newline=line.replace('onetable',new_file)
        else:
            newline = line

    elif ('ffile' in line):
        # Get the track file
        track_out = out_dir + 'track'
        if (args.tbt_dir != None):
           newline ='turns = 1, file = " '+track_out+' ", ffile=1, norm_no=1,'
        else:
           newline = 'turns=' + args.number_turns + ',file="'+track_out+'", ffile=1, norm_no=1,'

    else:
        # Default case
        newline = line
    
    # Write the tbt simulator line to a new file
    print(newline.strip(), file=simuf)

# If no more info, just stop
if (args.tbt_dir == None and  not args.twiss): print('stop;', file=simuf)

#To generate avermax according to initial conditions
if (args.tbt_dir != None):        
    print('PTC_CREATE_UNIVERSE;', file=simuf)
    print('PTC_CREATE_LAYOUT,model=2,method=6, nst=10;', file=simuf)
    print('PTC_START,x =zx, px =zpx, y = zy, py = zpy ;', file=simuf)
    print('call, file="'+ bpm_obs_file + '";', file=simuf) 
    print('PTC_TRACK,icase=4,  CLOSED_ORBIT=false, dump,', file=simuf) 
    print('element_by_element,', file=simuf)
    print('turns=1, file="'+ track_out + '", ffile=1, norm_no=1,', file=simuf)
    print("onetable, extension=V;", file=simuf)
    print('PTC_TRACK_END;', file=simuf)
    print('PTC_END;', file=simuf)
    if (not args.twiss): print('stop;', file=simuf)
    
# To generate twiss with errors (no-apj) 
if (args.twiss):
    print('select, flag=twiss, clear;', file=simuf)
    print('select, flag=twiss,column=name,s,betx,mux,bety,muy,x,y,alfx,alfy;', file=simuf)
    err_optics = out_dir + 'twiss_err.optics'
    print('twiss, file="'+err_optics+'";', file=simuf)
    print('select, flag=twiss, clear;', file=simuf)
    print('select, flag=twiss,class=monitor,column=name,s,betx,mux,bety,muy,x,y,alfx,alfy;', file=simuf)
    print('select, flag=twiss,pattern="^MQSX.*",column=name,s,betx,mux,bety,muy,x,y;', file=simuf)
    print('select, flag=twiss,pattern="^IP*",column=name,s,betx,mux,bety,muy,x,y;', file=simuf)
    err_model =  out_dir + 'my_model_err'
    print('twiss, file="'+err_model+'";', file=simuf)
    print('stop;', file=simuf)

simuf.close()


# Here we call the tbt simulation madx
current_dir=os.getcwd()
os.chdir(out_dir)
with open(out_dir + "madx.out", "wb") as madxout:  call([madx_exec, simu_file], stdout=madxout)

# Go back to the working Directory
os.chdir(current_dir)

# Basically, we save the twiss with errors file into a different format
if (args.twiss):
    twissfilename = out_dir + "my_model_err"
    latticefilename = out_dir + "lattice_err.asc"
    twissfile = open(twissfilename,'r')
    latticefile = open(latticefilename,'w')
    for line in twissfile:
        sline = line.split(None)
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
                print(sline[colname], sline[cols],sline[colbetx],'0',sline[colbety],sline[colalfx],sline[colalfy],'0','0','0',sline[colmux],sline[colmuy], file=latticefile)
    twissfile.close()
    latticefile.close()
    print("File",latticefilename,"created\n")



    """ Create measurements of twiss at the IP """
    nominallattice = model_dir + 'lattice.asc'
    
    # Read the twiss files (lattice files) using functions on utils_ActPhase10
    nameeln,seln,betzn,psizn,alfzn = leer_beta_mu3(nominallattice,'-x')         # Nominal lattice 
    nameel,sel,betz,psiz,alfz=leer_beta_mu3(latticefilename,'-x')               # Lattice with errors

    bpmsw1l1 = '"'+'BPMSW.1L'+ ip +'.B'+beam+'"'
    bpmsw1r1 = '"'+'BPMSW.1R'+ ip +'.B'+beam+'"'

    
    # TODO check what to do for fcc_ee
    if (args.accel == 'hl_lhc' and (args.ip == '1' or args.ip == '5')):
        bpmsw1l1 = '"'+'BPMSQ.1L'+ ip +'.B'+beam+'"'
        bpmsw1r1 = '"'+'BPMSQ.1R'+ ip +'.B'+beam+'"'        
    
    if (args.accel == "fcc_ee"):
        print("Checking fcc_ee") 
    else:
        # Get data from the files that will be used to get the measurement of beta_waist for x axis
        betllx = betz[nameel.index(bpmsw1l1)]*(1.000)
        betrrx = betz[nameel.index(bpmsw1r1)]
        phi_llx = psiz[nameel.index(bpmsw1l1)]/(2*np.pi)
        phi_rrx = psiz[nameel.index(bpmsw1r1)]/(2*np.pi)
        bw_guess = betzn[nameeln.index('"IP'+ip + '"')]
        
        # From the guess of beta_waist, solve the equations to find bw
        bwx,wx = fsolve(equations, (bw_guess, 0.01), args=(betllx,betrrx))

        # And from these, find the beta at the IP 
        # TODO for x axis this betaipx calculation is done BEFORE the gauss. For y axis it is after
        betaipx = bwx + (wx**2)/bwx
        
        # TODO maybe doing something to check for the error???
        sigma_w = 0
        wx = wx + gauss(0,sigma_w)
        sigma_bw = 0
        bwx = bwx + gauss(0,sigma_bw)

        
        # Do the exact same thing but for the y axis
        nameel,sel,betz,psiz,alfz=leer_beta_mu3(latticefilename,'-y')
        betlly = betz[nameel.index(bpmsw1l1)]*(1.000)
        betrry = betz[nameel.index(bpmsw1r1)]
        phi_lly = psiz[nameel.index(bpmsw1l1)]/(2*np.pi)
        phi_rry = psiz[nameel.index(bpmsw1r1)]/(2*np.pi)
        bw_guess = betzn[nameeln.index('"IP'+ip + '"')]
        bwy,wy = fsolve(equations, (bw_guess, 0.01), args=(betlly,betrry))
        wy = wy + gauss(0,sigma_w)
        bwy = bwy + gauss(0,sigma_bw)
        betaipy = bwy + (wy**2)/bwy
        
        # Write these results in a format we can handle
        outp = open(out_dir + 'ip.results','w')
        print('* LABEL		  BETASTAR   	  BETASTAR_ERR      WAIST          WAIST_ERR      BETAWAIST   	BETAWAIST_ERR', file=outp) 
        print('$ %s                 %le               %le            %le              %le           %le              %le     ', file=outp)


    # Now we check the quadrupoles physical parameters
    QLyKf_file = model_dir  + "Quad_KyL.txt"
    QLyKf=open(QLyKf_file,'r')

    
    # We'll save the quadrupoles data 
    nameq=[]
    length=[]
    strength=[]
    for i in QLyKf:
        il = i.split(None)
        if (len(il) > 0):         # This rh is for the FCC
            if ('MQ' in il[0]) or ('Q' in il[0]):
                nameq.append(il[0])
                length.append(float(il[3].strip(',')))
                strength.append(float(il[4]))
    QLyKf.close()
    
    # TODO check which QP we need for the fcc 
    if (args.accel == 'lhc'): strengthq1l = float(strength[nameq.index('MQXA.1L' + ip)])
    if (args.accel == 'hl_lhc' and (args.ip == '5' or args.ip == '1')): strengthq1l = float(strength[nameq.index('MQXFA.A1L' + ip)])
    if (args.accel == 'fcc_ee'): 
        number_from_ip = None 
        if ip in [1, 8]:
            number_from_ip = 4
        elif ip in [2, 3]:
            number_from_ip = 1 
        elif ip in [4, 5]:
            number_from_ip = 2 
        elif ip in [6, 7]:
            number_from_ip = 3

        strengthq1l = float(strength[nameq.index(f"QC1L1.{number_from_ip}")])

    # Check if the qp is F or D 
    if(strengthq1l < 0 ):
        if (beam == '1'):
            wx_lhc = wx
            wy_lhc = -wy
        if (beam == '2'):
            wx_lhc = -wx
            wy_lhc = wy

    if(strengthq1l > 0 ):
        if (beam == '1'):
            wx_lhc = -wx
            wy_lhc = wy
        if (beam == '2'):
            wx_lhc = wx
            wy_lhc = -wy


    outx = 'ip'+ip+'b'+beam+'.X' + '\t\t' + str(betaipx) + '\t\t' + '0'+ '\t' + str(wx_lhc) + '\t\t' + '0'+ '\t' +str(bwx)  + '\t\t' + '0'
    outy = 'ip'+ip+'b'+beam+'.Y' + '\t\t' + str(betaipy) + '\t\t' + '0'+ '\t' + str(wy_lhc) + '\t\t' + '0'+ '\t' +str(bwy)  + '\t\t' + '0'

    print(outx, file=outp)
    print(outy, file=outp)
    print("File",out_dir + 'ip.results',"created\n")

    outp.close()
else:
    # If we do not want the laticce functions of the error laticce, then we just delete them
    call(['rm', '-f', out_dir + 'lattice_err.asc'])
    call(['rm', '-f', out_dir + 'ip.results'])
    call(['rm', '-f', out_dir + 'twiss_err.optics'])
    call(['rm', '-f', out_dir + 'twiss_err.optics'])
    call(['rm', '-f', out_dir + 'my_model_err'])
    

# Here we just convert the tracking files to the format we actually use 
if (args.tbt_dir != None):    #To generate avermax according to initial conditions
    in_filex =  out_dir + "trackoneh"
    out_filex = sim_tbt_dir + "trackoneh.sdds.new"
    in_filey =  out_dir + "trackonev"
    out_filey = sim_tbt_dir + "trackonev.sdds.new"

    multiturn2sddsnew(in_filex,out_filex)
    print("\n")

    multiturn2sddsnew(in_filey,out_filey)
    print("\n")

    trackx=[line.split(None) for line in open(out_filex)]
    tracky=[line.split(None) for line in open(out_filey)]
    
    # We re-format the output so both axis are saved to a single file
    track_f= open(sim_tbt_dir + 'avermax.sdds.new','w')
    for i in range(len(trackx)):
        if i > 0:
            print(trackx[i][0], trackx[i][1], trackx[i][2],trackx[i][3], tracky[i][3], file=track_f)
    track_f.close()

    print("File", sim_tbt_dir + 'avermax.sdds.new', "created")

else:
    # In the case we only want to create the tbt data
    in_file =  out_dir + 'trackone'
    out_file = sim_tbt_dir + 'sim_tbt.sdds.new'
    multiturn2sddsnew(in_file,out_file)
    print("\n")


templ.close()    


version_file = open(out_dir + 'version_tbt_simulator.dat','a')
print("Version", version, "timestamp", datetime.datetime.now(), file=version_file)
version_file.close()


