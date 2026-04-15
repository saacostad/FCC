import sys
import os
import math
from math import pi
from scipy.integrate import quad
from subprocess import call
import argparse
from scipy.optimize import fsolve
import numpy as np
from random import gauss
from utils_ActPhase10 import *
# from sets import Set
import datetime

#for path in sys.path:
#    print path
version=5
#Version 5 to avoid names with @ inside madx files. March 5 2024
#Version 4 to correct bug related to creation of output directory and lower and upercases in filenames. March 4 2024.
#Version 3 to ln to acc-models-lhc and to run madx tbt_simulator.madx from the directory  where ln to acc-models-lhc was created. March 4 2024
#Version 2 to redifine outpu_dir, default for madx program and location of apj_files. Feb 28 2024
#Version 1 to include the fact that the sign of the waist depends on the polarity of triplet quadrupoles rather than the IP itself. February 22 2024
#Version 0 was copied  from /Users/jfcp/bin/tbt_simulator6 on February 13/2024.
print("tbt_simulator Version", version)

def multiturn2sddsnew(in_file,out_file):
    in_tracks = open(in_file,'r')
    #print "Warning:", in_file,  "in (m),", out_file, "in (mm)."

    bpms = []
    turn = []
    x = []
    y = []
    s = []
    for line in in_tracks:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            if '*' in sline:
                colturn = sline.index("TURN")-1
                colx = sline.index("X")-1
                coly = sline.index("Y")-1
                cols = sline.index("S")-1
            else:
                if ("#segment" in sline):
                    bpms.append(sline[5].upper())
                    #print sline[5],
                    #print sline[5]
                else:
                    #print sline
                    #print sline[1]
                    turn.append(int(sline[colturn]))
                    x.append(float(sline[colx])*1000.0)
                    y.append(float(sline[coly])*1000.0)
                    s.append(sline[cols])
    #Eliminar start
    bpms.pop(0)
    turn.pop(0)
    x.pop(0)
    y.pop(0)
    s.pop(0)
    print(len(bpms),"lines read from", in_file)

    #Salida x
    setS = []
    salida = []
    for i in range(len(s)):
        if not(s[i] in setS):
            setS.append(s[i])
            salida.append(['0',bpms[i],s[i]])
    #Colocar cada vuelta
    for i in range(len(s)):
        indice = setS.index(s[i])
        salida[indice].insert(turn[i]+3,x[i])
    #Eliminar lch b1 end
    salida.pop()
    #print len(salida)
    salidax = salida[:]

    #Salida y
    setS = []
    salida = []
    for i in range(len(s)):
        if not(s[i] in setS):
            setS.append(s[i])
            salida.append(['1',bpms[i],s[i]])
    #Colocar cada vuelta
    for i in range(len(s)):
        indice = setS.index(s[i])
        salida[indice].insert(turn[i]+3,y[i])
    #Eliminar lch b1 end
    salida.pop()
    #print len(salida)
    saliday = salida[:]

    salidatot=salidax+saliday

    out_tracks = open(out_file,'w')
    print("# title", file=out_tracks)
    for linea in salidatot:
        for slin in linea:
            print(slin, end=' ', file=out_tracks)
        print(file=out_tracks) #, '\n'
    print("File",out_file ,"created")
    in_tracks.close()
    out_tracks.close()    
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
        choices=['lhc','hl_lhc'],
        dest="accel",
        default='lhc'
    )

args = parser.parse_args()

#if (args.accel == 'lhc'): apj_files_dir='/Users/jfcp/apj/lhc/'
#if (args.accel == 'hl_lhc'):apj_files_dir='/Users/jfcp/hl-lhc/local_model/' 

if (args.accel == 'lhc'): apj_files_dir = os.path.dirname(sys.argv[0]) + '/lhc/'
if (args.accel == 'hl_lhc'):apj_files_dir= os.path.dirname(sys.argv[0]) + '/hl_lhc/'    

model_dir=args.model_dir
beam = args.beam
ip = args.ip

out_dir=args.out_dir
madx_exec=args.mad_program
IR_errors=args.IR_errors


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

    
def equations(p,betl,betr):
    bwv, wv = p
    return (bwv-betl+((lipl+wv)**2)/bwv,bwv-betr+((lipr-wv)**2)/bwv)

def ipvalues_exact(phi_l,betl,betr):
    bw,w = fsolve(equations, (bw_guess, 0.01), args=(betl,betr))
    return bw, w



call(["mkdir","-p", out_dir])

job_file=model_dir + "/" + "job.twiss_out.madx"
template_file=apj_files_dir + 'b'+ beam +'/' + 'TBT_from_PTC3.madx'
simu_file= out_dir+ "/" + "tbt_simulator.madx"
if(args.tbt_dir != None):
    ic_dir_i=args.tbt_dir
    #ic_dir_i_c=ic_dir_i.split('.')
    #dirs_and_names=ic_dir_i_c[0].split('/')
    dirs_and_names=ic_dir_i.split('/')
    name_dir=dirs_and_names[-2]
    sim_tbt_dir =  out_dir+ name_dir + "/"
else:
    sim_tbt_dir =  out_dir+ "sim_tbt/"
call(["ln","-s",model_dir + "/" + 'acc-models-lhc',out_dir + 'acc-models-lhc'])




    
call(['mkdir', '-p', sim_tbt_dir])
jobf = open(job_file,'r')
templ=open(template_file,'r')
simuf=open(simu_file,'w')
for line in jobf:
    if ('twiss_ac' in line or 'twiss_adt' in line or 'twiss_elements' in line or 'twiss_monitors' in line):
            print('!', end=' ', file=simuf)
            print(line.strip(), file=simuf)       
    else: print(line.strip(), file=simuf)
jobf.close()



for line in templ:
    if ('bpm.obs_ptc.madx' in line):
        new_file = apj_files_dir + '/b'+beam+ '/' + 'bpm.obs_ptc.madx'
        bpm_obs_file= new_file
        newline=line.replace("bpm.obs_ptc.madx",new_file)
    elif ('IR_errors.madx' in line):
        new_file = IR_errors 
        newline=line.replace("IR_errors.madx",new_file)
    #elif ('BPM.8L4.B1' in line):
    #    if (args.shift_tbt or args.tbt_dir != None):
    #        newline = line
    #    else:
    #        new_file = 'MSIA.EXIT.B1'
     #       newline=line.replace('BPM.8L4.B1',new_file)
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
        if (args.tbt_dir != None):
           new_file = "onetable, extension=H"
           newline=line.replace('onetable',new_file)
        else:
            newline = line
    elif ('ffile' in line):
        track_out= out_dir + 'track'
        if (args.tbt_dir != None):
           newline ='turns=1, file="'+track_out+'", ffile=1, norm_no=1,'
        else:
           newline = 'turns=' + args.number_turns + ',file="'+track_out+'", ffile=1, norm_no=1,'

    else:
        newline = line
    print(newline.strip(), file=simuf)
if (args.tbt_dir == None and  not args.twiss): print('stop;', file=simuf)

if (args.tbt_dir != None):    #To generate avermax according to initial conditions
    print('PTC_CREATE_UNIVERSE;', file=simuf)
    print('PTC_CREATE_LAYOUT,model=2,method=6, nst=10;', file=simuf)
    print('PTC_START,x =zx, px =zpx, y = zy, py = zpy ;', file=simuf)
    print('call, file="'+ bpm_obs_file + '";', file=simuf) 
    #print >> simuf, 'PTC_TRACK,icase=4, closed_orbit, dump,'
    print('PTC_TRACK,icase=4,  CLOSED_ORBIT=false, dump,', file=simuf) 
    print('element_by_element,', file=simuf)
    print('turns=1, file="'+ track_out + '", ffile=1, norm_no=1,', file=simuf)
    #print >> simuf, 'turns=100, file='+ track_out + ', ffile=1, norm_no=1,'
    print("onetable, extension=V;", file=simuf)
    print('PTC_TRACK_END;', file=simuf)
    print('PTC_END;', file=simuf)
    if (not args.twiss): print('stop;', file=simuf)
    

    
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

current_dir=os.getcwd()
os.chdir(out_dir)
with open(out_dir + "madx.out", "wb") as madxout:  call([madx_exec, simu_file], stdout=madxout)
os.chdir(current_dir)

if (args.twiss):
    twissfilename= out_dir + "my_model_err"
    latticefilename=out_dir + "lattice_err.asc"
    twissfile = open(twissfilename,'r')
    latticefile = open(latticefilename,'w')
    for line in twissfile:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):#header
            #print sline
        #else:
            if ('*' in sline):#column names
                colname = sline.index('NAME')-1
                cols = sline.index('S')-1
                colbetx = sline.index('BETX')-1
                colbety = sline.index('BETY')-1
                colmux = sline.index('MUX')-1
                colmuy = sline.index('MUY')-1
                colalfx = sline.index('ALFX')-1
                colalfy = sline.index('ALFY')-1             
            else:#print output
                print(sline[colname], sline[cols],sline[colbetx],'0',sline[colbety],sline[colalfx],sline[colalfy],'0','0','0',sline[colmux],sline[colmuy], file=latticefile)
    twissfile.close()
    latticefile.close()
    print("File",latticefilename,"created\n")
    ####creating ip file##############
    nominallattice = model_dir + 'lattice.asc'
    nameeln,seln,betzn,psizn,alfzn=leer_beta_mu3(nominallattice,'-x')
    nameel,sel,betz,psiz,alfz=leer_beta_mu3(latticefilename,'-x')
    bpmsw1l1 = '"'+'BPMSW.1L'+ ip +'.B'+beam+'"'
    bpmsw1r1 = '"'+'BPMSW.1R'+ ip +'.B'+beam+'"'
    if (args.accel == 'hl_lhc' and (args.ip == '1' or args.ip == '5')):
        bpmsw1l1 = '"'+'BPMSQ.1L'+ ip +'.B'+beam+'"'
        bpmsw1r1 = '"'+'BPMSQ.1R'+ ip +'.B'+beam+'"'        
    betllx = betz[nameel.index(bpmsw1l1)]*(1.000)
    betrrx = betz[nameel.index(bpmsw1r1)]
    phi_llx = psiz[nameel.index(bpmsw1l1)]/(2*np.pi)
    phi_rrx = psiz[nameel.index(bpmsw1r1)]/(2*np.pi)
    bw_guess = betzn[nameeln.index('"IP'+ip + '"')]
    bwx,wx = fsolve(equations, (bw_guess, 0.01), args=(betllx,betrrx))
    betaipx = bwx + (wx**2)/bwx
    sigma_w = 0
    wx = wx + gauss(0,sigma_w)
    sigma_bw = 0
    bwx = bwx + gauss(0,sigma_bw)
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
    
    outp = open(out_dir + 'ip.results','w')
    print('* LABEL		  BETASTAR   	  BETASTAR_ERR      WAIST          WAIST_ERR      BETAWAIST   	BETAWAIST_ERR', file=outp) 
    print('$ %s                 %le               %le            %le              %le           %le              %le     ', file=outp)
    QLyKf_file=model_dir  + "Quad_KyL.txt"
    QLyKf=open(QLyKf_file,'r')

    nameq=[]
    length=[]
    strength=[]
    for i in QLyKf:
        il = i.split(None)
        if (len(il) > 0):
            if ('MQ' in il[0]):
                nameq.append(il[0])
                length.append(float(il[3].strip(',')))
                strength.append(float(il[4]))
    QLyKf.close()
    if (args.accel == 'lhc'): strengthq1l = float(strength[nameq.index('MQXA.1L' + ip)])
    if (args.accel == 'hl_lhc' and (args.ip == '5' or args.ip == '1')): strengthq1l = float(strength[nameq.index('MQXFA.A1L' + ip)])
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
    call(['rm', '-f', out_dir + 'lattice_err.asc'])
    call(['rm', '-f', out_dir + 'ip.results'])
    call(['rm', '-f', out_dir + 'twiss_err.optics'])
    call(['rm', '-f', out_dir + 'twiss_err.optics'])
    call(['rm', '-f', out_dir + 'my_model_err'])
    

    
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
    track_f= open(sim_tbt_dir + 'avermax.sdds.new','w')
    for i in range(len(trackx)):
        if i > 0:
            print(trackx[i][0], trackx[i][1], trackx[i][2],trackx[i][3], tracky[i][3], file=track_f)
    track_f.close()

    #call(['rm', '-f', sim_tbt_dir + 'trackone'])
    #call(['rm', '-f', sim_tbt_dir + 'sim_tbt.sdds.new'])
    print("File", sim_tbt_dir + 'avermax.sdds.new', "created")
else:#To generate TBT data
     ###############Creating sdds file##################################################
    in_file =  out_dir + 'trackone'
    out_file = sim_tbt_dir + 'sim_tbt.sdds.new'
    multiturn2sddsnew(in_file,out_file)
    print("\n")


templ.close()    


version_file = open(out_dir + 'version_tbt_simulator.dat','a')
print("Version", version, "timestamp", datetime.datetime.now(), file=version_file)
version_file.close()


