
import os
import sys
from math import sqrt,cos,sin,atan2,fabs, tan
from math import pi
import math
from numpy import *
from scipy.optimize import fsolve
from scipy import optimize
from numpy import polyfit
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from utils_ActPhase10 import *
import random
from random import gauss
#from Utilities import outliers
import outliers
from subprocess import call
from subprocess import Popen
import argparse
import datetime
#-----------------------------------------------------------------------

version=7
#Version 7 to choose in the arcs only BPMs with even name numbers and to print the value of betas at BPMSWs estimated from w and bw. March 9 2024.
#Version 6 to be able to read the  omc3 getllm output files. March 8 2024
#Version 5 to solve problems that appear when directories have dots and name files that sometimes were lowercase when they should be uppercase. March 4 2024.
#Version 4 to  simplify the parser. March 1 2024.
#Version 3 to rename some variables and flags, to define arc regions with default values, to avoid MT2avermax using lattice_err.asc, to modified the variables
#saved in input_line.dat, to define the ip_phase_bpm like the only one that needs measured phase. February 29 2024.
#Version 2 to include the fact that the sign of the waist depends on the polarity of triplet quadrupoles rather than the IP itself. February 22 2024
#Version 1 to force reading of average trajectories using -fa flag regardless of bpm data type input 
#Version 0 was copied  from /Users/jfcp/bin/ActionPhaseJump23 on February 13/2024.
print("ActionPhaseJump Version", version)
####################################################################
####################Function definitions ##########################
def traj_data(orbitfile,nameel,eje):
    allinesx =[]
    if (eje == '-x'):
        plane = '0'
    if (eje == '-y'):
        plane = '1'

        
    ####Read the orbit###########################################
    for line in orbitfile:
        cutline =line.split(None)
        if("#" in line):
            #print line,
            lll=0
        else:
            name = '"'+cutline[1]+'"'
            if name in nameel:
                if (cutline[0] == plane):
                    locationx = float(cutline.pop(2)) #change axial position to float
                    cutline.insert(2,locationx)
                    allinesx.append(cutline)

    ####Sort the orbit data using axial position as index
    sort_linesx = sorted(allinesx, key=lambda pos: pos[2])
    #if (eje == '-x'):
    #    print  "number of BPMs read in X", len(allinesx)
    #if (eje == '-y'):
    #    print  "number of BPMs read in Y", len(allinesx)
    ####Do  average orbit and save in averx#########################
    posx1v=[]
    posxall=[]
    averx=[]
    for j in range(len(sort_linesx)):
        for i in range(len(sort_linesx[j])-3):
            posx1v.append(float(sort_linesx[j][i+3]))
        posxall.append(posx1v)
        averx.append(average(posxall[j]))
        posx1v=[]
    if (eje == '-x'): print("average values of",j, "horizontal bpms were read from avermax.sdds.new") 
    if (eje == '-y'): print("average values of", j,"vertical bpms were read from avermax.sdds.new") 


    ###Do Horizontal diff orbit and save in diforbx############################
    diforbx =[]
    bpmdiforbx =[]


    for j in range(len(sort_linesx)):
        for i in range(len(sort_linesx[j])-3):
            #print len(sort_linesx[j]), i, float(sort_linesx[j][i+3]), averx[j]
            #bpmdiforbx.append(float(sort_linesx[j][i+3])-averx[j])
            bpmdiforbx.append(float(sort_linesx[j][i+3]))
            #print len(bpmdiforbx)
        diforbx.append(bpmdiforbx)
        bpmdiforbx =[]

    orbitfile.close()
    return diforbx,  sort_linesx







def wbw(ipfile,eje, beam, ip, QLyKf_file, accel):
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
    if (accel == 'lhc'): strengthq1l = float(strength[nameq.index('MQXA.1L' + ip)])
    if (accel == 'hl_lhc' and (ip == '5' or ip == '1')): strengthq1l = float(strength[nameq.index('MQXFA.A1L' + ip)])

        
    bybwfile= open (ipfile,'r')
    bybwfile.readline()
    bybwfile.readline()
    ab=bybwfile.readline()
    if(strengthq1l < 0 ):
        if(eje == '-x'):
            sab = ab.split(None)
            if (beam == '1'): w = float(sab[3])
            if (beam == '2'): w = -float(sab[3])
            bw = float(sab[5])
        else:
            ac = bybwfile.readline()
            sac = ac.split(None)
            if (beam == '1'): w = -float(sac[3])
            if (beam == '2'): w = float(sac[3])        
            bw = float(sac[5])
            bybwfile.close()
            
    if(strengthq1l > 0 ):
        if(eje == '-x'):
            sab = ab.split(None)
            if (beam == '1'): w = -float(sab[3])
            if (beam == '2'): w = float(sab[3])
            bw = float(sab[5])
        else:
            ac = bybwfile.readline()
            sac = ac.split(None)
            if (beam == '1'): w = float(sac[3])
            if (beam == '2'): w = -float(sac[3])        
            bw = float(sac[5])
            bybwfile.close()    

        
    return w , bw




def equations(p,betl,betr):
    bwv, wv = p
    return (bwv-betl+((lipl+wv)**2)/bwv,bwv-betr+((lipr-wv)**2)/bwv)

def write_file(namef,data, parameter):
    fdata=open(namef,parameter)
    for line in data:
        print(line[0], end=" ", file=fdata)
        for i in range(1,len(line)):
            print(line[i], end=" ", file=fdata)
        print(file=fdata)
    fdata.close()


def test_func(x, a, b):
    return a * np.sin(x) + b*np.cos(x)

def get_JPaver(zred,psiz):
    x_data = np.array(psiz)
    y_data = np.array(zred)
    params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[1e-9, 1e-9])
    return params



def filter2(s_list0, values_list0,bpm_list0,plane):
    values_list1 =[]
    bpm_list1 =[]
    s_list1 =[]
    for i in bpm_list0:
        if ('BPM.' in i):
            bpm_list1.append(i)
            values_list1.append(values_list0[bpm_list0.index(i)])
            s_list1.append(s_list0[bpm_list0.index(i)])
    continuar=1
    while (continuar==1):
        l=len(values_list1)
        if l>1:
            #Promedio
            aver=doaverage(values_list1)
            ##Desviacion standard de la lista
            sd=dodesvstd(values_list1)
            #eliminando datos mayores a N*sigma
            ind=0
            nparasacar=0
            while (ind < l and nparasacar==0):
                if(fabs(values_list1[ind]-aver) <= fabs(2*sd)):
                    #print values_list1[ind]-aver, N*sd
                    ind=ind+1
                else:
                    s_list1.pop(ind)
                    bpm_list1.pop(ind)
                    values_list1.pop(ind)
                    nparasacar=1
            if (nparasacar==0):
                continuar=0
        else:
            continuar=0

    data_outfile =[]
    if (plane == '-x'): pl = 0
    if (plane == '-y'): pl = 1    
    #print "end", avervalues,len(s_list2), len(bpm_list2)
    for i in  bpm_list1:
        data_outfile.append([pl,i, s_list1[bpm_list1.index(i)], values_list1[bpm_list1.index(i)],doaverage(values_list1)])
    return data_outfile   

def filter1a(s_list0, values_list0,bpm_list0,plane):
    values_list1 =[]
    bpm_list1 =[]
    s_list1 =[]

    for i in bpm_list0:
        if ('BPM.' in i):
            bpm_list1.append(i)
            values_list1.append(values_list0[bpm_list0.index(i)])
            s_list1.append(s_list0[bpm_list0.index(i)])
    values_list1_np = np.array(values_list1)
    s_list1_np = np.array(s_list1)
    bpm_list1_np =np.array(bpm_list1)
    mask = outliers.get_filter_mask(values_list1_np,s_list1_np)
    avervalues = np.mean(values_list1_np[mask])
    Dvalues = values_list1_np[mask].std()
    values_list2 = values_list1_np[mask].tolist()
    s_list2 = s_list1_np[mask].tolist()
    bpm_list2 = bpm_list1_np[mask].tolist()
    
    #avervalues = average(values_list1)
    #valuesarray = array(values_list1)
    #Dvalues =   valuesarray.std()
    s_list3 =[]
    bpm_list3 =[]
    values_list3 =[]
    for i in values_list2:
        if ( i < (avervalues +2*Dvalues) and i > (avervalues -2*Dvalues)):
            bpm_list3.append(bpm_list2[values_list2.index(i)])
            values_list3.append(i)
            s_list3.append(s_list2[values_list2.index(i)])
    avervalues3 =  average(values_list3)       
    data_outfile =[]
    if (plane == '-x'): pl = 0
    if (plane == '-y'): pl = 1    
    for i in  bpm_list3:

        data_outfile.append([pl,i, s_list3[bpm_list3.index(i)], values_list3[bpm_list3.index(i)], avervalues3])
    return data_outfile   
            

def filter1(s_list0, values_list0,bpm_list0,plane):
    values_list1 =[]
    bpm_list1 =[]
    s_list1 =[]
    for i in bpm_list0:
        if ('BPM.' in i):
            bpm_list1.append(i)
            values_list1.append(values_list0[bpm_list0.index(i)])
            s_list1.append(s_list0[bpm_list0.index(i)])
    avervalues = average(values_list1)
    valuesarray = array(values_list1)
    Dvalues =   valuesarray.std()
    values_list2 =[]
    bpm_list2 =[]
    s_list2 =[]
    #print "start", avervalues, values_list0[0],  values_list0[1]
    for i in values_list1:
        if ( i < (avervalues +2*Dvalues) and i > (avervalues -2*Dvalues)):
            bpm_list2.append(bpm_list1[values_list1.index(i)])
            values_list2.append(i)
            s_list2.append(s_list1[values_list1.index(i)])
    avervalues2 =  average(values_list2)       
    data_outfile =[]
    if (plane == '-x'): pl = 0
    if (plane == '-y'): pl = 1    
    #print "end", avervalues,len(s_list2), len(bpm_list2)
    for i in  bpm_list2:
        #data_outfile.append([str(s_list2[bpm_list2.index(i)]), str(values_list2[bpm_list2.index(i)]), str(i), str(avervalues2)])
        data_outfile.append([pl,i, s_list2[bpm_list2.index(i)], values_list2[bpm_list2.index(i)], avervalues2])
        #data_outfile.append([ str(i), str(s_list2[bpm_list2.index(i)]), str(values_list2[bpm_list2.index(i)]), str(avervalues2)])
    #escribir_archivo(filt_file,data_outfile,'w')
    return data_outfile   





def JyP_trip(averact, averphase, nameel,  psiz,    nm,   betzn,psizn,   w, bw, beam, bpm_phase, ip, accel):
    name_BPMSW_l = '"BPMSW.1L' + str(ip) +'.B'+ str(beam)+'"'
    name_BPMSW_r = '"BPMSW.1R' + str(ip) +'.B'+ str(beam)+'"'
    name_2LBPM = '"BPMS.2L' + str(ip) +'.B'+ str(beam)+'"'
    name_2RBPM = '"BPMS.2R' + str(ip) +'.B'+ str(beam)+'"'
    if (accel == 'hl_lhc' and (ip == '1' or ip == '5')):
        name_BPMSW_l = '"BPMSQ.1L' + str(ip) +'.B'+ str(beam)+'"'
        name_BPMSW_r = '"BPMSQ.1R' + str(ip) +'.B'+ str(beam)+'"'
        name_2LBPM = '"BPMSQ.A2L' + str(ip) +'.B'+ str(beam)+'"'
        name_2RBPM = '"BPMSQ.A2R' + str(ip) +'.B'+ str(beam)+'"'        
    betll = betzn[nm.index(name_BPMSW_l)]*(1.000)
    betrr = betzn[nm.index(name_BPMSW_r)]
    bw_guess = betzn[nm.index('"IP'+ip + '"')]
    bwn,wn = fsolve(equations, (bw_guess, 0.01), args=(betll,betrr))
    #print "name_bpm", bpm_phase, name_BPMSW_l,name_BPMSW_r
    if (bpm_phase == name_BPMSW_l or bpm_phase == name_2LBPM):
        gammap = psiz[nameel.index(bpm_phase)]+ atan2(lipl+w,bw) - averphase
    if (bpm_phase == name_BPMSW_r or bpm_phase == name_2RBPM ):
        gammap = psiz[nameel.index(bpm_phase)] - atan2(lipl-w,bw) - averphase


    tangamma = (wn-w+tan(gammap)*bw)/bwn
    jota = averact*bwn*cos(gammap)**2*(1+tangamma**2)/bw
    gamma =atan2((wn-w+tan(gammap)*bw), bwn)
    if cos(gammap) < 0:
        gamma = gamma +np.pi

    if (bpm_phase == name_BPMSW_l or bpm_phase == name_2LBPM):
        delta_p = psizn[nm.index(bpm_phase)] + atan2(lipl,bwn) -gamma
    if (bpm_phase == name_BPMSW_r or bpm_phase == name_2RBPM):
        delta_p = psizn[nm.index(bpm_phase)] - atan2(lipl,bwn) -gamma 


    delta2 = modf(delta_p/(2*np.pi))[0]*2*np.pi
    if delta2 > np.pi:
        delta2 = delta2 - 2*np.pi
    #print "testjyp", jota, delta2, datosdeaccion[2][1], datosdeaccion[3][1]
    return jota, delta2




def Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpml,bpmr,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe,nameelxe,psixe,betxe,wx,bwx,s_acdipole,plane,beam,bpm_phase,ip, method, accel):
    s_c=[]
    bpm_c=[]
    xred_c=[]
    psix_c=[]

    s_nf=[]
    bpm_nf=[]
    xred_nf=[]
    psix_nf=[]
    
    s_nfapj=[]
    bpm_nfapj=[]
    xred_nfapj=[]
    psix_nfapj=[]

    s_ant=[]
    bpm_ant=[]
    xred_ant=[]
    psix_ant=[]

    s_desp=[]
    bpm_desp=[]
    xred_desp=[]
    psix_desp=[]
    
    x_BPM_l = ''
    x_BPM_r = ''
    col = sdds_col - 4
    bpmswl = '"BPMSW.1L' + str(ip) +'.B'+ str(beam)+'"'
    bpmswr = '"BPMSW.1R' + str(ip) +'.B'+ str(beam)+'"'
    if (accel == 'hl_lhc' and (ip == '1' or ip == '5')):
        bpmswl = '"BPMSQ.1L' + str(ip) +'.B'+ str(beam)+'"'
        bpmswr = '"BPMSQ.1R' + str(ip) +'.B'+ str(beam)+'"'
    for line in sort_linesx:
        namebpm = '"'+line[1]+'"'
        ss=line[2]
        posx=float(diforbx[sort_linesx.index(line)][col])

        if (namebpm == bpml):
            x_BPM_l=posx/1000 # converting mm to meters
        if (namebpm == bpmr):
            x_BPM_r=posx/1000 # converting mm to meters
        #print namebpm, bpmswl
        if (namebpm == bpmswl):
            x_bpm_swl=posx/1000 # converting mm to meters
        if (namebpm == bpmswr):
            x_bpm_swr=posx/1000 #


        if (namebpm in  nameelxe):
            xred_nf.append(posx*0.001/sqrt(betxe[nameelxe.index(namebpm)]))
            psix_nf.append(psixe[nameelxe.index(namebpm)])
            s_nf.append(ss)
            bpm_nf.append(namebpm)
        
        xred_nfapj.append(posx*0.001/sqrt(betx[nameelx.index(namebpm)]))
        psix_nfapj.append(psix[nameelx.index(namebpm)])
        s_nfapj.append(ss)
        bpm_nfapj.append(namebpm)
        def even_bpm(bpm):
            digit1 = int(bpm[5])
            try:
                    digit0 = int(bpm[6])
                    number = digit1*10 + digit0
            except ValueError:
                    number = digit1
            if (number % 2 == 0): out = True
            else: out = False
            return out
            
        
        if ((ss > sL_begin) and (ss < sL_end) and ('BPM.' in line[1]) and even_bpm(namebpm)):
            xred_ant.append(posx*0.001/sqrt(betx[nameelx.index(namebpm)]))
            psix_ant.append(psix[nameelx.index(namebpm)])
            s_ant.append(ss)
            bpm_ant.append(namebpm)
        if ((ss > sR_begin) and (ss < sR_end) and  ('BPM.' in line[1]) and even_bpm(namebpm)):
            xred_desp.append(posx*0.001/sqrt(betx[nameelx.index(namebpm)]))
            psix_desp.append(psix[nameelx.index(namebpm)])
            s_desp.append(ss)
            bpm_desp.append(namebpm)

        if ((ss > s_acdipole) and (namebpm in  nameelxe) and ('BPM.' in line[1]) and even_bpm(namebpm) ):
            xred_c.append(posx*0.001/sqrt(betxe[nameelxe.index(namebpm)]))
            psix_c.append(psixe[nameelxe.index(namebpm)])
            s_c.append(ss)
            bpm_c.append(namebpm)
       
    #if (x_BPM_l == ''):
    #    print bpml, " is no present in the horizontal plane"
    #    #print "In comandos, change elEqError. It can be BPMSY.4L*, BPMS.2L*, or BPMSW.1L*"
    #    exit()
    #if (x_BPM_r == ''):
    #    print bpmr, " is no present in the horizontal plane"
    #    #print "In comandos, change erEqError. It can be BPMSY.4R*, BPMS.2R*, or BPMSW.1R*"
    #    exit()

    act_nf, phase_nf =  doaccionyfase(xred_nf,psix_nf)
    if (plane == '-x'):
        lpad = [0]*len(bpm_nf)
    if (plane == '-y'):
        lpad = [1]*len(bpm_nf)    
    act_nfm =   list(zip(lpad,bpm_nf, s_nf, act_nf))
    phase_nfm = list(zip(lpad,bpm_nf, s_nf,  phase_nf))
    APJact_nf, APJphase_nf = doaccionyfase(xred_nfapj,psix_nfapj)
    APJact_nfm = list(zip(lpad,bpm_nfapj, s_nfapj, APJact_nf))
    APJphase_nfm = list(zip(lpad,bpm_nfapj, s_nfapj, APJphase_nf))
    action_c, phase_c = doaccionyfase(xred_c,psix_c)
    actm = filter1a(s_c, action_c,bpm_c,plane)
    averactc =float(actm[0][4])
    #averactc = (get_JPaver(xred_c,psix_c)[0]**2 + get_JPaver(xred_c,psix_c)[1]**2)/2    
    phasem = filter1a(s_c, phase_c,bpm_c,plane) 
    averphasec = float(phasem[0][4])
    #averphasec = atan2(-get_JPaver(xred_c,psix_c)[1],get_JPaver(xred_c,psix_c)[0])
    action_ant, phase_ant = doaccionyfase(xred_ant,psix_ant)


    actionm_ant = filter1a(s_ant, action_ant,bpm_ant,plane)
    averactant = float(actionm_ant[0][4])  
    phasem_ant = filter1a(s_ant, phase_ant,bpm_ant,plane)
    actionm_ant.append('&')
    phasem_ant.append('&')
    averphaseant = float(phasem_ant[0][4])
    action_desp, phase_desp = doaccionyfase(xred_desp,psix_desp)
    #ffasecont_v01(phase_desp)


    #action_ant, phase_ant = doaccionyfase(xred_ant,psix_ant)
    
    actionm_desp =  filter1a(s_desp, action_desp,bpm_desp,plane)
    averactdesp =  float(actionm_desp[0][4])
    phasem_desp =  filter1a(s_desp, phase_desp,bpm_desp,plane)
    actionm_desp.append('&')
    phasem_desp.append('&')
    averphasedesp = float(phasem_desp[0][4])

    #print "valores promedios", (get_JPaver(xred_c,psix_c)[0]**2 + get_JPaver(xred_c,psix_c)[1]**2)/2,atan2(-get_JPaver(xred_c,psix_c)[1],get_JPaver(xred_c,psix_c)[0]) ,averactc, averphasec

    xswl_red = x_bpm_swl/sqrt(betx[nameelx.index(bpmswl)])
    xswr_red = x_bpm_swr/sqrt(betx[nameelx.index(bpmswr)])
    psix_swl = psix[nameelx.index(bpmswl)]
    psix_swr = psix[nameelx.index(bpmswr)]
    if (method == 'bpmsw'): act_trip, phase_trip = inversion(xswl_red, xswr_red, psix_swl, psix_swr)
    if (method == 'kmod'): act_trip, phase_trip = JyP_trip(averactc,averphasec, nameelxe,   psixe,  nameelx,  betx,  psix,  wx, bwx, beam, bpm_phase, ip, accel)
    #print "action_trip2, action_trip", act_trip2, act_trip, bpmswl, bpmswr
    #print "phase_trip2, phase_trip", phase_trip2, phase_trip
    APJactm =actionm_ant + actionm_desp
    APJphasem =phasem_ant + phasem_desp
    #print phasem_ant
    #print "passsed"
   
    #APJactm =actionm_ant + actionm_desp
    #APJphasem =phasem_ant + phasem_desp
    #print len(actionm_ant), len(actionm_desp), len(APJactm)
    #write_file('test.dat',actionm_desp,'w')

    #action_ant=eliminar_data_nsigma(action_ant,nsigma)
    #averactant=doaverage(action_ant)
    #phase_ant=eliminar_data_nsigma(phase_ant,nsigma)
    #averphaseant=doaverage(phase_ant)
    #averphaseantx = averphaseant
    #action_desp=eliminar_data_nsigma(action_desp,nsigma)
    #averactdesp=doaverage(action_desp)
    #phase_desp=eliminar_data_nsigma(phase_desp,nsigma)
    #averphasedesp=doaverage(phase_desp)
    #averphasedespx = averphasedesp

    
    #averactant2 = averactant
    #averphaseant2 = averphaseant
    #averactdesp2 =  averactdesp
    #averphasedesp2 = averphasedesp

 
    #for i in actionm_ant:
    #    i.append(averactant2)

    #for i in actionm_desp:
    #    i.append(averactdesp2)

    #for i in phasem_ant:
    #    i.append(averphaseant2)

    #for i in phasem_desp:
    #    i.append(averphasedesp2)


    return APJact_nfm,APJphase_nfm,act_nfm,phase_nfm,APJactm,APJphasem,actm,phasem, averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, x_BPM_l, x_BPM_r


def diff_TBT(orbitfile, nameel,eje):
############   OUTPUT FILES   #######################
    if (eje == '-x'):
        plane = '0'
 
    if (eje == '-y'):
        plane = '1'
 
        
    ####Read the orbit###########################################
    allinesx=[]
    for line in orbitfile:
        cutline =line.split(None)
        if("#" in line):
            #print line,
            lll=0
        else:
            name = '"'+cutline[1]+'"'
            if name in nameel:
                if (cutline[0] == plane):
                    locationx = float(cutline.pop(2)) #change axial position to float
                    cutline.insert(2,locationx)
                    allinesx.append(cutline)

    ####Sort the orbit data using axial position as index
    sort_linesx = sorted(allinesx, key=lambda pos: pos[2])
    #if (eje == '-x'):
    #    print  "number of BPMs read in X", len(allinesx)
    #if (eje == '-y'):
    #    print  "number of BPMs read in Y", len(allinesx)
    ####Do  average orbit and save in averx#########################
    posx1v=[]
    posxall=[]
    averx=[]
    for j in range(len(sort_linesx)):
        for i in range(len(sort_linesx[j])-3):
            posx1v.append(float(sort_linesx[j][i+3]))
        posxall.append(posx1v)
        averx.append(average(posxall[j]))
        posx1v=[]



    ###Do Horizontal diff orbit and save in diforbx############################
    diforbx =[]
    bpmdiforbx =[]


    for j in range(len(sort_linesx)):
        for i in range(len(sort_linesx[j])-3):
            bpmdiforbx.append(float(sort_linesx[j][i+3])-averx[j])
            #bpmdiforbx.append(float(sort_linesx[j][i+3]))
        diforbx.append(bpmdiforbx)
        bpmdiforbx =[]



    
    #if (eje == '-x'):
    #    print "number of turns  read in X", turn
    #if (eje == '-y'):
    #    print "number of turns  read in Y", turn
    return diforbx, sort_linesx
    

def MT2avermax(tbt,bpm_max,betas_file,orb_apj_dir):
    theorbit = tbt     ##Input file trackone.sdds.new   with path
    element_seq = '\"'+ bpm_max + '\"'
    archivobetas = betas_file ## Input file lattice.asc or lattice_err.asc with path
    
    ############## OUTPUT FILES ################################
    max_file = orb_apj_dir + 'avermax.sdds.new'
    maxmax_file = orb_apj_dir + 'avermaxmax.sdds.new'
    angx_file = orb_apj_dir + 'angxVSturn.dat'
    angy_file = orb_apj_dir + 'angyVSturn.dat'
    avermax= open(max_file,'w')
    avermaxmax= open(maxmax_file,'w')
    angfilex= open(angx_file,'w')
    angfiley= open(angy_file,'w')
    ##See also output files of  TBT_phases function

    ############Read lattice file########################################
    #archivobetas="lattice.asc"
    nameel=[] # Element names
    sel=[]   # element s coordinate
    betx=[]   # beta(s) functions in z
    bety=[]   # beta(s) functions in z
    psix=[]   # mu(s)*2*PI, mu's are read from lattice.asc file or equivalent
    psiy=[]   # mu(s)*2*PI, mu's are read from lattice.asc file or equivalent
    latticef = open(archivobetas,'r')
    for line in latticef:
        sline=line.split(None)
        nameel.append(sline[0])
        sel.append(round(float(sline[1]),2))
        betx.append(float(sline[2]))
        psix.append(2*pi*float(sline[10]))
        bety.append(float(sline[4]))
        psiy.append(2*pi*float(sline[11]))
    latticef.close()

    phas_elementx = psix[nameel.index(element_seq)]
    phas_elementy = psiy[nameel.index(element_seq)]


    orbitfile = open(theorbit,'r')
    diforbx, sort_linesx = diff_TBT(orbitfile, nameel,'-x')
    orbitfile.close()

    orbitfile = open(theorbit,'r')
    diforby, sort_linesy = diff_TBT(orbitfile, nameel,'-y')
    orbitfile.close()
    



    ########Classify turns that have maximus and minimums around phas_element  within ddelta range#################################################################
    #turnx_max, turnx_min = turns_max2(angxtraj, phas_elementx,ddelta,'-x')
    #turny_max, turny_min = turns_max2(angytraj, phas_elementy,ddelta,'-y')
    turnx_max=[]
    turny_max=[]
    turnx_min=[]
    turny_min=[]
    for j in range(len(diforbx)):
        for turn in range(1,len(sort_linesx[j])-3):
             if (sort_linesx[j][1] == bpm_max  and  diforbx[j][turn-1] > 0):
                 turnx_max.append(turn)
             if (sort_linesx[j][1] == bpm_max  and  diforbx[j][turn-1] < 0):
                 turnx_min.append(turn)
    print(turn, "turns of data were read  in the horizontal plane from",j ,"bpms of the shifted TBT file")
    
    for j in range(len(diforby)):
        for turn in range(1,len(sort_linesy[j])-3):
             if (sort_linesy[j][1] == bpm_max  and  diforby[j][turn-1] > 0):
                 turny_max.append(turn)
             if (sort_linesy[j][1] == bpm_max  and  diforby[j][turn-1] < 0):
                 turny_min.append(turn)
    print(turn, "turns of data were read  in the vertical plane from",j,"bpms of the shifted TBT file")


    ########################Start building  trajectories with maximum in one plane ################################################################################
    bpm_readx=[]
    bpm_ready=[]
    for j in range(len(diforbx)):
        bpm_readx.append([0,sort_linesx[j][1],sort_linesx[j][2]])

    for j in range(len(diforby)):
        bpm_ready.append([1,sort_linesy[j][1],sort_linesy[j][2]])

    ####Trajectories  with Maximum in the horizontal  plane#########################################################
    turnx = turnx_max + turnx_min
 
    #print "phasex, turns used:", 360*math.modf(phas_elementx/(2*pi))[0],len(turnx_max),len(turnx_min),len(turnx)
    ######Horizontal Trajectories######################
    for j in range(len(diforbx)):
        maxline_hor =[]
        for turn in turnx_max:
            #print turn, j, diforbx[j][turn-1]
            maxline_hor.append(diforbx[j][turn-1])
        for turn in turnx_min:
            maxline_hor.append(-diforbx[j][turn-1])
        bpm_readx[j].append(average(maxline_hor))
    #######Vertical Trajectories#########################
    for j in range(len(diforby)):
        maxline_ver =[]
        for turn in turnx_max:
            maxline_ver.append(diforby[j][turn-1])
        for turn in turnx_min:
            maxline_ver.append(-diforby[j][turn-1])
        bpm_ready[j].append(average(maxline_ver))




    ####Trajectories with Maximum in the vertical plane############################################################
    turny = turny_max + turny_min
    #print "phasey, turns used:", 360*math.modf(phas_elementy/(2*pi))[0],len(turny_max),len(turny_min),len(turny)
    ######Horizontal Trajectories######################
    for j in range(len(diforbx)):
        minline_hor =[]
        for turn in turny_max:
            minline_hor.append(diforbx[j][turn-1])
        for turn in turny_min:
            minline_hor.append(-diforbx[j][turn-1])
        bpm_readx[j].append(average(minline_hor))
    #######Vertical Trajectories#########################
    for j in range(len(diforby)):
        minline_ver =[]
        for turn in turny_max:
            minline_ver.append(diforby[j][turn-1])
        for turn in turny_min:
            minline_ver.append(-diforby[j][turn-1])   
        bpm_ready[j].append(average(minline_ver))
    ##################################################################################################################




    ######## Save trajectory with maximum in horizontal plane in fourth column and trajectory with maximum in vertical plane in fifth column of avermax.sdds.new###
    for j in bpm_readx:
        for i in j:
            print(i, end=' ', file=avermax)
        print(file=avermax)

    for j in bpm_ready:
        for i in j:
            print(i, end=' ', file=avermax)
        print(file=avermax)
    avermax.close()
    avermaxmax.close()
    angfilex.close()
    angfiley.close()


        
    return
    ###########################################################################
def shift_exp_TBT(orbit, lattice, out_file, beam):
    orbitf = open(orbit, 'r')
    shifted_f = open(out_file,'w')
    latticef = open(lattice,'r')
    latt_list = []
    for line in latticef:
        lines = line.split(None)
        latt_list.append([lines[0], lines[1], lines[2]])



    #inj_element ='MKI.A5R8.B2'
    hbpm_n = 0
    vbpm_n = 0
    for line in orbitf:
        lines = line.split(None)
        if(len(lines) < 30):
            print(line.strip(), file=shifted_f)
        else:
            #hbpm_n = 0
            for i in range(0,len(lines)):
                if(i == 0):
                    if (lines[0] == '0'): hbpm_n = hbpm_n + 1
                    if (lines[0] == '0'): vbpm_n = vbpm_n + 1 
                    print(lines[0], end=' ', file=shifted_f)
                if (i == 1):
                    print(lines[1], end=' ', file=shifted_f)
                    bpmname = '"'+lines[1]+'"'
                    new_s =  latt_list[list(zip(*latt_list))[0].index(bpmname)][1]
                if (i == 2):
                    print("    ",new_s, end=' ', file=shifted_f)
                    if (beam == '1'):
                        if (float(new_s) > 20109): #New s for injection beam1
                            flag = 1
                        else:
                            flag = 0
                    if (beam == '2'):
                        if (float(new_s) > 13906): #New s for injection beam2
                            flag = 0
                        else:
                            flag = 1


                if ((i >2) and (i <(len(lines)-2))):
                    if (float(lines[i+flag]) < 0):
                           print(lines[i+flag], end=' ', file=shifted_f)
                    else:
                        print("", lines[i+flag], end=' ', file=shifted_f) 
                if (i == (len(lines)-2)):
                    if (float(lines[i+flag]) < 0):
                           print(lines[i+flag], file=shifted_f)
                    else:
                        print("", lines[i+flag], file=shifted_f)
        #print "Is this bpm number" , hbpm_n
    print(i, "turns of data were read from",hbpm_n,"horizontal bpms and",vbpm_n,"vertical bpms of the clean TBT file")
    shifted_f.close()
    latticef.close()
    orbitf.close()
    return

def get_meas_lattice(getllm_dir,beam,lattice, out_file):

    
    betaxname=getllm_dir+'/'+'getbetax_free.out'
    betayname=getllm_dir+'/'+'getbetay_free.out'
    phasexname=getllm_dir+'/'+'getphasetotx_free.out'
    phaseyname=getllm_dir+'/'+'getphasetoty_free.out'
    modelname=lattice


    betax = open(betaxname,'r')
    betay = open(betayname,'r')
    phasex = open(phasexname,'r')
    phasey = open(phaseyname,'r')
    model=open(modelname,'r')
    #lattmdl = open(lattmdlname,'r')
    latticefile = open(out_file,'w')




    betaxdata =[]
    for line in betax:
        sline = line.split(None)
        if ('Q1' in sline):qx=float(sline[3])
        if ('Q2' in sline):qy=float(sline[3])
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            if('*' in sline):
                colname = sline.index('NAME')-1
                cols = sline.index('S')-1
                colbetx =sline.index('BETX')-1
                errbetx = sline.index('ERRBETX')-1
                stdbetx = sline.index('STATBETX')-1
                betxmdl = sline.index('BETXMDL')-1
            else:
                if(float(sline[colbetx]) > 0):
                    betaxdata.append([sline[colname],sline[cols],sline[colbetx],sline[errbetx],sline[stdbetx],sline[betxmdl]])
    #print betaxdata
    betax.close()

    betaydata =[]
    for line in betay:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            if('*' in sline):
                colname = sline.index('NAME')-1
                cols = sline.index('S')-1
                colbety =sline.index('BETY')-1
                errbety = sline.index('ERRBETY')-1
                stdbety = sline.index('STATBETY')-1
                betymdl = sline.index('BETYMDL')-1
            else:
                if(float(sline[colbety]) > 0):
                    betaydata.append([sline[colname],sline[cols],sline[colbety],sline[errbety],sline[stdbety],sline[betymdl]])
    #print betaydata
    betay.close()



    modeldata =[]
    for line in model:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            modeldata.append([sline[0],sline[1],sline[10],sline[11]])
    model.close()

    if (beam == '1'):
        bpm_ini = '"BPM.8L4.B1"' #first BPM after ac dipole
        bpm_ini_orig='"BPMYB.5L2.B1"' # BPM were TbT data starts
        s_inj =  20109.4
    if (beam == '2'):
        bpm_ini = '"BPM.11L4.B2"' #first BPM after ac dipole
        bpm_ini_orig='"BPMYB.5R8.B2"' # BPM were TbT data starts
        s_inj = 13906.0


    modelline= [x for x in modeldata if x[0]==bpm_ini]
    phasex_ini = float(modelline[0][2])
    modelline= [x for x in modeldata if x[0]==bpm_ini_orig]
    phasex_ini_orig = float(modelline[0][2])
    phasexdata =[]
    for line in phasex:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline) or ('*' in sline)):
            search = sline[0]
            modelline= [x for x in modeldata if x[0]==search]
            s_new = float(modelline[0][1])
            phxmdl =  math.modf(float(modelline[0][2])-  phasex_ini )[0]
            muxmdl = float(modelline[0][2])
            if (s_new > s_inj):
                phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini)[0]
            else:
                phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini-qx)[0] 
               #phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini-0.31043057065384655)[0] #beam1 specific for 31-03-16 data
               #phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini-0.3099859352998008)[0] beam2
            muxmeas =  muxmdl+ phasex_exp -phxmdl 
            phasexdata.append([sline[0],sline[2],sline[5],sline[6],phxmdl,muxmdl,muxmeas,phasex_exp])
    phasex.close()


    modelline= [x for x in modeldata if x[0]==bpm_ini]
    phasey_ini = float(modelline[0][3])
    modelline= [x for x in modeldata if x[0]==bpm_ini_orig]
    phasey_ini_orig = float(modelline[0][3])

    phaseydata =[]
    for line in phasey:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline) or ('*' in sline)):
            search = sline[0]
            modelline= [x for x in modeldata if x[0]==search]
            s_new = float(modelline[0][1])
            phymdl =  math.modf(float(modelline[0][3])-  phasey_ini )[0]
            muymdl = float(modelline[0][3])
            if (s_new > s_inj):
                phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini)[0] 
            else:
                phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini-qy)[0] 
               #phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini-0.3296230537164725)[0]  #beam1 specific for 31-03-16 data
               #phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini-0.3198006119379839)[0] beam2
            muymeas =  muymdl+ phasey_exp -phymdl 
            phaseydata.append([sline[0],sline[2],sline[5],sline[6],phymdl,muymdl,muymeas,phasey_exp])
    phasey.close()



    lattice=[]
    for i in betaxdata:
        search = i[0]
        modelline= [x for x in modeldata if x[0]==search]
        s_new = float(modelline[0][1])
        betax_out = i[2]
        betayline = [x for x in betaydata if x[0]==search]
        phasexline = [x for x in phasexdata if x[0]==search]
        phaseyline = [x for x in phaseydata if x[0]==search]
        if(len(betayline) > 0 and len(phasexline) > 0 and len(phaseyline) > 0):
            betax_out = i[2]
            betay_out = betayline[0][2]
            phxmdl =  phasexline[0][4]
            phasex_out = phasexline[0][6]
            phasex_exp = phasexline[0][7]
            phymdl =  phaseyline[0][4]
            phasey_out = phaseyline[0][6]
            phasey_exp = phaseyline[0][7]
            lattice.append([i[0],s_new,betax_out,betay_out,phasex_out,phasey_out,phxmdl,phasex_exp,phymdl,phasey_exp])
            #print   i[0], s, betax_out,'0',betay_out,'0','0','0','0','0',phasex_out,phasey_out
            #print i[0], i[1],  betax_out,betay_out,phasex_out,phasey_out
    sorted_lattice = sorted(lattice, key=lambda x:x[1])

    for k in sorted_lattice:
        print(k[0], k[1], k[2], '0', k[3], '0','0','0','0','0',k[4], k[5], file=latticefile)
    latticefile.close()
    return


def get_meas_lattice_omc3(getllm_dir,beam,lattice, out_file):

    
    betaxname=getllm_dir+'/'+'beta_phase_x.tfs'
    betayname=getllm_dir+'/'+'beta_phase_y.tfs'
    phasexname=getllm_dir+'/'+'total_phase_x.tfs'
    phaseyname=getllm_dir+'/'+'total_phase_y.tfs'
    modelname=lattice


    betax = open(betaxname,'r')
    betay = open(betayname,'r')
    phasex = open(phasexname,'r')
    phasey = open(phaseyname,'r')
    model=open(modelname,'r')
    #lattmdl = open(lattmdlname,'r')
    latticefile = open(out_file,'w')




    betaxdata =[]
    for line in betax:
        sline = line.split(None)
        if ('Q1' in sline):qx=float(sline[3])
        if ('Q2' in sline):qy=float(sline[3])
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            if('*' in sline):
                colname = sline.index('NAME')-1
                cols = sline.index('S')-1
                colbetx =sline.index('BETX')-1
                errbetx = sline.index('ERRBETX')-1
                stdbetx = sline.index('ERRDELTABETX')-1
                betxmdl = sline.index('BETXMDL')-1
            else:
                if(float(sline[colbetx]) > 0):
                    betaxdata.append([sline[colname],sline[cols],sline[colbetx],sline[errbetx],sline[stdbetx],sline[betxmdl]])
    #print betaxdata
    betax.close()

    betaydata =[]
    for line in betay:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            if('*' in sline):
                colname = sline.index('NAME')-1
                cols = sline.index('S')-1
                colbety =sline.index('BETY')-1
                errbety = sline.index('ERRBETY')-1
                stdbety = sline.index('ERRDELTABETY')-1
                betymdl = sline.index('BETYMDL')-1
            else:
                if(float(sline[colbety]) > 0):
                    betaydata.append([sline[colname],sline[cols],sline[colbety],sline[errbety],sline[stdbety],sline[betymdl]])
    #print betaydata
    betay.close()



    modeldata =[]
    for line in model:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline)):
            modeldata.append([sline[0],sline[1],sline[10],sline[11]])
    model.close()

    if (beam == '1'):
        bpm_ini = '"BPM.8L4.B1"' #first BPM after ac dipole
        bpm_ini_orig='"BPMYB.5L2.B1"' # BPM were TbT data starts
        s_inj =  20109.4
    if (beam == '2'):
        bpm_ini = '"BPM.11L4.B2"' #first BPM after ac dipole
        bpm_ini_orig='"BPMYB.5R8.B2"' # BPM were TbT data starts
        s_inj = 13906.0


    modelline= [x for x in modeldata if x[0]==bpm_ini]
    phasex_ini = float(modelline[0][2])
    modelline= [x for x in modeldata if x[0]==bpm_ini_orig]
    phasex_ini_orig = float(modelline[0][2])
    phasexdata =[]
    for line in phasex:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline) or ('*' in sline)):
            search = sline[2]
            modelline= [x for x in modeldata if x[0]==search]
            s_new = float(modelline[0][1])
            phxmdl =  math.modf(float(modelline[0][2])-  phasex_ini )[0]
            muxmdl = float(modelline[0][2])
            if (s_new > s_inj):
                phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini)[0]
            else:
                phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini-qx)[0] 
               #phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini-0.31043057065384655)[0] #beam1 specific for 31-03-16 data
               #phasex_exp = math.modf(float(sline[5])+phasex_ini_orig-phasex_ini-0.3099859352998008)[0] beam2
            muxmeas =  muxmdl+ phasex_exp -phxmdl 
            phasexdata.append([sline[2],sline[0],sline[5],sline[6],phxmdl,muxmdl,muxmeas,phasex_exp])
    phasex.close()


    modelline= [x for x in modeldata if x[0]==bpm_ini]
    phasey_ini = float(modelline[0][3])
    modelline= [x for x in modeldata if x[0]==bpm_ini_orig]
    phasey_ini_orig = float(modelline[0][3])

    phaseydata =[]
    for line in phasey:
        sline = line.split(None)
        if not(('@' in sline) or ('$' in sline) or ('#' in sline) or ('*' in sline)):
            search = sline[2]
            modelline= [x for x in modeldata if x[0]==search]
            s_new = float(modelline[0][1])
            phymdl =  math.modf(float(modelline[0][3])-  phasey_ini )[0]
            muymdl = float(modelline[0][3])
            if (s_new > s_inj):
                phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini)[0] 
            else:
                phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini-qy)[0] 
               #phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini-0.3296230537164725)[0]  #beam1 specific for 31-03-16 data
               #phasey_exp = math.modf(float(sline[5])+phasey_ini_orig-phasey_ini-0.3198006119379839)[0] beam2
            muymeas =  muymdl+ phasey_exp -phymdl 
            phaseydata.append([sline[2],sline[0],sline[5],sline[6],phymdl,muymdl,muymeas,phasey_exp])
    phasey.close()



    lattice=[]
    for i in betaxdata:
        search = i[0]
        modelline= [x for x in modeldata if x[0]==search]
        s_new = float(modelline[0][1])
        betax_out = i[2]
        betayline = [x for x in betaydata if x[0]==search]
        phasexline = [x for x in phasexdata if x[0]==search]
        phaseyline = [x for x in phaseydata if x[0]==search]
        if(len(betayline) > 0 and len(phasexline) > 0 and len(phaseyline) > 0):
            betax_out = i[2]
            betay_out = betayline[0][2]
            phxmdl =  phasexline[0][4]
            phasex_out = phasexline[0][6]
            phasex_exp = phasexline[0][7]
            phymdl =  phaseyline[0][4]
            phasey_out = phaseyline[0][6]
            phasey_exp = phaseyline[0][7]
            lattice.append([i[0],s_new,betax_out,betay_out,phasex_out,phasey_out,phxmdl,phasex_exp,phymdl,phasey_exp])
            #print   i[0], s, betax_out,'0',betay_out,'0','0','0','0','0',phasex_out,phasey_out
            #print i[0], i[1],  betax_out,betay_out,phasex_out,phasey_out
    sorted_lattice = sorted(lattice, key=lambda x:x[1])

    for k in sorted_lattice:
        print(k[0], k[1], k[2], '0', k[3], '0','0','0','0','0',k[4], k[5], file=latticefile)
    latticefile.close()
    return




def tfs2outb1_v3(in_tfs, apj_dir,kk):
    inf = open(in_tfs,'r')
    kk = kk + 1
    out_file = apj_dir + 'ip.results' + str(kk)
    outf = open(out_file,'w')

    line = inf.readline()
    cutline = line.split(None)

    labeli=cutline.index('LABEL') -1 
    betsxi=cutline.index('BETSTARX') -1
    betsxi_err=cutline.index('ERRBETSTARX') -1
    betsyi=cutline.index('BETSTARY') -1
    betsyi_err=cutline.index('ERRBETSTARY') -1
    betwxi=cutline.index('BETWAISTX') -1
    betwxi_err=cutline.index('ERRBETWAISTX') -1
    betwyi=cutline.index('BETWAISTY') -1
    betwyi_err=cutline.index('ERRBETWAISTY') -1
    wxi=cutline.index('WAISTX') -1
    wxi_err=cutline.index('ERRWAISTX') -1
    wyi=cutline.index('WAISTY') -1
    wyi_err=cutline.index('ERRWAISTY') -1

    line = inf.readline()
    cutline = line.split(None)

    line = inf.readline()
    cutline = line.split(None)




    print('*   LABEL                        BETASTAR         BETASTAR_ERR                WAIST            WAIST_ERR            BETAWAIST        BETAWAIST_ERR', file=outf)
    print('$     %s                            %le                  %le                  %le                  %le                  %le                  %le', file=outf)
    print("   ", cutline[labeli], '{: 17.5f}'.format(float(cutline[betsxi])),  '{: 17.5f}'.format(float(cutline[betsxi_err])) , '{: 26.5f}'.format(float(cutline[wxi])),'{: 17.5f}'.format(float(cutline[wxi_err])) , '{: 19.5f}'.format(float(cutline[betwxi])), '{: 17.5f}'.format(float(cutline[betwxi_err])), file=outf)
    print("   ", cutline[labeli], '{: 17.5f}'.format(float(cutline[betsyi])),  '{: 17.5f}'.format(float(cutline[betsyi_err])) , '{: 26.5f}'.format(float(cutline[wyi])),'{: 17.5f}'.format(float(cutline[wyi_err])) , '{: 19.5f}'.format(float(cutline[betwyi])), '{: 17.5f}'.format(float(cutline[betwyi_err])), file=outf)
    outf.close()
    inf.close()    
    return out_file

def aver_kmod2(apj_dir, kmod_outf_l):
    
    
    #n_args=len(sys.argv)
    kmod_dir=apj_dir
    kmodout=apj_dir + 'ip_aver.results'

    lbetsx=[]
    lbetsx_err=[]
    lwx=[]
    lwx_err=[]
    lbetwx=[]
    lbetwx_err=[]

    lbetsy=[]
    lbetsy_err=[]
    lwy=[]
    lwy_err=[]
    lbetwy=[]
    lbetwy_err=[]

    for i in kmod_outf_l:
        kmod_name = i
        #print 'file_name', kmod_name
        #kmod_name=kmod_dir + "/" + 'ip.results' + str(ii)
        kmod_file=[line.split(None) for line in open(kmod_name)]
        lbetsx.append(float(kmod_file[2][1]))
        lbetsx_err.append(float(kmod_file[2][2]))
        lwx.append(float(kmod_file[2][3])) 
        lwx_err.append(float(kmod_file[2][4])) 
        lbetwx.append(float(kmod_file[2][5]))
        lbetwx_err.append(float(kmod_file[2][6])) 

        lbetsy.append(float(kmod_file[3][1]))
        lbetsy_err.append(float(kmod_file[3][2]))
        lwy.append(float(kmod_file[3][3])) 
        lwy_err.append(float(kmod_file[3][4])) 
        lbetwy.append(float(kmod_file[3][5]))
        lbetwy_err.append(float(kmod_file[3][6]))

    betsx=sum(lbetsx)/len(lbetsx)
    betsx_err=sum(lbetsx_err)/len(lbetsx_err)
    wx=sum(lwx)/len(lwx)
    wx_err=sum(lwx_err)/len(lwx_err)
    betwx=sum(lbetwx)/len(lbetwx)
    betwx_err=sum(lbetwx_err)/len(lbetwx_err)


    betsy=sum(lbetsy)/len(lbetsy)
    betsy_err=sum(lbetsy_err)/len(lbetsy_err)
    wy=sum(lwy)/len(lwy)
    wy_err=sum(lwy_err)/len(lwy_err)
    betwy=sum(lbetwy)/len(lbetwy)
    betwy_err=sum(lbetwy_err)/len(lbetwy_err)

    fout=open(kmodout,'w')




    for i in kmod_file[0]:
        if kmod_file[0].index(i) == 0:
            print('{:^3}'.format(i), end=' ', file=fout)
        else:
            print('{:^12}'.format(i), end=' ', file=fout)

    print(file=fout)

    for i in kmod_file[1]:
        if kmod_file[1].index(i) == 0:
            print('{:^3}'.format(i), end=' ', file=fout)
        else:
            print('{:^12}'.format(i), end=' ', file=fout)

    print(file=fout)


    print('{:^9}'.format(kmod_file[2][0]), '{:^6.4}'.format(betsx), '{:^17.4}'.format(betsx_err),  '{:^10.4}'.format(wx), '{:^12.4}'.format(wx_err), '{:^12.4}'.format(betwx), '{:^12.4}'.format(betwx_err), file=fout)

    print('{:^9}'.format(kmod_file[3][0]), '{:^6.4}'.format(betsy), '{:^17.4}'.format(betsy_err),  '{:^10.4}'.format(wy), '{:^12.4}'.format(wy_err), '{:^12.4}'.format(betwy), '{:^12.4}'.format(betwy_err), file=fout)
    fout.close()
    return


    ################################

def possibleBPMs(bpm,option,ip,beam, accel):
    if (accel == 'hl_lhc'):
        bl_ipB1IP1 = ['BPMSQ.1L1.B1', 'BPMSQ.1R1.B1', 'BPMSQ.A2L1.B1',  'BPMSQ.A2R1.B1']
        bl_lkB1IP1 = ['BPMSQ.4L1.B1','BPMSQ.B3L1.B1','BPMSQ.A3L1.B1','BPMSQT.B2L1.B1', 'BPMSQ.A2L1.B1', 'BPMSQ.1L1.B1']
        bl_rkB1IP1 =['BPMSQ.4R1.B1','BPMSQ.B3R1.B1','BPMSQ.A3R1.B1','BPMSQT.B2R1.B1', 'BPMSQ.A2R1.B1', 'BPMSQ.1R1.B1']
        bl_avB1IP1 =['BPMSQ.4L1.B1','BPMSQ.B3L1.B1','BPMSQ.A3L1.B1','BPMSQT.B2L1.B1', 'BPMSQ.A2L1.B1', 'BPMSQ.1L1.B1','BPMSQ.4R1.B1','BPMSQ.B3R1.B1','BPMSQ.A3R1.B1','BPMSQT.B2R1.B1', 'BPMSQ.A2R1.B1', 'BPMSQ.1R1.B1']

        bl_ipB2IP1 = ['BPMSQ.1L1.B2', 'BPMSQ.1R1.B2', 'BPMSQ.A2L1.B2',  'BPMSQ.A2R1.B2']
        bl_lkB2IP1 = ['BPMSQ.4L1.B2','BPMSQ.B3L1.B2','BPMSQ.A3L1.B2','BPMSQT.B2L1.B2', 'BPMSQ.A2L1.B2', 'BPMSQ.1L1.B2']
        bl_rkB2IP1 =['BPMSQ.4R1.B2','BPMSQ.B3R1.B2','BPMSQ.A3R1.B2','BPMSQT.B2R1.B2', 'BPMSQ.A2R1.B2', 'BPMSQ.1R1.B2']
        bl_avB2IP1 =['BPMSQ.4L1.B2','BPMSQ.B3L1.B2','BPMSQ.A3L1.B2','BPMSQT.B2L1.B2', 'BPMSQ.A2L1.B2', 'BPMSQ.1L1.B2','BPMSQ.4R1.B2','BPMSQ.B3R1.B2','BPMSQ.A3R1.B2','BPMSQT.B2R1.B2', 'BPMSQ.A2R1.B2', 'BPMSQ.1R1.B2']

        bl_ipB1IP5 = ['BPMSQ.1L5.B1', 'BPMSQ.1R5.B1', 'BPMSQ.A2L5.B1',  'BPMSQ.A2R5.B1']
        bl_lkB1IP5 = ['BPMSQ.4L5.B1','BPMSQ.B3L5.B1','BPMSQ.A3L5.B1','BPMSQT.B2L5.B1', 'BPMSQ.A2L5.B1', 'BPMSQ.1L5.B1']
        bl_rkB1IP5 =['BPMSQ.4R5.B1','BPMSQ.B3R5.B1','BPMSQ.A3R5.B1','BPMSQT.B2R5.B1', 'BPMSQ.A2R5.B1', 'BPMSQ.1R5.B1']
        bl_avB1IP5 =['BPMSQ.4L5.B1','BPMSQ.B3L5.B1','BPMSQ.A3L5.B1','BPMSQT.B2L5.B1', 'BPMSQ.A2L5.B1', 'BPMSQ.1L5.B1','BPMSQ.4R5.B1','BPMSQ.B3R5.B1','BPMSQ.A3R5.B1','BPMSQT.B2R5.B1', 'BPMSQ.A2R5.B1', 'BPMSQ.1R5.B1']

        bl_ipB2IP5 = ['BPMSQ.1L5.B2', 'BPMSQ.1R5.B2', 'BPMSQ.A2L5.B2',  'BPMSQ.A2R5.B2']
        bl_lkB2IP5 = ['BPMSQ.4L5.B2','BPMSQ.B3L5.B2','BPMSQ.A3L5.B2','BPMSQT.B2L5.B2', 'BPMSQ.A2L5.B2', 'BPMSQ.1L5.B2']
        bl_rkB2IP5 =['BPMSQ.4R5.B2','BPMSQ.B3R5.B2','BPMSQ.A3R5.B2','BPMSQT.B2R5.B2', 'BPMSQ.A2R5.B2', 'BPMSQ.1R5.B2']
        bl_avB2IP5 =['BPMSQ.4L5.B2','BPMSQ.B3L5.B2','BPMSQ.A3L5.B2','BPMSQT.B2L5.B2', 'BPMSQ.A2L5.B2', 'BPMSQ.1L5.B2','BPMSQ.4R5.B2','BPMSQ.B3R5.B2','BPMSQ.A3R5.B2','BPMSQT.B2R5.B2', 'BPMSQ.A2R5.B2', 'BPMSQ.1R5.B2']




    if (accel == 'lhc'):
        bl_ipB1IP1 = ['BPMSW.1L1.B1', 'BPMSW.1R1.B1', 'BPMS.2L1.B1',  'BPMS.2R1.B1']
        bl_lkB1IP1 = ['BPMSY.4L1.B1', 'BPMS.2L1.B1', 'BPMSW.1L1.B1']
        bl_rkB1IP1 =['BPMSY.4R1.B1', 'BPMS.2R1.B1', 'BPMSW.1R1.B1']
        bl_avB1IP1 =['BPMSY.4L1.B1', 'BPMS.2L1.B1', 'BPMSW.1L1.B1', 'BPMSY.4R1.B1', 'BPMS.2R1.B1', 'BPMSW.1R1.B1']

        bl_ipB2IP1 = ['BPMSW.1L1.B2', 'BPMSW.1R1.B2', 'BPMS.2L1.B2',  'BPMS.2R1.B2']
        bl_lkB2IP1 = ['BPMSY.4L1.B2', 'BPMS.2L1.B2', 'BPMSW.1L1.B2']
        bl_rkB2IP1 =['BPMSY.4R1.B2', 'BPMS.2R1.B2', 'BPMSW.1R1.B2']
        bl_avB2IP1 =['BPMSY.4L1.B2', 'BPMS.2L1.B2', 'BPMSW.1L1.B2', 'BPMSY.4R1.B2', 'BPMS.2R1.B2', 'BPMSW.1R1.B2']

        bl_ipB1IP5 = ['BPMSW.1L5.B1', 'BPMSW.1R5.B1', 'BPMS.2L5.B1',  'BPMS.2R5.B1']
        bl_lkB1IP5 = ['BPMSY.4L5.B1', 'BPMS.2L5.B1', 'BPMSW.1L5.B1']
        bl_rkB1IP5 =['BPMSY.4R5.B1', 'BPMS.2R5.B1', 'BPMSW.1R5.B1']
        bl_avB1IP5 =['BPMSY.4L5.B1', 'BPMS.2L5.B1', 'BPMSW.1L5.B1', 'BPMSY.4R5.B1', 'BPMS.2R5.B1', 'BPMSW.1R5.B1']

        bl_ipB2IP5 = ['BPMSW.1L5.B2', 'BPMSW.1R5.B2', 'BPMS.2L5.B2',  'BPMS.2R5.B2']
        bl_lkB2IP5 = ['BPMSY.4L5.B2', 'BPMS.2L5.B2', 'BPMSW.1L5.B2']
        bl_rkB2IP5 =['BPMSY.4R5.B2', 'BPMS.2R5.B2', 'BPMSW.1R5.B2']
        bl_avB2IP5 =['BPMSY.4L5.B2', 'BPMS.2L5.B2', 'BPMSW.1L5.B2', 'BPMSY.4R5.B2', 'BPMS.2R5.B2', 'BPMSW.1R5.B2']


        

    bl_ipB1IP2 = ['BPMSW.1L2.B1', 'BPMSW.1R2.B1', 'BPMS.2L2.B1',  'BPMS.2R2.B1']
    bl_lkB1IP2 = ['BPMSX.4L2.B1', 'BPMS.2L2.B1', 'BPMSW.1L2.B1']
    bl_rkB1IP2 =['BPMSX.4R2.B1', 'BPMS.2R2.B1', 'BPMSW.1R2.B1']
    bl_avB1IP2 =['BPMSX.4L2.B1', 'BPMS.2L2.B1', 'BPMSW.1L2.B1', 'BPMSX.4R2.B1', 'BPMS.2R2.B1', 'BPMSW.1R2.B1']

    bl_ipB2IP2 = ['BPMSW.1L2.B2', 'BPMSW.1R2.B2', 'BPMS.2L2.B2',  'BPMS.2R2.B2']
    bl_lkB2IP2 = ['BPMSX.4L2.B2', 'BPMS.2L2.B2', 'BPMSW.1L2.B2']
    bl_rkB2IP2 =['BPMSX.4R2.B2', 'BPMS.2R2.B2', 'BPMSW.1R2.B2']
    bl_avB2IP2 =['BPMSX.4L2.B2', 'BPMS.2L2.B2', 'BPMSW.1L2.B2', 'BPMSX.4R2.B2', 'BPMS.2R2.B2', 'BPMSW.1R2.B2']





    bl_ipB1IP8 = ['BPMSW.1L8.B1', 'BPMSW.1R8.B1', 'BPMS.2L8.B1',  'BPMS.2R8.B1']
    bl_lkB1IP8 = ['BPMSX.4L8.B1', 'BPMS.2L8.B1', 'BPMSW.1L8.B1']
    bl_rkB1IP8 =['BPMSX.4R8.B1', 'BPMS.2R8.B1', 'BPMSW.1R8.B1']
    bl_avB1IP8 =['BPMSX.4L8.B1', 'BPMS.2L8.B1', 'BPMSW.1L8.B1', 'BPMSX.4R8.B1', 'BPMS.2R8.B1', 'BPMSW.1R8.B1']

    bl_ipB2IP8 = ['BPMSW.1L8.B2', 'BPMSW.1R8.B2', 'BPMS.2L8.B2',  'BPMS.2R8.B2']
    bl_lkB2IP8 = ['BPMSX.4L8.B2', 'BPMS.2L8.B2', 'BPMSW.1L8.B2']
    bl_rkB2IP8 =['BPMSX.4R8.B2', 'BPMS.2R8.B2', 'BPMSW.1R8.B2']
    bl_avB2IP8 =['BPMSX.4L8.B2', 'BPMS.2L8.B2', 'BPMSW.1L8.B2', 'BPMSX.4R8.B2', 'BPMS.2R8.B2', 'BPMSW.1R8.B2']
    

    bl_ipB1 =[bl_ipB1IP1,bl_ipB1IP2,bl_ipB1IP5,bl_ipB1IP8]
    bl_ipB2 =[bl_ipB2IP1,bl_ipB2IP2,bl_ipB2IP5,bl_ipB2IP8]
    bl_ip = [ bl_ipB1, bl_ipB2]

    bl_lkB1 =[bl_lkB1IP1,bl_lkB1IP2,bl_lkB1IP5,bl_lkB1IP8]
    bl_lkB2 =[bl_lkB2IP1,bl_lkB2IP2,bl_lkB2IP5,bl_lkB2IP8]
    bl_lk = [ bl_lkB1, bl_lkB2]
     
    bl_rkB1 =[bl_rkB1IP1,bl_rkB1IP2,bl_rkB1IP5,bl_rkB1IP8]
    bl_rkB2 =[bl_rkB2IP1,bl_rkB2IP2,bl_rkB2IP5,bl_rkB2IP8]
    bl_rk = [ bl_rkB1, bl_rkB2]

    bl_avB1 =[bl_avB1IP1,bl_avB1IP2,bl_avB1IP5,bl_avB1IP8]
    bl_avB2 =[bl_avB2IP1,bl_avB2IP2,bl_avB2IP5,bl_avB2IP8]
    bl_av = [ bl_avB1, bl_avB2]

    bl = [bl_ip,bl_lk,bl_rk,bl_av]
    options = ['ip_phase_bpm','left_kick_bpm','right_kick_bpm','bpm_for_avermax']
    beams = ['1','2']
    ips = ['1','2','5','8']
    bpm_list=bl[options.index(option)][beams.index(beam)][ips.index(ip)]
    if bpm in bpm_list:
        bpm_list.remove(bpm)
    
    return bpm_list

def create_tbt_analysis_dir(datatype,tbt_i, apj_dir):
    if(datatype == "orig_bpmdata" or datatype == "clean_bpmdata" or datatype == "shifted_bpmdata"):
        #tbt_i_c=tbt_i.split('.')
        dirs_and_names=tbt_i.split('/')
        name_tbt=dirs_and_names[-1].split('.')[0]
        name_dir=dirs_and_names[-1].split('.')[0]
        orb_apj_dir = apj_dir + "/" + name_dir + "/"
        
    if(datatype == "aver_bpmdata"):
        #tbt_i_c=tbt_i.split('.')
        #dirs_and_names=tbt_i_c[0].split('/')
        dirs_and_names=tbt_i.split('/')
        #name_tbt=dirs_and_names[-1]
        name_tbt=dirs_and_names[-1].split('.')[0]
        name_dir=dirs_and_names[-2]
        orb_apj_dir = apj_dir + "/" + name_dir + "/"
        
        #tbt=tbt_i.split('/')
        #tbt.pop()
        #tbt.pop(0)
        #orb_apj_dir = '/'
        #for ii in tbt:
        #    orb_apj_dir = orb_apj_dir + '/' + ii
        #orb_apj_dir = orb_apj_dir + '/'
        #name_tbt = ''
    call(['mkdir','-p',orb_apj_dir])
    return orb_apj_dir, name_tbt

def inp_conditions(inputs,nameelx,nameely,latt_type, ip, beam,accel):
    if (inputs.left_kick_bpm != None and  inputs.right_kick_bpm != None  and inputs.ip_phase_bpm != None):
        bpml = '"'+inputs.left_kick_bpm +'"'
        bpmr ='"'+inputs.right_kick_bpm +'"'
        bpm_phase ='"'+inputs.ip_phase_bpm +'"'
        Isbpms = True
        if ((bpml not in nameelx or bpml not in nameely) and latt_type == 'nominal'):
            print('WARNING:',  inputs.left_kick_bpm, 'is not a valid BPM for "left_kick_bpm (-lb)". Instead, use', possibleBPMs(inputs.left_kick_bpm,"left_kick_bpm",ip,beam, accel), "\n")
            Isbpms = False
        #if ((bpml not in nameelx or bpml not in nameely) and latt_type == 'measured'):
        #    print  'WARNING:',inputs.left_kick_bpm , 'is not in lattice_err.asc. Measured lattice ( in either x or/and y plane) for that BPM might not be available. Instead, for "left_kick_bpm (-lb)", use one of the following BPMs:',possibleBPMs(inputs.left_kick_bpm,"left_kick_bpm",ip,beam,accel), "\n"
        #    Isbpms = False
        if ((bpmr not in nameelx or bpmr not in nameely) and latt_type == 'nominal'):
            print('WARNING:',inputs.right_kick_bpm, 'is not a valid BPM for "right_kick_bpm (-lb)". Instead use', possibleBPMs(inputs.right_kick_bpm,"right_kick_bpm",ip,beam,accel), "\n")
            Isbpms = False
        #if ((bpmr not in nameelx or bpmr not in nameely) and latt_type == 'measured'):
        #    print  'WARNING:',inputs.right_kick_bpm , 'is not in lattice_err.asc. Measured lattice ( in either x or/and y plane) for that BPM might not be available. Instead, for "right_kick_bpm (-lb)", use one of the following BPMs:',possibleBPMs(inputs.right_kick_bpm,"right_kick_bpm",ip,beam,accel), "\n"
        #    Isbpms = False            
        if ((bpm_phase not in nameelx or bpm_phase not in nameely) and latt_type == 'nominal'):
            print('WARNING:',inputs.ip_phase_bpm, 'is not a valid BPM for "ip_phase_bpm (-lb)". Instead, use', possibleBPMs(inputs.ip_phase_bpm,"ip_phase_bpm",ip,beam,accel), "\n")
            Isbpms = False
        if ((bpm_phase not in nameelx or bpm_phase not in nameely) and latt_type == 'measured'):
            print('WARNING:',inputs.ip_phase_bpm , 'is not in lattice_err.asc. Measured lattice ( in either x or/and y plane) for that BPM might not be available. Instead, for "ip_phase_bpm (-lb)", use one of the following BPMs:',possibleBPMs(inputs.ip_phase_bpm,"ip_phase_bpm",ip,beam,accel), "\n")
            Isbpms = False            
    else: Isbpms = False


        
        
    if (inputs.left_kick_bpm != None and  inputs.right_kick_bpm != None):
        bpml = '"'+inputs.left_kick_bpm +'"'
        bpmr ='"'+inputs.right_kick_bpm +'"'
        Isbpmlr = True
        if ((bpml not in nameelx or bpml not in nameely) and latt_type == 'nominal'):
            Isbpmlr = False
        #if ((bpml not in nameelx or bpml not in nameely) and latt_type == 'measured'):
        #    Isbpmlr = False
        if ((bpmr not in nameelx or bpmr not in nameely) and latt_type == 'nominal'):
            Isbpmlr = False
        #if ((bpmr not in nameelx or bpmr not in nameely) and latt_type == 'measured'):
        #    Isbpmlr = False         
    else: Isbpmlr = False

    #betaxname=getllm_dir+'/'+'getbetax_free.out'
    #betayname=getllm_dir+'/'+'getbetay_free.out'
    #phasexname=getllm_dir+'/'+'getphasetotx_free.out'
    #phaseyname=getllm_dir+'/'+'getphasetoty_free.out'
        
        
    if (inputs.getllm_dir != None or inputs.getllm_dir_omc3 != None or inputs.err_lattice_file != None):
        if (inputs.err_lattice_file != None):
            if (os.path.exists(inputs.err_lattice_file)): Iserrlattice= True
            else:
                Iserrlattice = False
                print("WARNING:", inputs.err_lattice_file, "was not found")
        if (inputs.getllm_dir != None):
                Iserrlattice = True
                if ( not os.path.exists(inputs.getllm_dir + '/'+'getbetax_free.out')):
                        print("WARNING:", inputs.getllm_dir + '/'+'getbetax_free.out', "was not found")
                        Iserrlattice = False
                if ( not os.path.exists(inputs.getllm_dir + '/'+'getbetay_free.out')):
                        print("WARNING:", inputs.getllm_dir + '/'+'getbetay_free.out', "was not found")
                        Iserrlattice = False
                if ( not os.path.exists(inputs.getllm_dir + '/'+'getphasetotx_free.out')):
                        print("WARNING:", inputs.getllm_dir + '/'+'getphasetotx_free.out', "was not found")
                        Iserrlattice = False
                if ( not os.path.exists(inputs.getllm_dir + '/'+'getphasetoty_free.out')):
                        print("WARNING:", inputs.getllm_dir + '/'+'getphasetoty_free.out', "was not found")
                        Iserrlattice = False

        if (inputs.getllm_dir_omc3 != None):
                Iserrlattice = True
                if ( not os.path.exists(inputs.getllm_dir_omc3 + '/'+'beta_phase_x.tfs')):
                        print("WARNING:", inputs.getllm_dir_omc3 + '/'+'beta_phase_x.tfs', "was not found")
                        Iserrlattice = False
                if ( not os.path.exists(inputs.getllm_dir_omc3 + '/'+'beta_phase_y.tfs')):
                        print("WARNING:", inputs.getllm_dir_omc3 + '/'+'beta_phase_y.tfs', "was not found")
                        Iserrlattice = False
                if ( not os.path.exists(inputs.getllm_dir_omc3 + '/'+'total_phase_x.tfs')):
                        print("WARNING:", inputs.getllm_dir_omc3 + '/'+'total_phase_x.tfs', "was not found")
                        Iserrlattice = False
                if ( not os.path.exists(inputs.getllm_dir_omc3 + '/'+'total_phase_y.tfs')):
                        print("WARNING:", inputs.getllm_dir_omc3 + '/'+'total_phase_y.tfs', "was not found")
                        Iserrlattice = False

     
    else: Iserrlattice = False
    if (inputs.long_kmod_file != None or inputs.short_kmod_file != None):
        Iskmod = True
        if (inputs.long_kmod_file != None):
            long_kmod_file_s = inputs.long_kmod_file.split(',')
            for ff in long_kmod_file_s:
                if (not os.path.exists(ff)):
                    print("WARNING:", ff, "was not found")
                    Iskmod = False
        if (inputs.short_kmod_file != None):
            short_kmod_file_s = inputs.short_kmod_file.split(',')
            for ff in short_kmod_file_s:
                if (not os.path.exists(ff)):
                    print("WARNING:", ff, "was not found")
                    Iskmod = False        
    else : Iskmod = False
    if (Isbpms == True and  Iserrlattice == True and Iskmod == True): Is_ip_ap_kmod = True
    else: Is_ip_ap_kmod = False
    if (not ((Iskmod and Iserrlattice and inputs.left_kick_bpm != None and inputs.right_kick_bpm != None  and inputs.ip_phase_bpm != None) or (Iskmod == False and  Iserrlattice == False and inputs.left_kick_bpm == None and inputs.right_kick_bpm == None  and inputs.ip_phase_bpm == None))):
        trying_2run4quads = True
    else: trying_2run4quads =False
    return Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms, Isbpmlr ,trying_2run4quads 

def dphi_rad(phim,phit):
    phim_dec =  float(str(phim-int(phim))[1:])*2*pi
    phit_dec =  float(str(phit-int(phit))[1:])*2*pi
    dphi = phim_dec - phit_dec
    return dphi


def get_arc_values(ip,parameter):
    ip1 = ['14200', '16800', '17500', '20000']
    ip2 = ['17500', '19700', '21000', '23000']
    ip5 = ['1000', '3200', '4200', '6500']
    ip8 = ['11000', '13400', '14100', '16600']
    par_l=['left_arc_start','left_arc_end','right_arc_start','right_arc_end']
    if (ip == '1'): value = ip1[par_l.index(parameter)]
    if (ip == '2'): value = ip2[par_l.index(parameter)]
    if (ip == '5'): value = ip5[par_l.index(parameter)]
    if (ip == '8'): value = ip8[par_l.index(parameter)]
    return value
###########################################################################################################################################################################









##########################MAIN PROGRAM##############################################################################################################################################
##########################PROGRAM INPUTS#######################

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog='''\
    Recommend ranges for arcs  left and right to the IRs

    IP1
       left arc range: from 14200 meters to 16800 meters 
       right arc range: from 17500 meters to 20000 meters

    IP2
       left arc range: from  17500 meters to 19700 meters
       right arc range: from 21000 meters to 23000 meters

    IP5
       left arc range: from 1000 meters to 3200 meters
       right arc range: from 4200 meters to 6500 meters

    IP8
       left arc range: from  11000 meters to 13400 meters
       right arc range: from 14100 meters to 16600 meters
    ''')
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group3 = parser.add_mutually_exclusive_group()
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
    help="Directory where lattice.asc , triplet.dat and twiss.dat are",
    required=True,
    dest="model_dir"
)
parser.add_argument(
    "-ls", "--left_arc_start",
    help="longitudinal position in meters. For recommend values, see help at the end",
    #required=True,
    dest="left_arc_start"
)
parser.add_argument(
    "-le", "--left_arc_end",
    help="longitudinal position in meters. For recommend values, see help at the end",
    #required=True,
    dest="left_arc_end"
)
parser.add_argument(
    "-rs", "--right_arc_start",
    help="longitudinal position in meters. For recommend values, see help at the end",
    #required=True,
    dest="right_arc_start"
)
parser.add_argument(
    "-re", "--right_arc_end",
    help="longitudinal position in meters. For recommend values, see help at the end",
    #required=True,
    dest="right_arc_end"
)
parser.add_argument(
    "-ab", "--bpm_for_avermax",
    help="reference bpm to build the avermax trajectory. It is also used as the place where the equivalent kick for the whole IR  is estimated",
    required=True,
    dest="bpm_for_avermax"
)

parser.add_argument(
    "-od", "--out_dir",
    help="Directory where APJ analysis is done",
    required=True,
    dest="out_dir"
)
group1.add_argument(
    "-of", "--orig_bpmdata",
    help="tbt file in sdds format",
    dest="orig_bpmdata"
)
group1.add_argument(
    "-cf", "--clean_bpmdata",
    help="clean  and synchronized tbt file in ascii format",
    dest="clean_bpmdata"
)
group1.add_argument(
    "-af", "--aver_bpmdata",
    help="average trajectory (avermax.sdds.new)",
    dest="aver_bpmdata"
)
group1.add_argument(
    "-sf", "--shifted_bpmdata",
    help="tbt file with turns starting at AC dipole location ",
    dest="shifted_bpmdata"
)
group2.add_argument(
    "-gd", "--getllm_dir",
    help="gettllm output directory",
    dest="getllm_dir"
)
group2.add_argument(
    "-g3", "--getllm_dir_omc3",
    help="omc3 gettllm output directory",
    dest="getllm_dir_omc3"
)
group2.add_argument(
    "-ml", "--err_lattice_file",
    help="measured lattice file in asc format and it should be given with the full pathname. Ussually, this file has name lattice_err.asc",
    dest="err_lattice_file"
)        
group3.add_argument(
    "-lk", "--long_kmod_file",
    help="kmod file in tfs format with full pathname. Ussually, this file has name results.tfs",
    dest="long_kmod_file"
)        
group3.add_argument(
    "-sk", "--short_kmod_file",
    help="kmod file in old format with full pathname. Ussually, this file has name ip_aver.results.tfs",
    dest="short_kmod_file"
)

parser.add_argument(
    "-pb", "--ip_phase_bpm",
    help="bpm used to extract lattice phase at the begining of the inter-triplet space",
    dest="ip_phase_bpm"
)
parser.add_argument(
    "-lb", "--left_kick_bpm",
    help="bpm where equivalent  kick in the left triplet is estimated",
    dest="left_kick_bpm"
)
parser.add_argument(
    "-rb", "--right_kick_bpm",
    help="bpm where equivalent  kick in the right triplet is estimated",
    dest="right_kick_bpm"
)
parser.add_argument(
   "-p", "--showplots",
    help="to show APJplots",
    action="store_true",
    dest="showplots"
)
parser.add_argument(
   "-a", "--accel",
    help="accelerator name",
    choices=['lhc','hl_lhc'],
    dest="accel",
    default='lhc'
)
parser.add_argument(
   "-fa", "--force_averages",
    help="It will force reading of average trajectories if they exist regardless of bpm data type input",
    action="store_true",
    dest="force_averages"
)


args = parser.parse_args()






print('\n')
if((args.orig_bpmdata == None) and (args.clean_bpmdata == None) and (args.aver_bpmdata == None) and (args.shifted_bpmdata == None)):
    print("bpm position data must be provided either as tbt files in sdds or ascii format, or as an average trayectory")
    exit()

if (args.orig_bpmdata != None):
    orig_bpmdata_s = args.orig_bpmdata.split(',')
    for ff in orig_bpmdata_s:
        if (not os.path.exists(ff)):
            print(ff, 'was not found')
            exit()

if (args.clean_bpmdata != None):
    clean_bpmdata_s = args.clean_bpmdata.split(',')
    for ff in clean_bpmdata_s:
        if (not os.path.exists(ff)):
            print(ff, 'was not found')
            exit()

if (args.shifted_bpmdata != None):
    shifted_bpmdata_s = args.shifted_bpmdata.split(',')
    for ff in shifted_bpmdata_s:
        if (not os.path.exists(ff)):
            print(ff, 'was not found')
            exit()

if (args.aver_bpmdata != None):
    aver_bpmdata_s = args.aver_bpmdata.split(',')
    for ff in aver_bpmdata_s:
        if (not os.path.exists(ff)):
            print(ff, 'was not found')
            exit()

accel=args.accel    
beam=args.beam
ip= args.ip

if (args.left_arc_start == None):left_arc_start = get_arc_values(ip,'left_arc_start')
else:left_arc_start = args.left_arc_start
if (args.left_arc_end == None): left_arc_end = get_arc_values(ip,'left_arc_end')
else:left_arc_end = args.left_arc_end     
if (args.right_arc_start == None): right_arc_start = get_arc_values(ip,'right_arc_start')
else: right_arc_start = args.right_arc_start
if (args.right_arc_end == None): right_arc_end = get_arc_values(ip,'right_arc_end')
else: right_arc_end = args.right_arc_end


#name_orbitmax = sys.argv[3]     #input file avermax.sdds.new, which contains two average trajectories: one with a maximum in the horizontal plane and the other in vertical plane
#name_orbitmaxmax = sys.argv[4] #input file with the four kind of trajectories avermaxmax, avermaxmin, averminmax and averminmin
latticef=  args.model_dir + "/" + "lattice.asc"
#########defining  region of analysis######
sL_begin = float(left_arc_start)
sL_end = float(left_arc_end)
sR_begin = float(right_arc_start)
sR_end = float(right_arc_end)
###############################################
getllm_dir= args.getllm_dir #input file (measured lattice)
getllm_dir_omc3= args.getllm_dir_omc3 #input file (measured lattice)
lattice_errf = args.err_lattice_file
in_tfs =  args.long_kmod_file     #input ip files in tfs format and separated by commas (no spaces)
ipff = args.short_kmod_file
#print "bpm_phase", bpm_phase
bpml_nf= args.left_kick_bpm
bpmr_nf=  args.right_kick_bpm
intf=  args.model_dir + "/" + "integrals.dat" #input file with triplet quad integrals
apj_dir = args.out_dir
#print "OUTPUT DIR", apj_dir
bpm_max=args.bpm_for_avermax
orig_model= args.model_dir + "/" + 'twiss.dat'

if (not os.path.exists(latticef)) :
    print(latticef, "was not found")
    exit()
if (not os.path.exists(intf)) :
    print(intf, "was not found")
    exit()
#print accel
if (accel == 'lhc'):
    if (not os.path.exists(orig_model)) :
        print(orig_model, "was not found")
        exit()


####################################################################################################################
call(['mkdir','-p',apj_dir])




    
if (beam == '1'): s_acdipole = 158
if (beam == '2'): s_acdipole = 287
    

########################## OUTPUT FILES  #########################################

b1x_file = apj_dir + 'b1x.dat'
resxf=open(b1x_file,'w')
b1y_file = apj_dir + 'b1y.dat'
resyf=open(b1y_file,'w')

b1xl_file = apj_dir + 'b1xl.dat'
resxlf=open(b1xl_file,'w')
b1yl_file = apj_dir + 'b1yl.dat'
resylf=open(b1yl_file,'w')

b1xr_file = apj_dir + 'b1xr.dat'
resxrf=open(b1xr_file,'w')
b1yr_file = apj_dir + 'b1yr.dat'
resyrf=open(b1yr_file,'w')


b1xl_file_bpmsw = apj_dir + 'b1xl_bpmsw.dat'
resxlf_bpmsw=open(b1xl_file_bpmsw,'w')
b1yl_file_bpmsw = apj_dir + 'b1yl_bpmsw.dat'
resylf_bpmsw=open(b1yl_file_bpmsw,'w')

b1xr_file_bpmsw = apj_dir + 'b1xr_bpmsw.dat'
resxrf_bpmsw=open(b1xr_file_bpmsw,'w')
b1yr_file_bpmsw = apj_dir + 'b1yr_bpmsw.dat'
resyrf_bpmsw=open(b1yr_file_bpmsw,'w')




#b1xl4_file = apj_dir + 'b1xl4.dat'
##call(['rm','-f',b1xl4_file])
#resxlf4=open(b1xl4_file,'w')
#b1yl4_file = apj_dir + 'b1yl4.dat'
##call(['rm','-f',b1yl4_file])
#resylf4=open(b1yl4_file,'w')

#b1xr4_file = apj_dir + 'b1xr4.dat'
##call(['rm','-f',b1xr4_file])
#resxrf4=open(b1xr4_file,'w')
#b1yr4_file = apj_dir + 'b1yr4.dat'
##call(['rm','-f',b1yr4_file])
#resyrf4=open(b1yr4_file,'w')


corrs_file = apj_dir + 'IR' + str(ip) + '_corrs_tbtbytbt.dat'
corrq2l_name = 'corrq2l' + str(ip)
corrq3l_name = 'corrq3l' + str(ip)
corrq2r_name = 'corrq2r' + str(ip)
corrq3r_name = 'corrq3r' + str(ip)
fcorrs_kmod=open(corrs_file,'w')

ir_err_2q_file= apj_dir + 'IR'+str(ip)+'_errors_2quads.madx'
ir_err_4q_file= apj_dir + 'IR'+str(ip)+'_errors_4quads.madx'
ir_2err = open(ir_err_2q_file,'w')
ir_4err= open(ir_err_4q_file,'w')

apj_gpl_file = open(apj_dir + 'gAPJ.gpl','w')
conventional_gpl_file = open (apj_dir + 'gConventional.gpl' ,'w')



#corrs_file_bpmsw = apj_dir + 'IR' + str(ip) + '_corrs_tbtbytbt_bpmsw.dat' ##Highly sensitive to BPM calibrations. Use for simulations only
#call(['rm','-f',corrs_file])

#fcorrs_bpmsw=open(corrs_file_bpmsw,'w')
#print >> fcorrs,'{:^11}'.format(corrq2l_name),'{:^11}'.format(corrq3l_name), '{:^11}'.format(corrq2r_name), '{:^11}'.format(corrq3r_name) 


####also see output files of JyP_trip function
##################################################################################
qslint='MQSX.3L'+str(ip)
qsrint='MQSX.3R'+str(ip)

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



if (ip == '1'):
    lipl = lipl1
    lipr = lipr1
if (ip == '5'):
    lipl = lipl5
    lipr = lipr5
if (ip == '2'):
    lipl = lipl2
    lipr = lipr2
if (ip == '8'):
    lipl = lipl8
    lipr = lipr8   

qal_nf = 'Q2L' + str(ip) 
qbl_nf = 'Q3L' + str(ip)
qar_nf = 'Q2R' + str(ip)
qbr_nf = 'Q3R' + str(ip)



####### Reading lattice files  ####################################################
nameelx,selx,betx,psix,alfx=leer_beta_mu3(latticef,'-x')
nameely,sely,bety,psiy,alfy=leer_beta_mu3(latticef,'-y')


Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms, Isbpmlr ,trying_2run4quads = inp_conditions(args,nameelx,nameely,'nominal', ip, beam,accel)
#print Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms, Isbpmlr ,trying_2run4quads
##############################handling measured lattice input#############################################
   
if ( getllm_dir != None):
    if (Iserrlattice):
        lattice_errf = apj_dir + 'lattice_err.asc'
        get_meas_lattice(getllm_dir,beam,latticef, lattice_errf)

if ( getllm_dir_omc3 != None):
    if (Iserrlattice):
        lattice_errf = apj_dir + 'lattice_err.asc'
        get_meas_lattice_omc3(getllm_dir_omc3,beam,latticef, lattice_errf)


####################################################################################################################


if (Iserrlattice):
#if( args.getllm_dir != None or args.err_lattice_file != None):
    nameelxe,selxe,betxe,psixe,alfxe=leer_beta_mu3(lattice_errf,'-x')
    nameelye,selye,betye,psiye,alfye=leer_beta_mu3(lattice_errf,'-y')
    Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms, Isbpmlr, trying_2run4quads = inp_conditions(args,nameelxe,nameelye,'measured', ip, beam,accel)
#print 'trying_2run4quads', trying_2run4quads
    #Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms = True,True,True,True

##############################handling kmod files##############################################
if (Iskmod):
    if (args.long_kmod_file != None):
        kmod_outf_l =[]
        kmod_file = in_tfs.split(',')
        for kk in range(len(kmod_file)):
            kmod_outf=tfs2outb1_v3(kmod_file[kk], apj_dir,kk)
            kmod_outf_l.append(kmod_outf)
        aver_kmod2(apj_dir, kmod_outf_l)
        ipff = apj_dir  + 'ip_aver.results'


####################################################################################################################
QLyKf_file=args.model_dir  + "Quad_KyL.txt"
if(Iskmod):
    wx, bwx = wbw(ipff,'-x',beam, ip,QLyKf_file, accel) #optical parameters inter-triplet space
    wy, bwy = wbw(ipff,'-y',beam, ip,QLyKf_file, accel)
else:
    wx = 0
    wy = 0
    ip_name = '"IP' + ip + '"'
    bwx =bety[nameely.index(ip_name)]
    bwy =bety[nameely.index(ip_name)]



#print four_quad_corr,wx, wy, bwx, bwy
    


#print "BPM names", 'bpml, bpmr, bpm_max, bpm_phase',  bpml, bpmr, bpm_max, bpm_phase
#options = ['ip_phase_bpm','left_kick_bpm','right_kick_bpm','bpm_for_avermax']
#possibleBPMs(bpm,option,ip,beam)

    


#phasx_element = psix[nameelx.index(element_seq)]
#phasy_element = psiy[nameely.index(element_seq)]
#ipff = apj_dir  + 'ip_aver.results'


#print  "kmod","BEAM", "ip", "wx", "bwx", "wy", "bwy"
#print "kmod" ,beam,ip, wx, bwx, wy, bwy
if(args.ip_phase_bpm != None):bpm_phase= '\"' + args.ip_phase_bpm + '\"'
    
if (Isbpmlr):
    bpml= '\"' + bpml_nf + '\"'
    #####lattice parameter X plane left side############################################
    betax_error_l=betx[nameelx.index(bpml)]
    Betxeql = betx[nameelx.index(bpml)]
    psix_error_l=psix[nameelx.index(bpml)]
    sx_error_l=selx[nameelx.index(bpml)] 
    sxBPM_l=sx_error_l
    betax_BPM_l=betx[nameelx.index(bpml)]# LHC
    psix_BPM_l=psix[nameelx.index(bpml)] # LHC

#####lattice parameter Y plane left side############################################
    betay_error_l=bety[nameely.index(bpml)]
    Betyeql = bety[nameely.index(bpml)]
    psiy_error_l=psiy[nameely.index(bpml)]
    sy_error_l=sely[nameely.index(bpml)] 
    syBPM_l=sy_error_l
    betay_BPM_l=bety[nameely.index(bpml)]# LHC
    psiy_BPM_l=psiy[nameely.index(bpml)] # LHC
        


    bpmr= '\"' + bpmr_nf + '\"'
    #####lattice parameter X plane right side############################################ 
    betax_error_r=betx[nameelx.index(bpmr)]
    Betxeqr = betx[nameelx.index(bpmr)]
    psix_error_r=psix[nameelx.index(bpmr)]
    sx_error_r=selx[nameelx.index(bpmr)] 
    sxBPM_r=sx_error_r
    betax_BPM_r=betx[nameelx.index(bpmr)]# LHC
    psix_BPM_r=psix[nameelx.index(bpmr)] # LHC

    #####lattice parameter Y plane right side############################################ 
    betay_error_r=bety[nameely.index(bpmr)]
    Betyeqr = bety[nameely.index(bpmr)]
    psiy_error_r=psiy[nameely.index(bpmr)]
    sy_error_r=sely[nameely.index(bpmr)] 
    syBPM_r=sy_error_r
    betay_BPM_r=bety[nameely.index(bpmr)]# LHC
    psiy_BPM_r=psiy[nameely.index(bpmr)] # LHC


#model
#tbts

    
bpm_max_ns='"'+bpm_max+'"'

if (bpm_max_ns not in nameelx or bpm_max_ns not in nameely):
    print(bpm_max, 'is not a valid BPM for "bpm_for_avermax (-ab)". Instead, use', possibleBPMs(bpm_max,"bpm_for_avermax",ip,beam,accel))
    exit()

#####lattice parameter X plane right side############################################
betax_error=betx[nameelx.index(bpm_max_ns)]
Betxeq = betx[nameelx.index(bpm_max_ns)]
psix_error=psix[nameelx.index(bpm_max_ns)]
sx_error=selx[nameelx.index(bpm_max_ns)] 
sxBPM=sx_error
betax_BPM=betx[nameelx.index(bpm_max_ns)]# LHC
psix_BPM=psix[nameelx.index(bpm_max_ns)] # LHC

#####lattice parameter Y plane right side############################################ 
betay_error=bety[nameely.index(bpm_max_ns)]
Betyeq = bety[nameely.index(bpm_max_ns)]
psiy_error=psiy[nameely.index(bpm_max_ns)]
sy_error=sely[nameely.index(bpm_max_ns)] 
syBPM=sy_error
betay_BPM=bety[nameely.index(bpm_max_ns)]# LHC
psiy_BPM=psiy[nameely.index(bpm_max_ns)] # LHC

##################################################################
if(trying_2run4quads):
    print("WARNING: If correction estimates with 4 quads are desired, the following files and options should be input:")
    print("measure lattice files (using -gd or -ml)")
    print("kmodulations files (using -lk or -sk)")
    print("ip_phase_bpm (using -pb)")
    print("left_kick_bpm (using -lp)")
    print("right_kick_bpm (using -rp)")
###############################################################




##########################################INTEGRALS##################################
integrals = leer_filedatos_v01(intf)

#variables for corrector integrals (left triplet)
IntBetxQal = 0
IntBetyQal = 0
IntBetxQbl = 0
IntBetyQbl = 0
Intbetxbety_l = 0

PhixQal = 0
PhiyQal = 0
PhixQbl = 0
PhiyQbl = 0


#variables for corrector integrals (right triplet)
IntBetxQar = 0
IntBetyQar = 0
IntBetxQbr = 0
IntBetyQbr = 0
Intbetxbety_r = 0

PhixQar = 0
PhiyQar = 0
PhixQbr = 0
PhiyQbr = 0

IntBetxQ1L = 0
IntBetyQ1L = 0
IntBetxQ1R = 0
IntBetyQ1R = 0


for j in integrals:
      #Integrals of Quads for correction (left triplet)
    if (qal_nf in j ):
      IntBetxQal= IntBetxQal  + float(j[1])
      IntBetyQal= IntBetyQal  + float(j[2])
      PhixQal= PhixQal  + float(j[4])
      PhiyQal= PhiyQal  + float(j[5])            
    if (qbl_nf in j ):
      IntBetxQbl= IntBetxQbl  + float(j[1])
      IntBetyQbl=  IntBetyQbl + float(j[2])
      PhixQbl= PhixQbl  + float(j[4])
      PhiyQbl=  PhiyQbl + float(j[5])       

    if (qslint in j ):
      Intbetxbety_l= Intbetxbety_l  + float(j[1])




      #integrals of Quads for correction (right triplet)
    if (qar_nf in j ):
      IntBetxQar= IntBetxQar  + float(j[1])
      IntBetyQar= IntBetyQar  + float(j[2])
      PhixQar= PhixQar  + float(j[4])
      PhiyQar= PhiyQar  + float(j[5])         
    if (qbr_nf in j ):
      IntBetxQbr= IntBetxQbr  + float(j[1])
      IntBetyQbr= IntBetyQbr  + float(j[2])
      PhixQbr= PhixQbr  + float(j[4])
      PhiyQbr= PhiyQbr  + float(j[5])         
    if (qsrint in j ):
      Intbetxbety_r= Intbetxbety_r  + float(j[1])
    if  ('Q1L'+ip in j):
        IntBetxQ1L=float(j[1])
        IntBetyQ1L=float(j[2])
        #print IntBetxQ1L, IntBetyQ1L
        if (args.accel == 'hl_lhc' and (ip =='1' or ip == '5')):
                IntBetxQ1L=IntBetxQ1L + float(j[1])
                IntBetyQ1L=IntBetyQ1L + float(j[2])
    if  ('Q1R'+ip in j):
        IntBetxQ1R=float(j[1])
        IntBetyQ1R=float(j[2])
        if (args.accel == 'hl_lhc' and (ip =='1' or ip == '5')):
                IntBetxQ1R=IntBetxQ1R + float(j[1])
                IntBetyQ1R=IntBetyQ1R + float(j[2])
 
      #Integrals no common region (only IR1 )
    if ('Q4L1' in j):
        IntBetxQ4L=float(j[1])
        IntBetyQ4L=float(j[2])
        PhixQ4L=float(j[4])
        PhiyQ4L=float(j[5])
    if ('Q5L1' in j):
        IntBetxQ5L=float(j[1])
        IntBetyQ5L=float(j[2])
        PhixQ5L=float(j[4])
        PhiyQ5L=float(j[5])
    if ('Q6L1' in j):
        IntBetxQ6L=float(j[1])
        IntBetyQ6L=float(j[2])
        PhixQ6L=float(j[4])
        PhiyQ6L=float(j[5])

    if ('Q4R1' in j):
        IntBetxQ4R=float(j[1])
        IntBetyQ4R=float(j[2])
        PhixQ4R=float(j[4])
        PhiyQ4R=float(j[5])
    if ('Q5R1' in j):
        IntBetxQ5R=float(j[1])
        IntBetyQ5R=float(j[2])
        PhixQ5R=float(j[4])
        PhiyQ5R=float(j[5])
    if ('Q6R1' in j):
        IntBetxQ6R=float(j[1])
        IntBetyQ6R=float(j[2])
        PhixQ6R=float(j[4])
        PhiyQ6R=float(j[5])
PhixQal = PhixQal/2
PhiyQal = PhiyQal/2
PhixQar = PhixQar/2
PhiyQar = PhiyQar/2
      
############################# handling bpm_data with different formats############################

        

if (args.orig_bpmdata != None): tbt_files = args.orig_bpmdata
if (args.clean_bpmdata != None): tbt_files = args.clean_bpmdata
if (args.shifted_bpmdata != None): tbt_files = args.shifted_bpmdata
if (args.aver_bpmdata != None): tbt_files = args.aver_bpmdata


    
    


tbt_file = tbt_files.split(',')

######put code associated to -fa flag  here   

        


print("\n")
print("\n")

bb_2corrs = []
aa_2corrs = []



bb_4corrsl = []
aa_4corrsl = []
bb_4corrsr = []
aa_4corrsr = []

for tbt_i in tbt_file:
    print("########## ANALYSIS OF ", tbt_i, "STARTS #######################")
    if (args.orig_bpmdata != None and not args.force_averages):
        datatype = "orig_bpmdata"
        orb_apj_dir, name_tbt = create_tbt_analysis_dir(datatype,tbt_i, apj_dir)
        hole_tbt_in = '--file=' + tbt_i
        hole_model = '--model=' + orig_model
        hole_output =  '--outputdir='  + orb_apj_dir
        print("Creating clean TBT from",  tbt_i, "...")
        call(['/afs/cern.ch/eng/sl/lintrack/miniconda2/bin/python','/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/hole_in_one.py',hole_tbt_in, hole_model,hole_output ,'clean','--svd_mode=numpy', '--startturn=1', '--endturn=6600', '--sing_val=12', '--pk-2-pk=0.00001', '--max-peak-cut=20', '--single_svd_bpm_threshold=0.925', '--bad_bpms=', '--wrong_polarity_bpms=', '--write_clean'])
        tbt_clean  = orb_apj_dir+ name_tbt + '.sdds.clean'
        tbt_shifted =orb_apj_dir+ name_tbt + '.sdds.shifted'
        print(tbt_clean, "was created")
        print("\n")
        print("Creating shifted TBT from  clean TBT...")        
        shift_exp_TBT(tbt_clean, latticef, tbt_shifted, beam)
        print(tbt_shifted, "was created")
        print("\n")
        print("Creating avermax trajectory from shifted TBT file ...")
        
        MT2avermax(tbt_shifted,bpm_max,latticef,orb_apj_dir)
        print("\n")
        call(['rm', '-f', tbt_shifted])
        call(['rm', '-f', tbt_clean])

        
    if (args.clean_bpmdata != None and not args.force_averages):
        datatype = "clean_bpmdata"
        orb_apj_dir, name_tbt = create_tbt_analysis_dir(datatype,tbt_i, apj_dir)
        tbt_clean  = tbt_i
        tbt_shifted =orb_apj_dir+ name_tbt + '.sdds.shifted'
        print("Creating shifted TBT from", tbt_i, "...")
        shift_exp_TBT(tbt_clean, latticef, tbt_shifted, beam)
        print(tbt_shifted, "was created")
        print("\n")
        print("Creating avermax trajectory from shifted TBT file...") 
        
        MT2avermax(tbt_shifted,bpm_max,latticef,orb_apj_dir)
        print("\n")
        #call(['rm', '-f', tbt_shifted])


    if (args.shifted_bpmdata != None and not args.force_averages):
        datatype = "shifted_bpmdata"    
        orb_apj_dir, name_tbt = create_tbt_analysis_dir(datatype,tbt_i, apj_dir)
        tbt_shifted =tbt_i
        print("Creating avermax trajectory from",  tbt_i, "...") 
        
        MT2avermax(tbt_shifted,bpm_max,latticef,orb_apj_dir)
        print("\n")

        
    if (args.aver_bpmdata != None or args.force_averages):
        if (args.force_averages):
            if (args.orig_bpmdata != None): datatype = "orig_bpmdata"
            if (args.clean_bpmdata != None): datatype = "clean_bpmdata"
            if (args.shifted_bpmdata != None): datatype = "shifted_bpmdata"
            if (args.aver_bpmdata != None):  datatype = "aver_bpmdata"
            orb_apj_dir, name_tbt = create_tbt_analysis_dir(datatype,tbt_i, apj_dir)
            tbt_i = orb_apj_dir + 'avermax.sdds.new'
            if (not os.path.exists(tbt_i)):
                print("avermax trajectory", tbt_i, "have not been created. First, run the program without the -fa flag")
                exit()

        datatype = "aver_bpmdata"
        orb_apj_dir, name_tbt = create_tbt_analysis_dir(datatype,tbt_i, apj_dir)
        new_aver = orb_apj_dir + 'avermax.sdds.new'
        if(not os.path.exists(new_aver)):
           call(['cp',tbt_i,new_aver])
    name_orbitmax = orb_apj_dir + 'avermax.sdds.new'
    if (args.aver_bpmdata == None): print("avermax trajectory",name_orbitmax , "was created")
    #print 'name_orbitmax', name_orbitmax



#if (args.force_averages):
#    tbt_file_n = []
#    for tbt_i in tbt_file:
#        full_path=os.path.abspath(tbt_i)
#       
#        tbt_i_n = dir_name + '/avermax.sdds.new'
#        tbt_file_n.append(tbt_i_n)
#    tbt_file = tbt_file_n
#    print tbt_file
        
        
###########################Estimating quadrupole components from trajectories with maximum in one plane###########################################################################
#Files created:
#for the two max trajectories (nofilt 5 columns, others 7 columns)
#H/VmaxAPJaction_nofilt.sdds      Action obtained with nominal lattice and no filter 
#H/VmaxAPJphase_nofilt.sdds       Phase obtained with nominal lattice and no filter 
#H/VmaxAction_nofilt.sdds         Action obtained with real lattice  and no filter 
#H/VmaxPhase_nofilt.sdds          Phase obtained with real lattice and no filter 
#H/VmaxAPJaction.sdds             Action obtained with nominal lattice and filter 
#H/VmaxAPJphase.sdds              Phase obtained with nominal lattice and  filter 
#H/VmaxAction.sdds                Action obtained with real lattice  and filter 
#H/VmaxPhase.sdds                 Phase obtained with real lattice and  filter 


    ##############Read average trajectory with max in horizontal plane and average trajectory with max in vertical plane#####

    orbitmax = open(name_orbitmax,'r')
    print("Reading avermax trajectory...") 
    diforbx, sort_linesx = traj_data(orbitmax, nameelx,'-x')

    orbitmax = open(name_orbitmax,'r')
    diforby, sort_linesy = traj_data(orbitmax, nameely,'-y')

    bpms_averx = []
    bpms_avery = []
    for jj in sort_linesx:
         bpms_averx.append(jj[1])
    for jj in sort_linesy:
         bpms_avery.append(jj[1])
    

    if ((bpm_max not in bpms_averx) or (bpm_max not in bpms_avery)):
        print(bpm_max , 'is not in',name_orbitmax,'. Position data ( in either x or/and y plane) for that BPM might not be available. Instead, Re-run the program and use one of the following BPMs',possibleBPMs(bpm_max,"bpm_for_avermax",ip,beam,accel), 'for "bpm_for_avermax (-ab)"')
        exit()
    if (Isbpms):
        
        
        if ((bpml_nf not in bpms_averx) or (bpml_nf not in bpms_avery)):
            print(bpml_nf , 'is not in',name_orbitmax,'. Position data ( in either x or/and y plane) for that BPM might not be available. Instead, Re-run the program and use one of the following BPMs',possibleBPMs(bpml_nf,"left_kick_bpm",ip,beam,accel), 'for "left_kick_bpm (-lb)"')
            exit()

        if ((bpmr_nf not in bpms_averx) or (bpmr_nf not in bpms_avery)):
            print(bpmr_nf , 'is not in',name_orbitmax,'. Position data ( in either x or/and y plane) for that BPM might not be available. Instead, Re-run the program and use one of the following BPMs',possibleBPMs(bpmr_nf,"right_kick_bpm",ip,beam,accel), 'for "right_kick_bpm (-rb)"')
            exit()


        


    #####################################################
    #Actions, phases and kicks from horizontal average trajectory with maximum in horizontal plane.
    ip_j_kmod =[]
    ip_ph_kmod =[]
    ip_j_bpmsw =[]
    ip_ph_bpmsw =[]    
    nsigma=2.0
    sdds_col = 4
    plane = '-x'
    #bpm_max_ns
#Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms
    
    #print "AVERAGE TRAYECTORY WITH MAX IN HORIZONTAL PLANE"
    #print "horizontal"
    if(Isbpms):
        bpm1 = bpml
        bpm2 = bpmr
        bpm_phase_i = bpm_phase
    else:
        bpm1 = bpm_max_ns
        bpm2 = bpm_max_ns
        bpm_phase_i = ''
        
    if(Iserrlattice):
        selxe_i = selxe
        nameelxe_i = nameelxe
        psixe_i = psixe
        betxe_i = betxe
        selye_i = selye
        nameelye_i = nameelye
        psiye_i = psiye
        betye_i = betye
    else:
        selxe_i = selx
        nameelxe_i = nameelx
        psixe_i = psix
        betxe_i = betx
        selye_i = sely
        nameelye_i = nameely
        psiye_i = psiy
        betye_i = bety
        
    APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,averactant, averphaseant, act_trip_bpmsw, phase_trip_bpmsw, averactdesp, averphasedesp, x_BPM_l, x_BPM_r =Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpm1,bpm2,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe_i,nameelxe_i,psixe_i,betxe_i,wx,bwx,s_acdipole,plane,beam,bpm_phase_i,ip, 'bpmsw', accel)
    
    icxf = open(orb_apj_dir+'icx.madx','w')
    if (beam == '1'):
        btx = betx[nameelx.index('"BPM.8L4.B1"')]
        alx = alfx[nameelx.index('"BPM.8L4.B1"')]
    if (beam == '2'):
        btx = betx[nameelx.index('"BPM.11L4.B2"')]
        alx = alfx[nameelx.index('"BPM.11L4.B2"')]
        
    zx=-sqrt(2*averactant*btx)*sin(averphaseant)
    zpx=sqrt(2*averactant/btx) *(alx*sin(averphaseant)+cos(averphaseant) )
    print('btx = ', btx, ';', file=icxf)
    print('alx = ', alx, ';', file=icxf)
    print('zx = ', zx, ';', file=icxf)
    print('zpx = ', zpx, ';', file=icxf)
    print('zy = ', '0.0000001', ';', file=icxf)
    print('zpy = ', '0', ';', file=icxf)
    icxf.close()

    ip_j_bpmsw.append([0,nameelx[nameelx.index('"'+'IP'+ip+ '"')],selx[nameelx.index('"'+'IP'+ip+'"')],act_trip_bpmsw])
    ip_ph_bpmsw.append([0,nameelx[nameelx.index('"'+'IP'+ip+ '"')],selx[nameelx.index('"'+'IP'+ip+'"')],phase_trip_bpmsw])
    if(Isbpmlr): 
        kickxl_bpmsw=kick_strength_v03(averactant,act_trip_bpmsw,averphaseant,phase_trip_bpmsw,betx[nameelx.index(bpml)],psix[nameelx.index(bpml)])
        kickxr_bpmsw=kick_strength_v03(act_trip_bpmsw,averactdesp,phase_trip_bpmsw,averphasedesp,betx[nameelx.index(bpmr)],psix[nameelx.index(bpmr)])
        if (beam == '2'):
            kickxl_bpmsw = -kickxl_bpmsw
            kickxr_bpmsw = -kickxr_bpmsw
        xeq_l=posestim(betax_error_l,betax_BPM_l,psix_error_l,psix_BPM_l,averphaseant,x_BPM_l)
        xeq_r=posestim(betax_error_r,betax_BPM_r,psix_error_r,psix_BPM_r,averphasedesp,x_BPM_r)
        B1xeql_bpmsw = -kickxl_bpmsw/xeq_l
        B1xeqr_bpmsw = -kickxr_bpmsw/xeq_r
        print(B1xeql_bpmsw,Betxeql, IntBetxQal,  IntBetxQbl, IntBetxQ4L,  IntBetxQ5L, IntBetxQ6L,PhixQal,  PhixQbl, PhixQ4L,  PhixQ5L, PhixQ6L, 0, IntBetxQ1L, file=resxlf_bpmsw)
        print(B1xeqr_bpmsw,Betxeqr, IntBetxQar,  IntBetxQbr, IntBetxQ4R,  IntBetxQ5R, IntBetxQ6R,PhixQar,  PhixQbr, PhixQ4R,  PhixQ5R, PhixQ6R, 0, IntBetxQ1R, file=resxrf_bpmsw)

    if (Is_ip_ap_kmod):
        #########betas at  bpmsws############
        betlx=bwx+((lipl+wx)**2)/bwx
        betrx=bwx+((lipr-wx)**2)/bwx 
        betly=bwy+((lipl+wy)**2)/bwy
        betry=bwy+((lipr-wy)**2)/bwy
        print("betax and betay at bpmswl (to be compared with lsa_results.tfs)", betlx, betly)
        print("betax and betay at bpmswr (to be compared with lsa_results.tfs)", betrx, betry)
        ######################################
        APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,averactant, averphaseant, act_trip_kmod, phase_trip_kmod, averactdesp, averphasedesp, x_BPM_l, x_BPM_r =Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpml,bpmr,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe,nameelxe,psixe,betxe,wx,bwx,s_acdipole,plane,beam,bpm_phase,ip, 'kmod', accel)
        ip_j_kmod.append([0,nameelx[nameelx.index('"'+'IP'+ip+ '"')],selx[nameelx.index('"'+'IP'+ip+'"')],act_trip_kmod])
        ip_ph_kmod.append([0,nameelx[nameelx.index('"'+'IP'+ip+ '"')],selx[nameelx.index('"'+'IP'+ip+'"')],phase_trip_kmod])
        kickxl_kmod=kick_strength_v03(averactant,act_trip_kmod,averphaseant,phase_trip_kmod,betx[nameelx.index(bpml)],psix[nameelx.index(bpml)])
        kickxr_kmod=kick_strength_v03(act_trip_kmod,averactdesp,phase_trip_kmod,averphasedesp,betx[nameelx.index(bpmr)],psix[nameelx.index(bpmr)])
        if (beam == '2'):
            kickxl_kmod = -kickxl_kmod
            kickxr_kmod = -kickxr_kmod
        #Position estimation
        xeq_l=posestim(betax_error_l,betax_BPM_l,psix_error_l,psix_BPM_l,averphaseant,x_BPM_l)
        xeq_r=posestim(betax_error_r,betax_BPM_r,psix_error_r,psix_BPM_r,averphasedesp,x_BPM_r)
        B1xeql_kmod = -kickxl_kmod/xeq_l
        B1xeqr_kmod = -kickxr_kmod/xeq_r    
        print(B1xeql_kmod,Betxeql, IntBetxQal,  IntBetxQbl, IntBetxQ4L,  IntBetxQ5L, IntBetxQ6L,PhixQal,  PhixQbl, PhixQ4L,  PhixQ5L, PhixQ6L, 0, IntBetxQ1L, file=resxlf)
        print(B1xeqr_kmod,Betxeqr, IntBetxQar,  IntBetxQbr, IntBetxQ4R,  IntBetxQ5R, IntBetxQ6R,PhixQar,  PhixQbr, PhixQ4R,  PhixQ5R, PhixQ6R, 0, IntBetxQ1R, file=resxrf)
        bb_4corrsr.append(-B1xeqr_kmod*Betxeqr)
        aa_4corrsr.append([IntBetxQar,IntBetxQbr])
        bb_4corrsl.append(-B1xeql_kmod*Betxeql)
        aa_4corrsl.append([IntBetxQal,IntBetxQbl])
       #previous run of Arc_Trip_APs must be with bpml=bpm_max_ns

    
    kickx=kick_strength_v03(averactant,averactdesp,averphaseant,averphasedesp,betx[nameelx.index(bpm_max_ns)],psix[nameelx.index(bpm_max_ns)])
    if (beam == '2'):    kickx = -kickx
    if (Isbpms):
        if ('L' in bpm_max_ns ):xeq=posestim(betax_error,betax_BPM_l,psix_error,psix_BPM_l,averphaseant,x_BPM_l)
        if ('R' in bpm_max_ns ):xeq=posestim(betax_error,betax_BPM_r,psix_error,psix_BPM_r,averphaseant,x_BPM_r) 
    else: xeq=posestim(betax_error,betax_BPM,psix_error,psix_BPM,averphaseant,x_BPM_l)
    
    
    B1xeq = -kickx/xeq
    print(B1xeq, Betxeq, IntBetxQal,  IntBetxQar, IntBetxQ4L,  IntBetxQ5L, IntBetxQ6L,PhixQal,  PhixQbl, PhixQ4L,  PhixQ5L, PhixQ6L, 0, IntBetxQ1L, B1xeq*Betxeq, file=resxf)
    bb_2corrs.append(-B1xeq*Betxeq)
    aa_2corrs.append([IntBetxQal,  IntBetxQar])

    #print "averages", averactant, averphaseant, averactdesp, averphasedesp, actlx[0][4], phaselx[0][4], act_trip, phase_trip

    #Actions, phases and kicks from vertical average trajectory with maximum in horizontal plane.

    nsigma=2.0
    sdds_col = 4
    plane = '-y'

    #print "vertical"
    APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely, averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, y_BPM_l, y_BPM_r =Arc_Trip_APs(sdds_col, sort_linesy,diforby,bpm1,bpm2,sely,nameely,psiy,bety,sL_begin,sR_end,nsigma,selye_i,nameelye_i,psiye_i,betye_i,wy,bwy,s_acdipole,plane,beam,bpm_phase_i,ip, 'bpmsw', accel)



    APJact_nfl = APJact_nflx + APJact_nfly
    APJphase_nfl = APJphase_nflx + APJphase_nfly
    act_nfl = act_nflx + act_nfly
    phase_nfl = phase_nflx + phase_nfly
    APJactl = APJactlx + APJactly
    APJphasel = APJphaselx + APJphasely
    actl = actlx + actly
    phasel = phaselx + phasely


    write_file(orb_apj_dir + 'HmaxAPJaction_nofilt.sdds',APJact_nfl,'w')
    write_file(orb_apj_dir + 'HmaxAPJphase_nofilt.sdds',APJphase_nfl,'w')
    if(args.getllm_dir_omc3 != None or args.getllm_dir != None or args.err_lattice_file != None):write_file(orb_apj_dir + 'HmaxAction_nofilt.sdds', act_nfl,'w')
    else:
        file_to_remove =orb_apj_dir + 'HmaxAction_nofilt.sdds' 
        call(['rm', '-f',file_to_remove])
    if(args.getllm_dir_omc3 != None or args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'HmaxPhase_nofilt.sdds', phase_nfl,'w')
    else:
        file_to_remove =orb_apj_dir + 'HmaxPhase_nofilt.sdds' 
        call(['rm', '-f',file_to_remove])
        

    write_file(orb_apj_dir + 'HmaxAPJaction.sdds',APJactl,'w')
    write_file(orb_apj_dir + 'HmaxAPJphase.sdds',APJphasel,'w')
    if(args.getllm_dir_omc3 != None or args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'HmaxAction.sdds', actl,'w')
    else:
        file_to_remove =orb_apj_dir + 'HmaxAction.sdds' 
        call(['rm','-f',file_to_remove])
    if(args.getllm_dir_omc3 != None or  args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'HmaxPhase.sdds', phasel,'w')
    else:
        file_to_remove =orb_apj_dir + 'HmaxPhase.sdds' 
        call(['rm', '-f',file_to_remove])

    
    #Actions, phases and kicks from horizontal average trajectory with maximum in vertical plane.
    nsigma=2.0
    sdds_col = 5
    plane = '-x'


    #print "AVERAGE TRAYECTORY WITH MAX IN VERTICAL PLANE"
    #print "horizontal"
    APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, x_BPM_l, x_BPM_r =Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpm1,bpm2,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe_i,nameelxe_i,psixe_i,betxe_i,wx,bwx,s_acdipole,plane,beam,bpm_phase_i,ip, 'bpmsw', accel)






    #Actions, phases and kicks from vertical average trajectory with maximum in vertical plane.
    nsigma=2.0
    sdds_col = 5
    plane = '-y'

    #print "vertical"
    APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely,averactant, averphaseant, act_trip_bpmsw, phase_trip_bpmsw, averactdesp, averphasedesp, y_BPM_l, y_BPM_r =Arc_Trip_APs(sdds_col, sort_linesy,diforby,bpm1,bpm2,sely,nameely,psiy,bety,sL_begin,sR_end,nsigma,selye_i,nameelye_i,psiye_i,betye_i,wy,bwy,s_acdipole,plane,beam,bpm_phase_i,ip, 'bpmsw', accel)
    icyf = open(orb_apj_dir+'icy.madx','w')
    if (beam == '1'):
        bty = bety[nameely.index('"BPM.8L4.B1"')]
        aly = alfy[nameely.index('"BPM.8L4.B1"')]
    if (beam == '2'):
        bty = bety[nameely.index('"BPM.11L4.B2"')]
        aly = alfy[nameely.index('"BPM.11L4.B2"')]
    zy=-sqrt(2*averactant*bty)*sin(averphaseant)
    zpy=sqrt(2*averactant/bty) *(aly*sin(averphaseant)+cos(averphaseant) )
    print('bty = ', bty, ';', file=icyf)
    print('aly = ', aly, ';', file=icyf)
    print('zy = ', zy, ';', file=icyf)
    print('zpy = ', zpy, ';', file=icyf)
    print('zx = ', '0.0000001', ';', file=icyf)
    print('zpx = ', '0', ';', file=icyf)
    icyf.close()
    ip_j_bpmsw.append([1,nameely[nameely.index('"'+'IP'+ip+'"')],sely[nameely.index('"'+'IP'+ip+'"')],act_trip_bpmsw])
    ip_ph_bpmsw.append([1,nameely[nameely.index('"'+'IP'+ip+'"')],sely[nameely.index('"'+'IP'+ip+'"')],phase_trip_bpmsw])
    if(Isbpmlr): 
        kickyl_bpmsw=kick_strength_v03(averactant,act_trip_bpmsw,averphaseant,phase_trip_bpmsw,bety[nameely.index(bpml)],psiy[nameely.index(bpml)])
        kickyr_bpmsw=kick_strength_v03(act_trip_bpmsw,averactdesp,phase_trip_bpmsw,averphasedesp,bety[nameely.index(bpmr)],psiy[nameely.index(bpmr)])
        if (beam == '2'):
            kickyl_bpmsw = -kickyl_bpmsw
            kickyr_bpmsw = -kickyr_bpmsw
        yeq_l=posestim(betay_error_l,betay_BPM_l,psiy_error_l,psiy_BPM_l,averphaseant,y_BPM_l)
        yeq_r=posestim(betay_error_r,betay_BPM_r,psiy_error_r,psiy_BPM_r,averphasedesp,y_BPM_r)
        B1yeql_bpmsw = kickyl_bpmsw/yeq_l
        B1yeqr_bpmsw = kickyr_bpmsw/yeq_r
        print(B1yeql_bpmsw,Betyeql, IntBetyQal,  IntBetyQbl, IntBetyQ4L,  IntBetyQ5L, IntBetyQ6L,PhiyQal,  PhiyQbl, PhiyQ4L,  PhiyQ5L, PhiyQ6L, 0, IntBetyQ1L, file=resylf_bpmsw)
        print(B1yeqr_bpmsw,Betyeqr, IntBetyQar,  IntBetyQbr, IntBetyQ4R,  IntBetyQ5R, IntBetyQ6R,PhiyQar,  PhiyQbr, PhiyQ4R,  PhiyQ5R, PhiyQ6R, 0, IntBetyQ1R, file=resyrf_bpmsw)


    if (Is_ip_ap_kmod):
            APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely,averactant, averphaseant, act_trip_kmod, phase_trip_kmod, averactdesp, averphasedesp, y_BPM_l, y_BPM_r =Arc_Trip_APs(sdds_col, sort_linesy,diforby,bpml,bpmr,sely,nameely,psiy,bety,sL_begin,sR_end,nsigma,selye,nameelye,psiye,betye,wy,bwy,s_acdipole,plane,beam,bpm_phase,ip, 'kmod', accel)
            ip_j_kmod.append([1,nameely[nameely.index('"'+'IP'+ip+'"')],sely[nameely.index('"'+'IP'+ip+'"')],act_trip_kmod])
            ip_ph_kmod.append([1,nameely[nameely.index('"'+'IP'+ip+'"')],sely[nameely.index('"'+'IP'+ip+'"')],phase_trip_kmod])
            kickyl_kmod=kick_strength_v03(averactant,act_trip_kmod,averphaseant,phase_trip_kmod,bety[nameely.index(bpml)],psiy[nameely.index(bpml)])
            kickyr_kmod=kick_strength_v03(act_trip_kmod,averactdesp,phase_trip_kmod,averphasedesp,bety[nameely.index(bpmr)],psiy[nameely.index(bpmr)])
            if (beam == '2'):
                kickyl_kmod = -kickyl_kmod
                kickyr_kmod = -kickyr_kmod    
            yeq_l=posestim(betay_error_l,betay_BPM_l,psiy_error_l,psiy_BPM_l,averphaseant,y_BPM_l)
            yeq_r=posestim(betay_error_r,betay_BPM_r,psiy_error_r,psiy_BPM_r,averphasedesp,y_BPM_r)
            B1yeql_kmod = kickyl_kmod/yeq_l
            B1yeqr_kmod = kickyr_kmod/yeq_r
            print(B1yeql_kmod,Betyeql, IntBetyQal,  IntBetyQbl, IntBetyQ4L,  IntBetyQ5L, IntBetyQ6L,PhiyQal,  PhiyQbl, PhiyQ4L,  PhiyQ5L, PhiyQ6L, 0, IntBetyQ1L, file=resylf)
            print(B1yeqr_kmod,Betyeqr, IntBetyQar,  IntBetyQbr, IntBetyQ4R,  IntBetyQ5R, IntBetyQ6R,PhiyQar,  PhiyQbr, PhiyQ4R,  PhiyQ5R, PhiyQ6R, 0, IntBetyQ1R, file=resyrf)
            bb_4corrsr.append(-B1yeqr_kmod*Betyeqr)
            aa_4corrsr.append([IntBetyQar,  IntBetyQbr])
            bb_4corrsl.append(-B1yeql_kmod*Betyeql)
            aa_4corrsl.append([IntBetyQal,  IntBetyQbl])
    
    kicky=kick_strength_v03(averactant,averactdesp,averphaseant,averphasedesp,bety[nameely.index(bpm_max_ns)],psiy[nameely.index(bpm_max_ns)])
    if (beam == '2'): kicky = -kicky
    if (Isbpms):
        if ('L' in bpm_max_ns): yeq=posestim(betay_error,betay_BPM_l,psiy_error,psiy_BPM_l,averphaseant,y_BPM_l)
        if ('R' in bpm_max_ns): yeq=posestim(betay_error,betay_BPM_r,psiy_error,psiy_BPM_r,averphaseant,y_BPM_r)
    else: yeq=posestim(betay_error,betay_BPM,psiy_error,psiy_BPM,averphaseant,y_BPM_l)
    

    B1yeq = kicky/yeq
 

    print(B1yeq,Betyeq, IntBetyQal,  IntBetyQar, IntBetyQ4L,  IntBetyQ5L, IntBetyQ6L,PhiyQal,  PhiyQbl, PhiyQ4L,  PhiyQ5L, PhiyQ6L, 0, IntBetyQ1L , B1yeq*Betyeq, file=resyf)
    bb_2corrs.append(-B1yeq*Betyeq)
    aa_2corrs.append([IntBetyQal,  IntBetyQar])
    #dphim = dphi_rad()
    #
    #
    #
    #
    APJact_nfl = APJact_nflx + APJact_nfly
    APJphase_nfl = APJphase_nflx + APJphase_nfly
    act_nfl = act_nflx + act_nfly
    phase_nfl = phase_nflx + phase_nfly
    APJactl = APJactlx + APJactly
    APJphasel = APJphaselx + APJphasely
    actl = actlx + actly
    phasel = phaselx + phasely


    write_file(orb_apj_dir + 'VmaxAPJaction_nofilt.sdds',APJact_nfl,'w')
    write_file(orb_apj_dir + 'VmaxAPJphase_nofilt.sdds',APJphase_nfl,'w')
    if(args.getllm_dir_omc3 != None or  args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'VmaxAction_nofilt.sdds', act_nfl,'w')
    else:
        file_to_remove =orb_apj_dir + 'VmaxAction_nofilt.sdds' 
        call(['rm', '-f',file_to_remove])

    if(args.getllm_dir_omc3 != None or args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'VmaxPhase_nofilt.sdds', phase_nfl,'w')
    else:
        file_to_remove =orb_apj_dir + 'VmaxPhase_nofilt.sdds' 
        call(['rm', '-f',file_to_remove])

    write_file(orb_apj_dir + 'VmaxAPJaction.sdds',APJactl,'w')
    write_file(orb_apj_dir + 'VmaxAPJphase.sdds',APJphasel,'w')
    if(args.getllm_dir_omc3 != None or args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'VmaxAction.sdds', actl,'w')
    else:
        file_to_remove =orb_apj_dir + 'VmaxAction.sdds' 
        call(['rm', '-f',file_to_remove])
    if(args.getllm_dir_omc3 != None or args.getllm_dir != None or args.err_lattice_file != None): write_file(orb_apj_dir + 'VmaxPhase.sdds', phasel,'w')
    else:
        file_to_remove =orb_apj_dir + 'VmaxPhase.sdds' 
        call(['rm', '-f',file_to_remove])

    if (Is_ip_ap_kmod):
        write_file(orb_apj_dir + 'IPaction.sdds', ip_j_kmod,'w')
        write_file(orb_apj_dir + 'IPphase.sdds', ip_ph_kmod,'w')
    else:
        file_to_remove =orb_apj_dir + 'IPaction.sdds'
        call(['rm', '-f',file_to_remove])
        file_to_remove =orb_apj_dir + 'IPphase.sdds'
        call(['rm', '-f',file_to_remove])

    if (Isbpmlr):
        write_file(orb_apj_dir + 'IPaction_bpmsw.sdds', ip_j_bpmsw,'w')
        write_file(orb_apj_dir + 'IPphase_bpmsw.sdds', ip_ph_bpmsw,'w')
    else:
        file_to_remove =orb_apj_dir + 'IPaction_bpmsw.sdds'
        call(['rm', '-f',file_to_remove])
        file_to_remove =orb_apj_dir + 'IPphase_bpmsw.sdds'
        call(['rm', '-f',file_to_remove])                

    #Is_ip_ap_kmod, Iskmod ,  Iserrlattice, Isbpms
    #print "averages", averactant, averphaseant, averactdesp, averphasedesp, actly[0][4], phasely[0][4], act_trip, phase_trip
    if(Is_ip_ap_kmod):
        corrq2l_kmod = (B1yeql_kmod*Betyeql*IntBetxQbl - B1xeql_kmod*Betxeql*IntBetyQbl)/(IntBetxQal*IntBetyQbl - IntBetxQbl*IntBetyQal)
        corrq3l_kmod = (B1xeql_kmod*Betxeql*IntBetyQal - B1yeql_kmod*Betyeql*IntBetxQal)/(IntBetxQal*IntBetyQbl - IntBetxQbl*IntBetyQal)

        corrq2r_kmod = (B1yeqr_kmod*Betyeqr*IntBetxQbr - B1xeqr_kmod*Betxeqr*IntBetyQbr)/(IntBetxQar*IntBetyQbr - IntBetxQbr*IntBetyQar)
        corrq3r_kmod = (B1xeqr_kmod*Betxeqr*IntBetyQar - B1yeqr_kmod*Betyeqr*IntBetxQar)/(IntBetxQar*IntBetyQbr - IntBetxQbr*IntBetyQar)
        print('{:^8.6}'.format(corrq2l_kmod), '{:^8.6}'.format(corrq3l_kmod), '{:^8.6}'.format(corrq2r_kmod), '{:^8.6}'.format(corrq3r_kmod), file=fcorrs_kmod)
        print(" corrections with 4 quads per IR")
        print(corrq2l_name, corrq2l_kmod)
        print(corrq3l_name , corrq3l_kmod)
        print(corrq2r_name , corrq2r_kmod)
        print(corrq3r_name , corrq3r_kmod)
        print("\n")
    print(" corrections with 2 quads per IR")
    corrq2l = (B1yeq*Betyeq*IntBetxQar - B1xeq*Betxeq*IntBetyQar)/(IntBetxQal*IntBetyQar - IntBetxQar*IntBetyQal)
    corrq2r = (B1xeq*Betxeq*IntBetyQal - B1yeq*Betyeq*IntBetxQal)/(IntBetxQal*IntBetyQar - IntBetxQar*IntBetyQal)
    print(corrq2l_name, corrq2l)
    print(corrq2r_name , corrq2r)
    #print >> fcorrs_bpmsw, '{:^8.6}'.format(corrq2l_bpmsw), '{:^8.6}'.format(corrq3l_bpmsw), '{:^8.6}'.format(corrq2r_bpmsw), '{:^8.6}'.format(corrq3r_bpmsw)##Highly sensitive to BPM calibrations. Use for simulations only
    print("########## END #######################")
    print("\n")
    print("\n")

    #print "B1xl ", B1xeql
    #print "B1yl ", B1yeql
    #print "B1xr ", B1xeqr
    #print "B1yr ", B1yeqr
    #print "B1x ", B1xeq
    #print "B1y ", B1yeq

    #print corrq2l_name, corrq2l
    #print corrq3l_name , corrq3l
    #print corrq2r_name , corrq2r
    #print corrq3r_name , corrq3r

print("CORRECTIONS FOR BEAM", beam,": \n")
errq2l_ip = 'errq2l' +ip+' = '
errq2r_ip = 'errq2r' +ip+' = '
errq3l_ip = 'errq3l' +ip+' = '
errq3r_ip = 'errq3r' +ip+' = '

if (Is_ip_ap_kmod):
    aa_4corrsl_array = np.array(aa_4corrsl)
    bb_4corrsl_array = np.array(bb_4corrsl)
    aa_4corrsr_array = np.array(aa_4corrsr)
    bb_4corrsr_array = np.array(bb_4corrsr)
    corrq2l_beam_4corrs, corrq3l_beam_4corrs =  np.linalg.lstsq(aa_4corrsl_array,bb_4corrsl_array)[0]
    corrq2r_beam_4corrs, corrq3r_beam_4corrs =  np.linalg.lstsq(aa_4corrsr_array,bb_4corrsr_array)[0]
    print("corrections with 4 quads per IR")
    print(corrq2l_name, corrq2l_beam_4corrs)
    print(corrq3l_name , corrq3l_beam_4corrs)
    print(corrq2r_name , corrq2r_beam_4corrs)
    print(corrq3r_name , corrq3r_beam_4corrs)
    print(errq2l_ip ,-float(corrq2l_beam_4corrs), " ;", file=ir_4err)
    print(errq3l_ip  ,-float(corrq3l_beam_4corrs), " ;", file=ir_4err)
    print(errq2r_ip ,-float(corrq2r_beam_4corrs), " ;", file=ir_4err)
    print(errq3r_ip  ,-float(corrq3r_beam_4corrs), " ;", file=ir_4err)

print("\n")
aa_2corrs_array = np.array(aa_2corrs)
bb_2corrs_array = np.array(bb_2corrs)
corrq2l_beam_2corrs, corrq2r_beam_2corrs =  np.linalg.lstsq(aa_2corrs_array,bb_2corrs_array)[0]
print("corrections with 2 quads per IR")
print(corrq2l_name, corrq2l_beam_2corrs)
print(corrq2r_name , corrq2r_beam_2corrs)


print(errq2l_ip ,-float(corrq2l_beam_2corrs), " ;", file=ir_2err)
print(errq2r_ip  ,-float(corrq2r_beam_2corrs), " ;", file=ir_2err)

#apj_gpl_file = open(apj_dir + 'gAPJ.gpl','w')
#conventional_gpl_file = open (apj_dir + 'gConventional.gpl' ,'w')

hanff =  "'"+orb_apj_dir + 'HmaxAPJaction_nofilt.sdds' + "'"
haff =  "'"+orb_apj_dir + 'HmaxAPJaction.sdds' + "'"
vanff =  "'"+orb_apj_dir + 'VmaxAPJaction_nofilt.sdds' + "'"
vaff =  "'"+orb_apj_dir + 'VmaxAPJaction.sdds' + "'"
ipaf =  "'"+orb_apj_dir + 'IPaction.sdds' + "'"
ipaf_bpmsw =  "'"+orb_apj_dir + 'IPaction_bpmsw.sdds' + "'"

hpnff =  "'"+orb_apj_dir + 'HmaxAPJphase_nofilt.sdds' + "'"
hpff =  "'"+orb_apj_dir + 'HmaxAPJphase.sdds' + "'"
vpnff =  "'"+orb_apj_dir + 'VmaxAPJphase_nofilt.sdds' + "'"
vpff =  "'"+orb_apj_dir + 'VmaxAPJphase.sdds' + "'"
ippf =  "'"+orb_apj_dir + 'IPphase.sdds' + "'"
ippf_bpmsw =  "'"+orb_apj_dir + 'IPphase_bpmsw.sdds' + "'"

han_cff =  "'"+orb_apj_dir + 'HmaxAction_nofilt.sdds' + "'"
ha_cff =  "'"+orb_apj_dir + 'HmaxAction.sdds' + "'"
van_cff =  "'"+orb_apj_dir + 'VmaxAction_nofilt.sdds' + "'"
va_cff =  "'"+orb_apj_dir + 'VmaxAction.sdds' + "'"


hpn_cff =  "'"+orb_apj_dir + 'HmaxPhase_nofilt.sdds' + "'"
hp_cff =  "'"+orb_apj_dir + 'HmaxPhase.sdds' + "'"
vpn_cff =  "'"+orb_apj_dir + 'VmaxPhase_nofilt.sdds' + "'"
vp_cff =  "'"+orb_apj_dir + 'VmaxPhase.sdds' + "'"


#hanff = haff
#vanff = vaff
#hpnff = hpff
#vpnff = vpff

#han_cff = ha_cff
#van_cff = va_cff
#hpn_cff = hp_cff
#vpn_cff = vp_cff
print('set terminal unknown', file=apj_gpl_file)
gp_start = sL_begin/1000 - 1
gp_end = sR_end/1000 + 1
print('set xrange[',gp_start,':',gp_end,']', file=apj_gpl_file)

print('set title "APJ actionx" font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ actionx(nm)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p', haff, "u ($3/1000):($1==0?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", haff, "u ($3/1000):($1==0?($5*1e9):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ipaf, "u ($3/1000):($1==0?($4*1e9):NaN)  w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p', haff, "u ($3/1000):($1==0?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", haff, "u ($3/1000):($1==0?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ipaf_bpmsw, "u ($3/1000):($1==0?($4*1e9):NaN) w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p', haff, "u ($3/1000):($1==0?($4*1e9):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", haff, "u ($3/1000):($1==0?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('\n', file=apj_gpl_file)
print('ymax1 = GPVAL_DATA_Y_MAX*1.1', file=apj_gpl_file)
print('ymin1 = GPVAL_DATA_Y_MIN*0.9', file=apj_gpl_file)



print('set title "APJ actiony" font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ actiony(nm)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p', vaff, "u ($3/1000):($1==1?($5*1e9):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ipaf, "u ($3/1000):($1==1?($4*1e9):NaN)  w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p', vaff, "u ($3/1000):($1==1?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", vaff, "u ($3/1000):($1==1?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ipaf_bpmsw, "u ($3/1000):($1==1?($4*1e9):NaN) w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p', vaff, "u ($3/1000):($1==1?($4*1e9):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vaff, "u ($3/1000):($1==1?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('\n', file=apj_gpl_file)
print('ymax2 = GPVAL_DATA_Y_MAX*1.1', file=apj_gpl_file)
print('ymin2 = GPVAL_DATA_Y_MIN*0.9', file=apj_gpl_file)


print('set title "APJ phasex " font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ phasex(rad)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p', hpff, "u ($3/1000):($1==0?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", hpff, "u ($3/1000):($1==0?($5):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ippf, "u ($3/1000):($1==0?($4):NaN)  w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p', hpff, "u ($3/1000):($1==0?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", hpff, "u ($3/1000):($1==0?($5):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ippf_bpmsw, "u ($3/1000):($1==0?($4):NaN) w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p',hpff, "u ($3/1000):($1==0?($4):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", hpff, "u ($3/1000):($1==0?($5):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('\n', file=apj_gpl_file)
print('ymax3 = GPVAL_DATA_Y_MAX + 0.1*abs(GPVAL_DATA_Y_MAX)', file=apj_gpl_file)
print('ymin3 = GPVAL_DATA_Y_MIN - 0.1*abs(GPVAL_DATA_Y_MIN)', file=apj_gpl_file)


print('set title "APJ phasey " font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ phasey(rad)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p', vpff, "u ($3/1000):($1==1?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vpff, "u ($3/1000):($1==1?($5):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ippf, "u ($3/1000):($1==1?($4):NaN)  w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p', vpff, "u ($3/1000):($1==1?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", vpff, "u ($3/1000):($1==1?($5):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ippf_bpmsw, "u ($3/1000):($1==1?($4):NaN) w lp lt 1  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p', vpff, "u ($3/1000):($1==1?($4):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vpff, "u ($3/1000):($1==1?($5):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('ymax4 = GPVAL_DATA_Y_MAX + 0.1*abs(GPVAL_DATA_Y_MAX)', file=apj_gpl_file)
print('ymin4 = GPVAL_DATA_Y_MIN - 0.1*abs(GPVAL_DATA_Y_MIN)', file=apj_gpl_file)









#################################################################################
#################################################################################
#################################################################################
gen_title = 'Analysis for ' + name_orbitmax +'\\n' +' IR' + ip  + ' Beam' + beam 
print('reset', file=apj_gpl_file)
print('set terminal X11', file=apj_gpl_file)
print('set multiplot layout 2, 2 title "', gen_title, '" font ",15"', file=apj_gpl_file)

print('set xrange[',gp_start,':',gp_end,']', file=apj_gpl_file)
print('set yrange[ymin1:ymax1]', file=apj_gpl_file)
print('set title "APJ actionx" font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ actionx(nm)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p',hanff, "u ($3/1000):($1==0?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", haff, "u ($3/1000):($1==0?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", haff, "u ($3/1000):($1==0?($5*1e9):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ipaf, "u ($3/1000):($1==0?($4*1e9):NaN)  w p   pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p',hanff, "u ($3/1000):($1==0?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", haff, "u ($3/1000):($1==0?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", haff, "u ($3/1000):($1==0?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ipaf_bpmsw, "u ($3/1000):($1==0?($4*1e9):NaN) w p  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p',hanff, "u ($3/1000):($1==0?($4*1e9):NaN) w p  pt 7 ps 0.5 lc rgb 'green' title 'raw',", haff, "u ($3/1000):($1==0?($4*1e9):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", haff, "u ($3/1000):($1==0?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('\n', file=apj_gpl_file)
print('unset key', file=apj_gpl_file)
print('set yrange[ymin2:ymax2]', file=apj_gpl_file)
print('set title "APJ actiony" font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ actiony(nm)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p',vanff, "u ($3/1000):($1==1?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", vaff, "u ($3/1000):($1==1?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vaff, "u ($3/1000):($1==1?($5*1e9):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ipaf, "u ($3/1000):($1==1?($4*1e9):NaN)  w p   pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p',vanff, "u ($3/1000):($1==1?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", vaff, "u ($3/1000):($1==1?($4*1e9):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", vaff, "u ($3/1000):($1==1?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ipaf_bpmsw, "u ($3/1000):($1==1?($4*1e9):NaN) w p   pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p',vanff, "u ($3/1000):($1==1?($4*1e9):NaN) w p  pt 7 ps 0.5 lc rgb 'green' title 'raw',", vaff, "u ($3/1000):($1==1?($4*1e9):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vaff, "u ($3/1000):($1==1?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('\n', file=apj_gpl_file)

print('set yrange[ymin3:ymax3]', file=apj_gpl_file)
print('set title "APJ phasex " font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ phasex(rad)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p',hpnff, "u ($3/1000):($1==0?($4):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", hpff, "u ($3/1000):($1==0?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", hpff, "u ($3/1000):($1==0?($5):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ippf, "u ($3/1000):($1==0?($4):NaN)  w p  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p',hpnff, "u ($3/1000):($1==0?($4):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", hpff, "u ($3/1000):($1==0?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", hpff, "u ($3/1000):($1==0?($5):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ippf_bpmsw, "u ($3/1000):($1==0?($4):NaN) w p   pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p',hpnff, "u ($3/1000):($1==0?($4):NaN) w p  pt 7 ps 0.5 lc rgb 'green' title 'raw',", hpff, "u ($3/1000):($1==0?($4):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", hpff, "u ($3/1000):($1==0?($5):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)
print('\n', file=apj_gpl_file)

print('set yrange[ymin4:ymax4]', file=apj_gpl_file)
print('set title "APJ phasey " font ",14"', file=apj_gpl_file)
print('set xlabel \'s(km)\'', file=apj_gpl_file)
print('set ylabel \'APJ phasey(rad)\'', file=apj_gpl_file)
if (Is_ip_ap_kmod): print('p',vpnff, "u ($3/1000):($1==1?($4):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", vpff, "u ($3/1000):($1==1?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vpff, "u ($3/1000):($1==1?($5):NaN) w l lw 2 lt 1 lc rgb 'red'      title 'average',", ippf, "u ($3/1000):($1==1?($4):NaN)  w p  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
elif (Isbpmlr): print('p',vpnff, "u ($3/1000):($1==1?($4):NaN) w p pt 7 ps 0.5 lc rgb 'green' title 'raw',", vpff, "u ($3/1000):($1==1?($4):NaN) w p pt 7 ps 0.5 lc rgb 'red'  title 'filtered',", vpff, "u ($3/1000):($1==1?($5):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average',", ippf_bpmsw, "u ($3/1000):($1==1?($4):NaN) w p  pt 7 ps 1 lc rgb 'red' title 'inter-triplet'", file=apj_gpl_file)
else: print('p',vpnff, "u ($3/1000):($1==1?($4):NaN) w p  pt 7 ps 0.5 lc rgb 'green' title 'raw',", vpff, "u ($3/1000):($1==1?($4):NaN) w p  pt 7 ps 0.5 lc rgb 'red' title 'filtered',", vpff, "u ($3/1000):($1==1?($5):NaN) w l  lw 2 lt 1 lc rgb 'red'    title 'average'", file=apj_gpl_file)










if(Iserrlattice):
    gen_title = 'Analysis for ' + name_orbitmax +'\\n' + ' Beam' + beam 
    print('set terminal unknown', file=conventional_gpl_file)

    print('p',ha_cff, "u 3:($1==0?($4*1e9):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",ha_cff,"u 3:($1==0?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)
    print('ymax5 = GPVAL_DATA_Y_MAX*1.1', file=conventional_gpl_file)
    print('ymin5 = GPVAL_DATA_Y_MIN*0.9', file=conventional_gpl_file)

    print('p',va_cff, "u 3:($1==1?($4*1e9):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",va_cff,"u 3:($1==1?($5*1e9):NaN) w l lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)
    print('ymax6 = GPVAL_DATA_Y_MAX*1.1', file=conventional_gpl_file)
    print('ymin6 = GPVAL_DATA_Y_MIN*0.9', file=conventional_gpl_file)

    print('p',hp_cff, "u 3:($1==0?($4):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",hp_cff,"u 3:($1==0?($5):NaN) w l lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)
    print('ymax7 = GPVAL_DATA_Y_MAX + 0.1*abs(GPVAL_DATA_Y_MAX)', file=conventional_gpl_file)
    print('ymin7 = GPVAL_DATA_Y_MIN - 0.1*abs(GPVAL_DATA_Y_MIN)', file=conventional_gpl_file)

    print('p',vp_cff, "u 3:($1==1?($4):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",vp_cff,"u 3:($1==1?($5):NaN) w l lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)
    print('ymax8 = GPVAL_DATA_Y_MAX + 0.1*abs(GPVAL_DATA_Y_MAX)', file=conventional_gpl_file)
    print('ymin8 = GPVAL_DATA_Y_MIN - 0.1*abs(GPVAL_DATA_Y_MIN)', file=conventional_gpl_file)



###############################################################################
###############################################################################
    print('reset', file=conventional_gpl_file)
    print('set terminal X11', file=conventional_gpl_file)
    print('set multiplot layout 2, 2 title "', gen_title, '" font ",15"', file=conventional_gpl_file)

    print('set yrange[ymin5:ymax5]', file=conventional_gpl_file)
    print('set title "actionx" font ",14"', file=conventional_gpl_file)
    print('set xlabel \'s(km)\'', file=conventional_gpl_file)
    print('set ylabel \'actionx(nm)\'', file=conventional_gpl_file)
    print('p', han_cff , "u 3:($1==0?($4*1e9):NaN) w p pt 7 ps 0.6 lc rgb 'green' title 'raw',",ha_cff, "u 3:($1==0?($4*1e9):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",ha_cff,"u 3:($1==0?($5*1e9):NaN) w l  lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)
    print('unset key', file=conventional_gpl_file)

    print('set yrange[ymin6:ymax6]', file=conventional_gpl_file)    
    print('set title "actiony" font ",14"', file=conventional_gpl_file)
    print('set xlabel \'s(km)\'', file=conventional_gpl_file)
    print('set ylabel \'actiony(nm)\'', file=conventional_gpl_file)
    print('p', van_cff , "u 3:($1==1?($4*1e9):NaN) w p pt 7 ps 0.6 lc rgb 'green' title 'raw',",va_cff, "u 3:($1==1?($4*1e9):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",va_cff,"u 3:($1==1?($5*1e9):NaN) w l lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)

    print('set yrange[ymin7:ymax7]', file=conventional_gpl_file)    
    print('set title "phasex" font ",14"', file=conventional_gpl_file)
    print('set xlabel \'s(km)\'', file=conventional_gpl_file)
    print('set ylabel \'phasex(rad)\'', file=conventional_gpl_file)
    print('p', hpn_cff , "u 3:($1==0?($4):NaN) w p pt 7 ps 0.6 lc rgb 'green' title 'raw',",hp_cff, "u 3:($1==0?($4):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",hp_cff,"u 3:($1==0?($5):NaN) w l lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)


    print('set yrange[ymin8:ymax8]', file=conventional_gpl_file)    
    print('set title "phasey" font ",14"', file=conventional_gpl_file)
    print('set xlabel \'s(km)\'', file=conventional_gpl_file)
    print('set ylabel \'phasey(rad)\'', file=conventional_gpl_file)
    print('p', vpn_cff , "u 3:($1==1?($4):NaN) w p pt 7 ps 0.6 lc rgb 'green' title 'raw',",vp_cff, "u 3:($1==1?($4):NaN) w p pt 7 ps 0.6 lc rgb 'red'  title 'filtered',",vp_cff,"u 3:($1==1?($5):NaN) w l lw 2 lt 1 lc rgb 'red' title 'average'", file=conventional_gpl_file)






fcorrs_kmod.close
resxlf.close()
resxrf.close()
resylf.close()
resyrf.close()
resxf.close()
resyf.close()
resxlf_bpmsw.close()
resylf_bpmsw.close()
resxrf_bpmsw.close()
resyrf_bpmsw.close()
ir_4err.close()
ir_2err.close()
apj_gpl_file.close()
conventional_gpl_file.close()

gpl_apj_name = apj_dir + 'gAPJ.gpl'
gpl_convetional_name = apj_dir + 'gConventional.gpl'

if(args.showplots):
    call(['gnuplot', '-p', gpl_apj_name])
    if(Iserrlattice): call(['gnuplot', '-p',gpl_convetional_name])

if (not Is_ip_ap_kmod):
    file_to_remove =apj_dir + 'b1xl.dat' 
    call(['rm', '-f',file_to_remove])
    file_to_remove =apj_dir + 'b1yl.dat' 
    call(['rm', '-f',file_to_remove])
    file_to_remove =apj_dir + 'b1xr.dat' 
    call(['rm', '-f',file_to_remove])
    file_to_remove =apj_dir + 'b1yr.dat' 
    call(['rm', '-f',file_to_remove])
    call(['rm', '-f',ir_err_4q_file])

if (not Isbpmlr):
    file_to_remove =apj_dir + 'b1xl_bpmsw.dat' 
    call(['rm', '-f',file_to_remove])
    file_to_remove =apj_dir + 'b1yl_bpmsw.dat' 
    call(['rm', '-f',file_to_remove])
    file_to_remove =apj_dir + 'b1xr_bpmsw.dat' 
    call(['rm', '-f',file_to_remove])
    file_to_remove =apj_dir + 'b1yr_bpmsw.dat' 
    call(['rm', '-f',file_to_remove])
    call(['rm', '-f',ir_err_4q_file])
if(not Iserrlattice):
    file_to_remove =apj_dir + 'gConventional.gpl' 
    call(['rm', '-f',file_to_remove])
#print 'name_dir', orb_apj_dir

finput_line=open(apj_dir + 'input_line.dat','w')
print('ip', args.ip, file=finput_line)
print('beam', args.beam, file=finput_line)
print('model_dir', args.model_dir, file=finput_line)
print('bpm_for_avermax', args.bpm_for_avermax, file=finput_line)
print('ip_phase_bpm', args.ip_phase_bpm, file=finput_line)
print('left_kick_bpm', args.left_kick_bpm, file=finput_line)
print('right_kick_bpm', args.right_kick_bpm, file=finput_line)
print('left_arc_start', left_arc_start, file=finput_line)
print('right_arc_start', right_arc_start, file=finput_line)
print('left_arc_end',  left_arc_end, file=finput_line)
print('right_arc_end', right_arc_end, file=finput_line)
finput_line.close()


version_file = open(apj_dir + 'version_ActionPhaseJump.dat','a')
print("Version", version, "timestamp", datetime.datetime.now(), file=version_file)
version_file.close()



#######hereif de if false
if False: '''


print "OTHER TRAJECTORIES"


###########################Estimating quadrupole components from minmax, maxmax, maxmin trajectories###########################################################################
#for the four maxmax trajectories (nofilt 7 columns, others 11 columns)

#min/maxAPJaction_nofilt.sdds      Action obtained with nominal lattice and no filter 
#min/maxAPJphase_nofilt.sdds       Phase obtained with nominal lattice and no filter 
#min/maxAction_nofilt.sdds         Action obtained with real lattice  and no filter 
#min/maxPhase_nofilt.sdds          Phase obtained with real lattice and no filter 
#min/maxAPJaction.sdds             Action obtained with nominal lattice and filter 
#min/maxAPJphase.sdds              Phase obtained with nominal lattice and  filter 
#min/maxAction.sdds                Action obtained with real lattice  and filter 
#min/maxPhase.sdds                 Phase obtained with real lattice and  filter


#Files with prefix APJ is created with the nominal lattice,  others files are created with the real lattice (nominal + grad errors)

##############Read all four kind of average trajectories minmax, maxmax, maxmin and minmin#####
orbitmaxmax = open(name_orbitmaxmax,'r')
diforbx, sort_linesx = traj_data(orbitmaxmax, nameelx,'-x')
print "horizontal trajectories were read from avermaxmax.sdds.new"
orbitmaxmax = open(name_orbitmaxmax,'r')
diforby, sort_linesy = traj_data(orbitmaxmax, nameely,'-y')
print "horizontal trajectories were read from avermaxmax.sdds.new"





#Actions, phases and kicks from horizontal plane of minmax trajectory###############################
sdds_col = 7
plane = '-x'    
APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, x_BPM_l, x_BPM_r =Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpml,bpmr,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe,nameelxe,psixe,betxe,wx,bwx,s_acdipole,plane,beam,bpm_phase,ip,'bpmsw')

dx1l=kick_strength_v03(averactant,act_trip,averphaseant,phase_trip,betx[nameelx.index(bpml)],psix[nameelx.index(bpml)])
dx1r=kick_strength_v03(act_trip,averactdesp,phase_trip,averphasedesp,betx[nameelx.index(bpmr)],psix[nameelx.index(bpmr)])
if (beam == '2'):
    dx1l = -dx1l
    dx1r = -dx1r
#Position estimation
x1l=posestim(betax_error_l,betax_BPM_l,psix_error_l,psix_BPM_l,averphaseant,x_BPM_l)
x1r=posestim(betax_error_r,betax_BPM_r,psix_error_r,psix_BPM_r,averphasedesp,x_BPM_r)

#Actions, phases and kicks from  vertical plane of minmax trajectory###############################
sdds_col = 7
plane = '-y'    
APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, y_BPM_l, y_BPM_r =Arc_Trip_APs(sdds_col, sort_linesy,diforby,bpml,bpmr,sely,nameely,psiy,bety,sL_begin,sR_end,nsigma,selye,nameelye,psiye,betye,wy,bwy,s_acdipole,plane,beam,bpm_phase,ip,'bpmsw')
    
dy1l=kick_strength_v03(averactant,act_trip,averphaseant,phase_trip,bety[nameely.index(bpml)],psiy[nameely.index(bpml)])
dy1r=kick_strength_v03(act_trip,averactdesp,phase_trip,averphasedesp,bety[nameely.index(bpmr)],psiy[nameely.index(bpmr)])
if (beam == '2'):
    dy1l = -dy1l
    dy1r = -dy1r
#Position estimation
y1l=posestim(betay_error_l,betay_BPM_l,psiy_error_l,psiy_BPM_l,averphaseant,y_BPM_l)
y1r=posestim(betay_error_r,betay_BPM_r,psiy_error_r,psiy_BPM_r,averphasedesp,y_BPM_r)

APJact_nfl = APJact_nflx + APJact_nfly
APJphase_nfl = APJphase_nflx + APJphase_nfly
act_nfl = act_nflx + act_nfly
phase_nfl = phase_nflx + phase_nfly
APJactl = APJactlx + APJactly
APJphasel = APJphaselx + APJphasely
actl = actlx + actly
phasel = phaselx + phasely


write_file(orb_apj_dir + 'minmaxAPJaction_nofilt.sdds',APJact_nfl,'w')
write_file(orb_apj_dir + 'minmaxAPJphase_nofilt.sdds',APJphase_nfl,'w')
write_file(orb_apj_dir + 'minmaxAction_nofilt.sdds', act_nfl,'w')
write_file(orb_apj_dir + 'minmaxPhase_nofilt.sdds', phase_nfl,'w')

write_file(orb_apj_dir + 'minmaxAPJaction.sdds',APJactl,'w')
write_file(orb_apj_dir + 'minmaxAPJphase.sdds',APJphasel,'w')
write_file(orb_apj_dir + 'minmaxAction.sdds', actl,'w')
write_file(orb_apj_dir + 'minmaxPhase.sdds', phasel,'w')









#Actions, phases and kicks from horizontal plane of maxmax trajectory###############################
sdds_col = 4
plane = '-x'    
APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, x_BPM_l, x_BPM_r =Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpml,bpmr,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe,nameelxe,psixe,betxe,wx,bwx,s_acdipole,plane,beam,bpm_phase,ip,'bpmsw')

dx2l=kick_strength_v03(averactant,act_trip,averphaseant,phase_trip,betx[nameelx.index(bpml)],psix[nameelx.index(bpml)])
dx2r=kick_strength_v03(act_trip,averactdesp,phase_trip,averphasedesp,betx[nameelx.index(bpmr)],psix[nameelx.index(bpmr)])
if (beam == '2'):
    dx2l = -dx2l
    dx2r = -dx2r
#Position estimation
x2l=posestim(betax_error_l,betax_BPM_l,psix_error_l,psix_BPM_l,averphaseant,x_BPM_l)
x2r=posestim(betax_error_r,betax_BPM_r,psix_error_r,psix_BPM_r,averphasedesp,x_BPM_r)

#Actions, phases and kicks from  vertical plane of maxmax trajectory###############################
sdds_col = 4
plane = '-y'    
APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, y_BPM_l, y_BPM_r =Arc_Trip_APs(sdds_col, sort_linesy,diforby,bpml,bpmr,sely,nameely,psiy,bety,sL_begin,sR_end,nsigma,selye,nameelye,psiye,betye,wy,bwy,s_acdipole,plane,beam,bpm_phase,ip,'bpmsw')
    
dy2l=kick_strength_v03(averactant,act_trip,averphaseant,phase_trip,bety[nameely.index(bpml)],psiy[nameely.index(bpml)])
dy2r=kick_strength_v03(act_trip,averactdesp,phase_trip,averphasedesp,bety[nameely.index(bpmr)],psiy[nameely.index(bpmr)])
if (beam == '2'):
    dy2l = -dy2l
    dy2r = -dy2r
#Position estimation
y2l=posestim(betay_error_l,betay_BPM_l,psiy_error_l,psiy_BPM_l,averphaseant,y_BPM_l)
y2r=posestim(betay_error_r,betay_BPM_r,psiy_error_r,psiy_BPM_r,averphasedesp,y_BPM_r)



B1yeql = (x1l*dy2l-x2l*dy1l)/(x1l*y2l-x2l*y1l)
A1yeql = (dy1l*y2l-dy2l*y1l)/(x1l*y2l-x2l*y1l)

B1yeqr = (x1r*dy2r-x2r*dy1r)/(x1r*y2r-x2r*y1r)
A1yeqr = (dy1r*y2r-dy2r*y1r)/(x1r*y2r-x2r*y1r)

B1yeql_bpmsw = B1yeql 
B1yeqr_bpmsw = B1yeqr
A1yeql_bpmsw = A1yeql
A1yeqr_bpmsw = A1yeqr


APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,
APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely,

APJact_nfl = APJact_nflx + APJact_nfly
APJphase_nfl = APJphase_nflx + APJphase_nfly
act_nfl = act_nflx + act_nfly
phase_nfl = phase_nflx + phase_nfly
APJactl = APJactlx + APJactly
APJphasel = APJphaselx + APJphasely
actl = actlx + actly
phasel = phaselx + phasely


write_file(orb_apj_dir + 'maxmaxAPJaction_nofilt.sdds',APJact_nfl,'w')
write_file(orb_apj_dir + 'maxmaxAPJphase_nofilt.sdds',APJphase_nfl,'w')
write_file(orb_apj_dir + 'maxmaxAction_nofilt.sdds', act_nfl,'w')
write_file(orb_apj_dir + 'maxmaxPhase_nofilt.sdds', phase_nfl,'w')

write_file(orb_apj_dir + 'maxmaxAPJaction.sdds',APJactl,'w')
write_file(orb_apj_dir + 'maxmaxAPJphase.sdds',APJphasel,'w')
write_file(orb_apj_dir + 'maxmaxAction.sdds', actl,'w')
write_file(orb_apj_dir + 'maxmaxPhase.sdds', phasel,'w')










#Actions, phases and kicks from horizontal plane of maxmin trajectory###############################
sdds_col = 6
plane = '-x'    
APJact_nflx,APJphase_nflx,act_nflx,phase_nflx,APJactlx,APJphaselx,actlx,phaselx,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, x_BPM_l, x_BPM_r =Arc_Trip_APs(sdds_col, sort_linesx,diforbx,bpml,bpmr,selx,nameelx,psix,betx,sL_begin,sR_end,nsigma,selxe,nameelxe,psixe,betxe,wx,bwx,s_acdipole,plane,beam,bpm_phase,ip,'bpmsw')

dx1l=kick_strength_v03(averactant,act_trip,averphaseant,phase_trip,betx[nameelx.index(bpml)],psix[nameelx.index(bpml)])
dx1r=kick_strength_v03(act_trip,averactdesp,phase_trip,averphasedesp,betx[nameelx.index(bpmr)],psix[nameelx.index(bpmr)])
if (beam == '2'):
    dx1l = -dx1l
    dx1r = -dx1r
#Position estimation
x1l=posestim(betax_error_l,betax_BPM_l,psix_error_l,psix_BPM_l,averphaseant,x_BPM_l)
x1r=posestim(betax_error_r,betax_BPM_r,psix_error_r,psix_BPM_r,averphasedesp,x_BPM_r)

#Actions, phases and kicks from  vertical plane of maxmin trajectory###############################
sdds_col = 6
plane = '-y'    
APJact_nfly,APJphase_nfly,act_nfly,phase_nfly,APJactly,APJphasely,actly,phasely,averactant, averphaseant, act_trip, phase_trip, averactdesp, averphasedesp, y_BPM_l, y_BPM_r =Arc_Trip_APs(sdds_col, sort_linesy,diforby,bpml,bpmr,sely,nameely,psiy,bety,sL_begin,sR_end,nsigma,selye,nameelye,psiye,betye,wy,bwy,s_acdipole,plane,beam,bpm_phase,ip,'bpmsw')
    
dy1l=kick_strength_v03(averactant,act_trip,averphaseant,phase_trip,bety[nameely.index(bpml)],psiy[nameely.index(bpml)])
dy1r=kick_strength_v03(act_trip,averactdesp,phase_trip,averphasedesp,bety[nameely.index(bpmr)],psiy[nameely.index(bpmr)])
if (beam == '2'):
    dy1l = -dy1l
    dy1r = -dy1r
#Position estimation
y1l=posestim(betay_error_l,betay_BPM_l,psiy_error_l,psiy_BPM_l,averphaseant,y_BPM_l)
y1r=posestim(betay_error_r,betay_BPM_r,psiy_error_r,psiy_BPM_r,averphasedesp,y_BPM_r)


B1xeql = (y1l*dx2l-y2l*dx1l)/(x1l*y2l-x2l*y1l)
A1xeql = (x1l*dx2l-x2l*dx1l)/(x1l*y2l-x2l*y1l)

B1xeqr = (y1r*dx2r-y2r*dx1r)/(x1r*y2r-x2r*y1r)
A1xeqr = (x1r*dx2r-x2r*dx1r)/(x1r*y2r-x2r*y1r)

B1xeql_bpmsw = B1xeql 
B1xeqr_bpmsw = B1xeqr

A1xeql_bpmsw = A1xeql
A1xeqr_bpmsw = A1xeqr

print >>resxlf4,B1xeql,Betxeql, IntBetxQal,  IntBetxQbl, IntBetxQ4L,  IntBetxQ5L, IntBetxQ6L,PhixQal,  PhixQbl, PhixQ4L,  PhixQ5L, PhixQ6L, 0, IntBetxQ1L
print >>resxrf4,B1xeqr,Betxeqr, IntBetxQar,  IntBetxQbr, IntBetxQ4R,  IntBetxQ5R, IntBetxQ6R,PhixQar,  PhixQbr, PhixQ4R,  PhixQ5R, PhixQ6R, 0, IntBetxQ1R

print >>resylf4,B1yeql,Betyeql, IntBetyQal,  IntBetyQbl, IntBetyQ4L,  IntBetyQ5L, IntBetyQ6L,PhiyQal,  PhiyQbl, PhiyQ4L,  PhiyQ5L, PhiyQ6L, 0, IntBetyQ1L 
print >>resyrf4,B1yeqr,Betyeqr, IntBetyQar,  IntBetyQbr, IntBetyQ4R,  IntBetyQ5R, IntBetyQ6R,PhiyQar,  PhiyQbr, PhiyQ4R,  PhiyQ5R, PhiyQ6R, 0, IntBetyQ1R 


APJact_nfl = APJact_nflx + APJact_nfly
APJphase_nfl = APJphase_nflx + APJphase_nfly
act_nfl = act_nflx + act_nfly
phase_nfl = phase_nflx + phase_nfly
APJactl = APJactlx + APJactly
APJphasel = APJphaselx + APJphasely
actl = actlx + actly
phasel = phaselx + phasely


write_file(orb_apj_dir + 'maxminAPJaction_nofilt.sdds',APJact_nfl,'w')
write_file(orb_apj_dir + 'maxminAPJphase_nofilt.sdds',APJphase_nfl,'w')
write_file(orb_apj_dir + 'maxminAction_nofilt.sdds', act_nfl,'w')
write_file(orb_apj_dir + 'maxminPhase_nofilt.sdds', phase_nfl,'w')

write_file(orb_apj_dir + 'maxminAPJaction.sdds',APJactl,'w')
write_file(orb_apj_dir + 'maxminAPJphase.sdds',APJphasel,'w')
write_file(orb_apj_dir + 'maxminAction.sdds', actl,'w')
write_file(orb_apj_dir + 'maxminPhase.sdds', phasel,'w')

#qslint='MQSX.3L'+str(ip)
#corrs_file = 'IR' + str(ip) + '_corrs.madx'
#fcorrs=open(corrs_file,'w')

#print IntBetxQal*IntBetyQbl
#- IntBetxQbl*IntBetyQal
corrq2l = (B1yeql_kmod*Betyeql*IntBetxQbl - B1xeql_kmod*Betxeql*IntBetyQbl)/(IntBetxQal*IntBetyQbl - IntBetxQbl*IntBetyQal)
corrq3l = (B1xeql_kmod*Betxeql*IntBetyQal - B1yeql_kmod*Betyeql*IntBetxQal)/(IntBetxQal*IntBetyQbl - IntBetxQbl*IntBetyQal)

corrq2r = (B1yeqr_kmod*Betyeqr*IntBetxQbr - B1xeqr_kmod*Betxeqr*IntBetyQbr)/(IntBetxQar*IntBetyQbr - IntBetxQbr*IntBetyQar)
corrq3r = (B1xeqr_kmod*Betxeqr*IntBetyQar - B1yeqr_kmod*Betyeqr*IntBetxQar)/(IntBetxQar*IntBetyQbr - IntBetxQbr*IntBetyQar)


print >> fcorrs, '{:^8.6}'.format(corrq2l), '{:^8.6}'.format(corrq3l), '{:^8.6}'.format(corrq2r), '{:^8.6}'.format(corrq3r)

print "B1xl_kmod ", B1xeql_kmod
print "B1yl_kmod ", B1yeql_kmod
print "B1xr_kmod ", B1xeqr_kmod
print "B1yr_kmod ", B1yeqr_kmod

print "B1xl_bpmsw ", B1xeql_bpmsw
print "B1yl_bpmsw ", B1yeql_bpmsw
print "B1xr_bpmsw ", B1xeqr_bpmsw
print "B1yr_bpmsw ", B1yeqr_bpmsw


print "A1xl ", A1xeql_bpmsw
print "A1yl ", A1yeql_bpmsw
print "A1xr ", A1xeqr_bpmsw
print "A1yr ", A1yeqr_bpmsw

print corrq2l_name, corrq2l
print corrq3l_name , corrq3l
print corrq2r_name , corrq2r
print corrq3r_name , corrq3r








#################Calibration of BPMSWs##########################
print >> calf,sum(calcx_l)/len(calcx_l)-1.0, sum(calcx_l2)/len(calcx_l2)-1.0, sum(calcy_l)/len(calcy_l)-1.0, sum(calcy_l2)/len(calcy_l2)-1.0, sum(calcx_r)/len(calcx_r)-1.0, sum(calcx_r2)/len(calcx_r2)-1.0,sum(calcy_r)/len(calcy_r)-1.0, sum(calcy_r2)/len(calcy_r2)-1.0

############Calibration of ARC BPMS###########################
for line in sort_linesx:
    allcalx_arc=[]
    for apjx in jcdcx:
        bpm_arc = '"'+line[1]+'"'
        bpm_arc2 = line[1]
        posx=float(diforbx[sort_linesx.index(line)][apjx[0]])
        x_arc = posx/1000
        turnx2 = apjx[0]+1
        if (turnx2 <= len(diforbx[0])/2 and abs(sin(psixe[nameelx.index(bpm_arc)]-apjx[2])) > 0.9):
            x_arc_jc = sqrt(2*apjx[1]*betxe[nameelx.index(bpm_arc)])*sin(psixe[nameelx.index(bpm_arc)]-apjx[2])
            calx_arc =  x_arc/x_arc_jc
            allcalx_arc.append(calx_arc)
        t_list=map(list, zip(*setcal_list))
    meas_calx = sum(allcalx_arc)/len(allcalx_arc)
    nom_calx =float(t_list[1][t_list[0].index(bpm_arc2)])
    diff_calx=meas_calx-nom_calx
    print >> calxf, bpm_arc,meas_calx,nom_calx, diff_calx


for line in sort_linesy:
    allcaly_arc=[]
    for apjy in jcdcy:
        bpm_arc = '"'+line[1]+'"'
        bpm_arc2 = line[1]
        posy=float(diforby[sort_linesy.index(line)][apjy[0]])
        y_arc = posy/1000
        turny2 = apjy[0]+1
        if (turny2 > len(diforby[0])/2 and abs(sin(psiye[nameely.index(bpm_arc)]-apjy[2])) > 0.9):
            y_arc_jc = sqrt(2*apjy[1]*betye[nameely.index(bpm_arc)])*sin(psiye[nameely.index(bpm_arc)]-apjy[2])
            caly_arc =  y_arc/y_arc_jc
            allcaly_arc.append(caly_arc)
            #print  y_arc_jc, y_arc, caly_arc, len(allcaly_arc)
        t_list=map(list, zip(*setcal_list))
    meas_caly = sum(allcaly_arc)/len(allcaly_arc)
    nom_caly =float(t_list[2][t_list[0].index(bpm_arc2)])
    diff_caly=meas_caly-nom_caly
    print >> calyf, bpm_arc,meas_caly,nom_caly, diff_caly
 

####################Calibration of IR BPMs except for  the BPMSWs#######################3
t_list2=map(list, zip(*setcal_list2))

calx_IRBPMS=[]
IRBPMS=[]
for line in IR_datax2:
    bpmll=line[0]
    line.pop(0)
    line.pop(0)
    nom_calx =float(t_list2[1][t_list2[0].index(bpmll)])
    IRBPMS.append(bpmll)
    calx_IRBPMS.append(nom_calx-sum(line)/len(line))
for ii in calx_IRBPMS:
    print >> calx_IRf, ii,
print >> calx_IRf, ""

#print IRBPMS
caly_IRBPMS=[]
IRBPMS=[]
for line in IR_datay2:
    bpmll=line[0]
    line.pop(0)
    line.pop(0)
    nom_caly =float(t_list2[2][t_list2[0].index(bpmll)])
    IRBPMS.append(bpmll)
    caly_IRBPMS.append(nom_caly-sum(line)/len(line))
for ii in caly_IRBPMS:
    print >> caly_IRf, ii,
print >> caly_IRf, ""





    #bpmyy=IR_datay2[ii][0]
    #IR_datay2[ii].pop(0)
    #IR_datay2[ii].pop(0)
    #nom_caly =float(t_list2[2][t_list2[0].index(bpmll)])
    #print bpmll,nom_calx,sum(line)/len(line), nom_calx-sum(line)/len(line), bpmyy, nom_caly, sum(IR_datay2[ii])/len(IR_datay2[ii]), nom_caly-sum(IR_datay2[ii])/len(IR_datay2[ii])
    #ii = ii+1




#print >> calf, sum(calcx_l)/len(calcx_l) ,sum(calcx_r)/len(calcx_r),sum(relx)/len(relx),  sum(calcy_l)/len(calcy_l) ,sum(calcy_r)/len(calcy_r),sum(rely)/len(rely)

#sortsinx = sorted(allsinx, key=lambda pos: pos[1], reverse = True)

#print "x cal factors ordered by sin"
#for line in sortsinx:
#    print line

#print "y cal factors ordered by sin"
#sortsiny = sorted(allsiny, key=lambda pos: pos[1], reverse = True)

#for line in sortsiny:
#    print line





      







########################Y plane##################




###Estimating quadrupole components of the equivalent kicks from kicks and BPM data in b1*4.dat files with eq 22 of PRAB 2017####    
resxl.close()
resyl.close()
resxr.close()
resyr.close()

resxl=open('b1xl4.dat','r')
resyl=open('b1yl4.dat','r')
resxr=open('b1xr4.dat','r')
resyr=open('b1yr4.dat','r')

dxlvsxl = []
dylvsyl = []
dxrvsxr = []
dyrvsyr = []
for line in resxl:
 cutline = line.split(None)
 dxl = float(cutline[13])
 xl =   float(cutline[16])
 dxlvsxl.append([dxl,xl])

for line in resyl:
 cutline = line.split(None)
 dyl = -float(cutline[13])
 yl =   float(cutline[16])
 dylvsyl.append([dyl,yl])

x1l = dxlvsxl[0][1]
dx1l = dxlvsxl[0][0]
y1l = dylvsyl[0][1]
dy1l = dylvsyl[0][0]

x2l = dxlvsxl[1][1]
dx2l = dxlvsxl[1][0]
y2l = dylvsyl[1][1]
dy2l = dylvsyl[1][0]

B1yeql = (x1l*dy2l-x2l*dy1l)/(x1l*y2l-x2l*y1l)
A1yeql = (dy1l*y2l-dy2l*y1l)/(x1l*y2l-x2l*y1l)


x1l = dxlvsxl[2][1]
dx1l = dxlvsxl[2][0]
y1l = dylvsyl[2][1]
dy1l = dylvsyl[2][0]

x2l = dxlvsxl[1][1]
dx2l = dxlvsxl[1][0]
y2l = dylvsyl[1][1]
dy2l = dylvsyl[1][0]

B1xeql = (y1l*dx2l-y2l*dx1l)/(x1l*y2l-x2l*y1l)
A1xeql = (x1l*dx2l-x2l*dx1l)/(x1l*y2l-x2l*y1l)





for line in resxr:
 cutline = line.split(None)
 dxr = float(cutline[13])
 xr =   float(cutline[16])
 dxrvsxr.append([dxr,xr])
 
for line in resyr:
 cutline = line.split(None)
 dyr = -float(cutline[13])
 yr =   float(cutline[16])
 dyrvsyr.append([dyr,yr])


x1r = dxrvsxr[0][1]
dx1r = dxrvsxr[0][0]
y1r = dyrvsyr[0][1]
dy1r = dyrvsyr[0][0]

x2r = dxrvsxr[1][1]
dx2r = dxrvsxr[1][0]
y2r = dyrvsyr[1][1]
dy2r = dyrvsyr[1][0]

B1yeqr = (x1r*dy2r-x2r*dy1r)/(x1r*y2r-x2r*y1r)
A1yeqr = (dy1r*y2r-dy2r*y1r)/(x1r*y2r-x2r*y1r)


x1r = dxrvsxr[2][1]
dx1r = dxrvsxr[2][0]
y1r = dyrvsyr[2][1]
dy1r = dyrvsyr[2][0]

x2r = dxrvsxr[1][1]
dx2r = dxrvsxr[1][0]
y2r = dyrvsyr[1][1]
dy2r = dyrvsyr[1][0]

B1xeqr = (y1r*dx2r-y2r*dx1r)/(x1r*y2r-x2r*y1r)
A1xeqr = (x1r*dx2r-x2r*dx1r)/(x1r*y2r-x2r*y1r)



 


###########Quadrupole components of the equivalent kick fitting the 4 average max trajectories############
p0 =  1, 1
#B1xeq , A1xeq =  curve_fit(funcx2,(allxs0, allys0),allkickx,p0)[0]
#B1xeql, A1xeql =  curve_fit(funcx2,(allxleq,allyleq),allkickxl,p0)[0]
#B1xeqr, A1xeqr =  curve_fit(funcx2,(allxreq,allyreq),allkickxr,p0)[0]
#print 'B1xeq and A1 from curve_fit', curve_fit(funcx2,(allxs0, allys0),allkickx,p0)[0]   
#A1xeq = curve_fit(funcx2,(allxs0, allys0),allkickx,p0)[0][1]


p0 =  1, 1
#B1yeq, A1yeq =  curve_fit(funcy2,(allxs0,allys0),allkicky,p0)[0]
#B1yeql, A1yeql =  curve_fit(funcy2,(allxleq,allyleq),allkickyl,p0)[0]
#B1yeqr, A1yeqr =  curve_fit(funcy2,(allxreq,allyreq),allkickyr,p0)[0]
#B1xeq , A1xeq , B1yeq, A1yeq = 0, 0, 0, 0



###################Change sign if beam 2#################33
if beam == '2' :
    B1xeql = -B1xeql
    B1xeqr = -B1xeqr
    B1yeql = -B1yeql
    B1yeqr = -B1yeqr


    #if beam == '2' :
    #A1xeql = -A1xeql
    #A1xeqr = -A1xeqr
    #A1yeql = -A1yeql
    #A1yeqr = -A1yeqr

    ###############quadrupole components are finally printed in b1*.dat files##############################3
print >>resxlf,B1xeql,Betxeql, IntBetxQal,  IntBetxQbl, IntBetxQ4L,  IntBetxQ5L, IntBetxQ6L,PhixQal,  PhixQbl, PhixQ4L,  PhixQ5L, PhixQ6L, 0, IntBetxQ1L 
print >>resxrf,B1xeqr,Betxeqr, IntBetxQar,  IntBetxQbr, IntBetxQ4R,  IntBetxQ5R, IntBetxQ6R,PhixQar,  PhixQbr, PhixQ4R,  PhixQ5R, PhixQ6R, 0, IntBetxQ1R
print >>resylf,B1yeql,Betyeql, IntBetyQal,  IntBetyQbl, IntBetyQ4L,  IntBetyQ5L, IntBetyQ6L,PhiyQal,  PhiyQbl, PhiyQ4L,  PhiyQ5L, PhiyQ6L, 0, IntBetyQ1L 
print >>resyrf,B1yeqr,Betyeqr, IntBetyQar,  IntBetyQbr, IntBetyQ4R,  IntBetyQ5R, IntBetyQ6R,PhiyQar,  PhiyQbr, PhiyQ4R,  PhiyQ5R, PhiyQ6R, 0, IntBetyQ1R 



#print 'B1yeq and A1 from curve_fit', curve_fit(funcy2,(allxs0,allys0),allkicky,p0)[0]   
#A1yeq = curve_fit(funcy2,(allxs0, allys0),allkicky,p0)[0][1]
#A1p = ((A1xeq + A1yeq)*sqrt(Betxeq)*sqrt(Betyeq))/(2*Intbetxbety)

#print 'k_corr', -A1p

#xarr = np.array(allxs0)
#kickxarr=np.array(allkickx)
#MATx = np.vstack([-xarr, np.ones(len(xarr))]).T
#B1xeq, c = np.linalg.lstsq(MATx, kickxarr)[0]
#print 'B1xeq2',  B1xeq


#yarr = np.array(allys0)
#kickyarr=np.array(allkicky)
#MATy = np.vstack([yarr, np.ones(len(yarr))]).T
#B1yeq, c = np.linalg.lstsq(MATy, kickyarr)[0]
#print  'B1yeq2', B1yeq
##################Corrections Whole IR####################
###################Correction in Qa Qb#################################
#mx= IntBetxQ2L/IntBetxQ2R
#my= IntBetyQ2L/IntBetyQ2R
#Cx= B1xeq*Betxeq/IntBetxQ2R
#Cy= B1yeq*Betyeq/IntBetyQ2R
#B1pQ2L= (Cy - Cx)/(my - mx)
#B1pQ2R= (Cx*my -Cy*mx)/(my-mx)
#A1p = ((A1xeq + A1yeq)*sqrt(Betxeq)*sqrt(Betyeq))/(2*Intbetxbety)
#print"Equivalent errors at Qa, Qb, MQSX"
#print B1pQ2L, B1pQ2R, A1p
#print "Corrections suggested at Qa, Qb, MQSX"
#print -B1pQ2L, -B1pQ2R, -A1p

##################Corrections left triplet####################
###################Correction in Qal Qbl#################################
mxl= IntBetxQal/IntBetxQbl
myl= IntBetyQal/IntBetyQbl
Cxl= B1xeql*Betxeql/IntBetxQbl
Cyl= B1yeql*Betyeql/IntBetyQbl
B1pQal= (Cyl - Cxl)/(myl - mxl)
B1pQbl= (Cxl*myl -Cyl*mxl)/(myl-mxl)
#A1pl = ((A1xeql + A1yeql)*sqrt(Betxeql)*sqrt(Betyeql))/(2*Intbetxbety_l)
#print"Equivalent errors in left triplet  at Qa, Qb"
#print B1pQal, B1pQbl
#print "Corrections suggested in left triplet at Qa, Qb"
#print -B1pQal, -B1pQbl


##################Corrections right triplet####################
###################Correction in Qar Qbr#################################
mxr= IntBetxQar/IntBetxQbr
myr= IntBetyQar/IntBetyQbr
Cxr= B1xeqr*Betxeqr/IntBetxQbr
Cyr= B1yeqr*Betyeqr/IntBetyQbr
B1pQar= (Cyr - Cxr)/(myr - mxr)
B1pQbr= (Cxr*myr -Cyr*mxr)/(myr-mxr)
#A1pr = ((A1xeqr + A1yeqr)*sqrt(Betxeqr)*sqrt(Betyeqr))/(2*Intbetxbety_r)
#print"Equivalent errors in right triplet  at Qa, Qb"
#print B1pQar, B1pQbr
#print "Corrections suggested in right  triplet at Qa, Qb"
#print -B1pQar, -B1pQbr
#print  B1pQal, B1pQbl, 0, B1pQar, B1pQbr, 0
'''
