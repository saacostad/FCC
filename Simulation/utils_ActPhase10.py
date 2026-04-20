from math import sqrt,cos,sin,atan2,fabs, asin, modf
from math import pi
# from string import join
import numpy as np
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Inversion
#Subroutine to Find the action and the source angle from a set of 
#two BPM measuraments.
#Cardona, Peggs. Linear and nonlinear magnetic error measurements using action and phase
#jump analysis. Phys. Rev. ST Accel. Beams 12, 014002 (2009) [11 pages]. Eq. 13 and 14.
#Need to see, Erratum of this article
def inversion(z1, z2, psi1, psi2):
    J=(z2*z2+z1*z1-2.0*z1*z2*cos(psi2-psi1))/(sin(psi2-psi1)*sin(psi2-psi1))*0.5
    #a=sqrt(2.0*J)
    sindelta=(z1*sin(psi2)-z2*sin(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    cosdelta=(z1*cos(psi2)-z2*cos(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    #sindelta=1.0/a*(z1*sin(psi2)-z2*sin(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    #cosdelta=1.0/a*(z1*cos(psi2)-z2*cos(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    #sindelta=(z1*sin(psi2)-z2*sin(psi1))
    #cosdelta=(z1*cos(psi2)-z2*cos(psi1))
    delta=atan2(sindelta,cosdelta)
    #print J
    return J,delta
#-----------------------------------------------------------------------
def inversion2(z1, z2, psi1, psi2):
    J=(z2*z2+z1*z1-2.0*z1*z2*cos(psi2-psi1))/(sin(psi2-psi1)*sin(psi2-psi1))*0.5
    #a=sqrt(2.0*J)
    sindelta=(z1*sin(psi2)-z2*sin(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    cosdelta=(z1*cos(psi2)-z2*cos(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    #sindelta=1.0/a*(z1*sin(psi2)-z2*sin(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    #cosdelta=1.0/a*(z1*cos(psi2)-z2*cos(psi1))/(sin(psi1)*cos(psi2)-cos(psi1)*sin(psi2))
    #sindelta=(z1*sin(psi2)-z2*sin(psi1))
    #cosdelta=(z1*cos(psi2)-z2*cos(psi1))
    delta=np.arctan2(sindelta,cosdelta)
    #print J
    return J,delta



#-----------------------------------------------------------------------
#Calcular accion y fase de una lista
def doaccionyfase(zred,psiz):
#El primer elemento no se calcula, debido a que el calculo es entre dos BPMs
#y el final de la orbita no se une con el principio
	action=[0]
	phase=[0]
    #for i in range(len(s)-1):
	for i in range(len(zred)-1):
		if (zred[i]==0) and (zred[i+1]==0):#Valores para ser descartados luego
			j=0.0
			delta=100.0
		else:#Calculo real
			j,delta=inversion(zred[i],zred[i+1],psiz[i],psiz[i+1])

		action.append(j)
		phase.append(delta)
	#El primer elemento igual al segundo
	action[0]=action[1]
	phase[0]=phase[1]
	return action,phase
#-----------------------------------------------------------------------
# CALCULAR KICK
#Calcula la magnitud y el signo del kick
#Cardona, Peggs. Linear and nonlinear magnetic error measurements using action and phase
#jump analysis. Phys. Rev. ST Accel. Beams 12, 014002 (2009) [11 pages]. Eq. 15
#Need to see, Erratum of this article
def kick_strength_v03(J0, J1, delta0, delta1, bet_z_theta, psi_z_theta):
    # Calcular magnitud
    nume = (2.0*J0+2.0*J1-4.0*sqrt(J0*J1)*cos(delta1-delta0))
    if nume < 0:
        print('WARNING!!, ROOT ARGUMENT IN KICK IS NEGATIVE!!!, ASSIGNING ZERO')
        nume=0 
    theta_z=sqrt(nume/bet_z_theta)
    # Calcular signo
    signo=sqrt(2.0*J0/bet_z_theta)*sin(delta1-delta0)/sin(psi_z_theta-delta1)
    if(signo<0.0):
	    theta_z=-theta_z
    # Magnitud y signo calculadas
    #print "testjyp", theta_z
    return theta_z
    #return signo
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# CALCULAR KICK
#Calcula la magnitud y el signo del kick
#Cardona, Peggs. Linear and nonlinear magnetic error measurements using action and phase
#jump analysis. Phys. Rev. ST Accel. Beams 12, 014002 (2009) [11 pages]. Eq. 15
#Need to see, Erratum of this article
def kick_strength_v02(J0, J1, delta0, delta1, bet_z_theta, psi_z_theta):
    # Calcular magnitud
    theta_z=sqrt((2.0*J0+2.0*J1-4.0*sqrt(J0*J1)*cos(delta1-delta0))/bet_z_theta)
    # Calcular signo
    signo=sqrt(2.0*J0/bet_z_theta)*sin(delta1-delta0)/sin(psi_z_theta-delta1)
    if(signo<0.0):
	    theta_z=-theta_z
    # Magnitud y signo calculadas
    return theta_z
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Position estimation
#Cardona. Local magnetic error estimation using action and phase jump analysis of orbit data.
#Proceedings of PAC07, Alburquerque, New Mexico, USA
def posestim(bet_z_s0, bet_z_sBPM, psi_s0, psi_sBPM, phi, z_sBPM):
    #zs0=sqrt(bet_z_s0/bet_z_sBPM)*z_sBPM#*sin(psi_s0-phi)/sin(psi_sBPM-phi)
    zs0=sqrt(bet_z_s0/bet_z_sBPM)*z_sBPM*sin(psi_s0-phi)/sin(psi_sBPM-phi)
    return zs0
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Continuum Phase function
#Se corrigen los saltos de la funcion de fase, por medio del promedio que lleva la funcion
def ffasecont_v01(phase):
	aver=phase[0]
	averup=avermid=averdwn=0
	for i in range(len(phase)-1):
		n=0
		flag=1
		while (flag == 1):
			#Se revisa hacia arriba y hacia abajo
			averup =(aver*(i+1)+phase[i+1]+2.0*pi*(n+1))/(i+2.0)
			avermid=(aver*(i+1)+phase[i+1]+2.0*pi*n)/(i+2)
			averdwn=(aver*(i+1)+phase[i+1]+2.0*pi*(n-1))/(i+2.0)
			if (fabs(aver-averup) < fabs(aver-avermid)):
				n=n+1
			elif (fabs(aver-averdwn)<fabs(aver-avermid)):
				n=n-1
			else:
				flag=0
	phase[i+1]=phase[i+1]+2.0*pi*n
	aver=(aver*(i+1)+phase[i+1])/(i+2)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Average
def doaverage(lista):
    return sum(lista)/len(lista)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Diferencia al cuadrado de dos datos
def cuaddif(aver,data):
    return (aver-data)*(aver-data)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Desviacion estandar de una lista
def dodesvstd(lista):
	lista_aver=[]
	n=len(lista)
	aver=doaverage(lista)
	for i in range(n):
		lista_aver.append(aver)
		list_desv=list(map(cuaddif,lista_aver,lista))
	return sqrt(sum(list_desv)/(n-1))
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Eliminar datos con dispersion superior a N*sigma
def eliminar_data_nsigma(lista,N):
	continuar=1
	while (continuar==1):
		l=len(lista)
		if l>1:
			#Promedio
			aver=doaverage(lista)
			##Desviacion standard de la lista
			sd=dodesvstd(lista)
			#eliminando datos mayores a N*sigma
			ind=0
			nparasacar=0
			while (ind < l and nparasacar==0):
				if(fabs(lista[ind]-aver) <= fabs(N*sd)):
					#print lista[ind]-aver, N*sd
					ind=ind+1
				else:
					lista.pop(ind)
					nparasacar=1

				if (nparasacar==0):
					continuar=0
				else:
					continuar=0
	return lista
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Buscar elemento mas cercano a "posicion" en coordenada s y retornar el indice
def buscar_mas_cercano_s(s,posicion,condicion):
	distBPM=float('inf')
	if (condicion=="antes"):
		for ss in s:
			if((ss-posicion)<distBPM and distBPM>0):
				distBPM=ss-posicion
				sBPM=ss
	if (condicion=="despues"):
		for ss in s:
			if((posicion-ss)<distBPM and distBPM>0):
				distBPM=posicion-ss
				sBPM=ss
	if (condicion==None):
		for ss in s:
			if(fabs((posicion-ss))<distBPM):
				distBPM=fabs(posicion-ss)
				sBPM=ss
	return s.index(sBPM)
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Definir region
#Antes del error, despues del error, punto del error, no promediar
def definir_region(s,ant_lim_inf,ant_lim_sup,s_error,desp_lim_inf,desp_lim_sup):
	region=[]
	for ss in s:
		if ss<ant_lim_inf:
			tipo_de_region="no_promediar"
		elif ss<ant_lim_sup:
			tipo_de_region="antes_del_error"
		elif ss<desp_lim_inf:
			tipo_de_region="no_promediar"
		elif ss<desp_lim_sup:
			tipo_de_region="desp_del_error"
		elif ss==s_error:
			tipo_de_region="s_error"
		else:
			tipo_de_region="no_promediar"
		region.append(tipo_de_region)
	return region
#-----------------------------------------------------------------------
def definir_region2(s,ant_lim_inf,ant_lim_sup,s_error,desp_lim_inf,desp_lim_sup):
	region=[]
	for ss in s:
		if ss<ant_lim_inf:
			tipo_de_region="no_promediar"
		elif ss<ant_lim_sup:
			tipo_de_region="antes_del_error"
		elif ss<desp_lim_inf:
			tipo_de_region="no_promediar"
		elif ss<desp_lim_sup:
			tipo_de_region="desp_del_error"
		elif ss==s_error:
			tipo_de_region="s_error"
		else:
			tipo_de_region="no_promediar"
		region.append(tipo_de_region)
	return region
#-----------------------------------------------------------------------
def definir_region3(s,ant_lim_inf,ant_lim_sup,desp_lim_inf,desp_lim_sup):
	region=[]
	for ss in s:
		if ss<ant_lim_inf:
			tipo_de_region="no_promediar"
		elif ss<ant_lim_sup:
			tipo_de_region="antes_del_error"
		elif ss<desp_lim_inf:
			tipo_de_region="no_promediar"
		elif ss<desp_lim_sup:
			tipo_de_region="desp_del_error"
		else:
			tipo_de_region="no_promediar"
		region.append(tipo_de_region)
	return region


#-----------------------------------------------------------------------
#Leer las posiciones de un archivo de datos sdds en uno de los ejes
def leer_sdds(laorbita,eje):
	s=[]
	z=[]
	datos=leer_filedatos_v01(laorbita)
	if(eje=='-x'):
		for fila in datos:
			#Determinar si los datos son validos
			#print fila
			if (fila[4]=="1"):#dato valido
				s.append(round(float(fila[0]),2))
				z.append(float(fila[1])*1.0e-3)#Convertir a [m]
	if(eje=="-y"):
		for fila in datos:
			if (fila[9]=="1"):#dato valido
				s.append(round(float(fila[5]),2))
				z.append(float(fila[6])*1.0e-3)#Convertir a [m]
	return s,z
#-----------------------------------------------------------------------
#Leer las posiciones de un archivo de datos sdds en uno de los ejes y saca nombre del bpm
def leer_sdds2(laorbita,eje):
	s=[]
	z=[]
	nbpm=[]
	datos=leer_filedatos_v01(laorbita)
	if(eje=='-x'):
		for fila in datos:
			#Determinar si los datos son validos
			#print fila
			if (fila[4]=="1"):#dato valido
				s.append(round(float(fila[0]),2))
				z.append(float(fila[1])*1.0e-3)#Convertir a [m]
				nbpm.append(fila[10])
	if(eje=="-y"):
		for fila in datos:
			if (fila[9]=="1"):#dato valido
				s.append(round(float(fila[5]),2))
				z.append(float(fila[6])*1.0e-3)#Convertir a [m]
				nbpm.append(fila[11])
	return s,z,nbpm
#-----------------------------------------------------------------------

#Leer betas y mus de un eje
def leer_beta_mu(betasfile,eje):
	nameel=[] # Element names
	sel=[]    # element s coordinate
	betz=[]   # beta(s) functions in z
	psiz=[]   # mu(s)*2*PI, mu's are read from lattice.asc file or equivalent
	datos=leer_filedatos_v01(betasfile)
	for fila in datos:
		nameel.append(fila[0])
		sel.append(round(float(fila[1]),2))
	if(eje=='-x'):
		for fila in datos:
			betz.append(float(fila[2]))
			psiz.append(2.0*pi*float(fila[10]))# Psi=2*pi*mu
	if(eje=="-y"):
		for fila in datos:
			betz.append(float(fila[4]))
			psiz.append(2.0*pi*float(fila[11]))# Psi=2*pi*mu
	return nameel,sel,betz,psiz
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Leer betas y mus de un eje
def leer_beta_mu2(betasfile,eje):
	nameel=[] # Element names
	sel=[]    # element s coordinate
	betz=[]   # beta(s) functions in z
	alfz=[]   # alpha(s) functions
	psiz=[]   # mu(s)*2*PI, mu's are read from lattice.asc file or equivalent
	datos=leer_filedatos_v01(betasfile)
	for fila in datos:
		nameel.append(fila[0])
		sel.append(round(float(fila[1]),2))
	if(eje=='-x'):
		for fila in datos:
			betz.append(float(fila[2]))
			alfz.append(float(fila[5]))
			psiz.append(2.0*pi*float(fila[10]))# Psi=2*pi*mu
	if(eje=="-y"):
		for fila in datos:
			betz.append(float(fila[4]))
			alfz.append(float(fila[5]))
			psiz.append(2.0*pi*float(fila[11]))# Psi=2*pi*mu
	return nameel,sel,betz,psiz,alfz
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Leer betas y mus de un eje
def leer_beta_mu3(betasfile,eje):
	nameel=[] # Element names
	sel=[]    # element s coordinate
	betz=[]   # beta(s) functions in z
	alfz=[]   # alpha(s) functions
	psiz=[]   # mu(s)*2*PI, mu's are read from lattice.asc file or equivalent
	datos=leer_filedatos_v01(betasfile)
	for fila in datos:
		nameel.append(fila[0])
		sel.append(float(fila[1]))
	if(eje=='-x'):
		for fila in datos:
			betz.append(float(fila[2]))
			alfz.append(float(fila[5]))
			psiz.append(2.0*pi*float(fila[10]))# Psi=2*pi*mu
	if(eje=="-y"):
		for fila in datos:
			betz.append(float(fila[4]))
			alfz.append(float(fila[6]))
			psiz.append(2.0*pi*float(fila[11]))# Psi=2*pi*mu
	return nameel,sel,betz,psiz,alfz
#-----------------------------------------------------------------------



#Leer archivo de datos separados por espacios
def leer_filedatos_v01(nombrearch):
	data=[]
	fdatos=open(nombrearch,'r')
	for line in fdatos:
		sline = line.split(None)
		data.append(sline)
	fdatos.close()
	return data
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Escribir archivo de datos separados por espacios
def escribir_archivo(nombrearch,data,parametro):
	#El parametro es acorde con la funcion open de Python
	fdatos=open(nombrearch,parametro)
	for fila in data:
		linea=join(fila)+'\n'
		fdatos.write(linea)
		fdatos.close()
#-----------------------------------------------------------------------
