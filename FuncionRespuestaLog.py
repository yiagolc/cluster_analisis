# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:49:14 2021

@author: yagol
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from operator import itemgetter


scale = 3  #  Escala el valor de los errores pues no son poisson
ilft  = 1
irht  = 10
corr_ang=0.4877901472720794

# para (0-25) es 0.5611079105900089, para (25-37) es 0.5767182701600874,
# para (37-46) es 0.4877901472720794 y para (46-53) es 0.3781501448175211
################# lectura del fichero ##########################

file=open("of_clustp_H3.txt","r")
#mdat = np.loadtxt(file)
lineasH=file.readlines()
nH=len(lineasH)
file.close()

# Arrays con los datos

PrimCRH=np.zeros(nH)
EnePCRH=np.zeros(nH)
MnHPCRH=np.zeros(nH)
NShowH=np.zeros(nH)
NtSePsH=np.zeros(nH)
NtCltsH=np.zeros(nH)
NtGamH=np.zeros(nH)
NteleH=np.zeros(nH)
NMuH=np.zeros(nH)
NNeutH=np.zeros(nH)
NProtH=np.zeros(nH)
NOthrH=np.zeros(nH)
NClElH=np.zeros(nH)
NClMuH=np.zeros(nH)
NClMxH=np.zeros(nH)
NClRmH=np.zeros(nH)
Ntel50H=np.zeros(nH)
Nte100H=np.zeros(nH)
Nte150H=np.zeros(nH)
Nte200H=np.zeros(nH)
Nte300H=np.zeros(nH)
Nte500H=np.zeros(nH)
Ntel1KH=np.zeros(nH)
Ntel2KH=np.zeros(nH)

# Asignacion de datos a cada array

for i in range(nH):
    l=lineasH[i].split()
    PrimCRH[i]=l[0]
    EnePCRH[i]=l[1]
    MnHPCRH[i]=l[2]
    NShowH[i]=l[3]
    NtSePsH[i]=l[4]
    NtCltsH[i]=l[5]
    NtGamH[i]=l[6]
    NteleH[i]=l[7]
    NMuH[i]=l[8]
    NNeutH[i]=l[9]
    NProtH[i]=l[10]
    NOthrH[i]=l[11]
    NClElH[i]=l[12]
    NClMuH[i]=l[13]
    NClMxH[i]=l[14]
    NClRmH[i]=l[15]
    Ntel50H[i]=l[16]
    Nte100H[i]=l[17]
    Nte150H[i]=l[18]
    Nte200H[i]=l[19]
    Nte300H[i]=l[20]
    Nte500H[i]=l[21]
    Ntel1KH[i]=l[22]
    Ntel2KH[i]=l[23]
    
    
    
    
###################### Definimos funciones

def F(E1,E2):
    f=1.8/1.7*(E1**(-1.7)-E2**(-1.7))*10**4
    return(f)
def gauss(x,sigma1,media1,a1):
    y=(1/(sigma1*np.sqrt(2*np.pi)))*np.e**(-(x-media1)**2/(2*sigma1**2))*(a1*x)
    return(y)
def gaussS(x,c1,sigma1,media1):
    y=c1*np.e**(-(x-media1)**2/(2*sigma1**2))
    return(y)



J=np.zeros(len(EnePCRH))
#infabs=10**(np.log10(EnePCRH[i])-0.25/2)
for i in range(len(EnePCRH)):
    a=EnePCRH[i]
    inf1=a-0.25/2
    sup1=a+0.25/2
    inf=10**inf1
    sup=10**sup1
    J[i]=F(inf,sup)

J=J*corr_ang

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,J,"r.",label="spectrum of protons")
plt.xlabel("Energia $\\left[log_{10}(GeV)\\right]$")
plt.ylabel("Integral de J(E)")
plt.figure(figsize=[8,3])
plt.plot(EnePCRH,NMuH/(NShowH*0.25),"b.",label="yield function data")
plt.xlabel("Energia $\\left[log_{10}(GeV)\\right]$")
plt.ylabel("Yield Function")


####################### MUONES ############################

G=J*NMuH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*NMuH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.text(3.4,10**0*3.5,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.plot(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de muones")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.ylim(10**(-1)*2.5,10**1*4.5)

plt.xlim(0,4.4)

errores=scale*np.sqrt(NMuH)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor


#p=[0.3,1.5,0,0.1]

desplazamiento=1

Eneprime=EnePCRH-EnePCRH[desplazamiento]

pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
smu= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento],5.2-EnePCRH[desplazamiento],1000)

plt.semilogy(x+EnePCRH[desplazamiento],gauss(x,pmu[0],pmu[1],pmu[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")



for i in range(2,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
Gmu=G.copy() 
Erroresmu=errores.copy()

### Para guardar la funcion de respuesta en un fichero por columnas

#file=open("Gmu.txt","a")

#aux=np.zeros(3)
#for i in range(len(Gmu)):
#    file.write(str(round(EnePCRH[i],3)))
#    file.write("\t")
#    file.write(format(Erroresmu[i],".3E"))
#    file.write("\t")
#    file.write(str(round(Gmu[i],3)))  
#    file.write("\n")
    ##aux[0]=round(EnePCRH[i],3)
    ##aux[1]=format(Erroresmu[i],".2E")
    ##aux[2]=round(Gmu[i],3)
    ##np.savetxt(file,aux,delimiter="\t",newline=" ")
    ##file.write("\n")

#file.close()
  
  
####################### ELECTRONES ############################ 

G=J*NteleH/(NShowH*0.25)
integral=0

for i in range(nH):
    integral=integral+0.25*NteleH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")  para escribir en not cientifica, el numero es el numero de decimales

integral=round(integral,3)
integral=str(integral)    

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.3,10**(-1)*7,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-2)*8,10**(0)*5)

errores=3*np.sqrt(NteleH)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor


desplazamiento=1

Eneprime=EnePCRH-EnePCRH[desplazamiento]

pe,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
se= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento],5.2-EnePCRH[desplazamiento],1000)

plt.plot(x+EnePCRH[desplazamiento],gauss(x,pe[0],pe[1],pe[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

print(G[2],errores[2])
for i in range(2,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
  
####################### CLUSTER ELECTRONES ############################ 

G=J*NClElH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*NClElH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta cluster electrones")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.4,10**(-2)*5,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-3)*6,10**(-1)*1.6)

p0=[0,1.5,1.8]

pce,s=curve_fit(gaussS,EnePCRH,G,maxfev=1000000)
sce= np.sqrt(np.diag(s))

x=np.linspace(0.3,5.2,1000)

#plt.plot(x,gaussS(x,pe[0],pce[1],pce[2]),label="ajuste gaussiano")
plt.legend(loc="upper right")

errores=3*np.sqrt(NClElH)*J/(NShowH*0.25)

for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
####################### CLUSTER MUONES ############################ 

G=J*NClMuH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*NClMuH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,5)
print(integral)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta cluster muones")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(0.2,3*10**(-5),"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-6)*8,10**(-4))

errores=3*np.sqrt(NClMuH)*J/NShowH

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

pcmu,s=curve_fit(gauss,EnePCRH,G,maxfev=1000000)
scmu= np.sqrt(np.diag(s))


pcmu,s=curve_fit(gauss,EnePCRH,G,maxfev=1000000,sigma=errores,p0=pcmu)
scmu= np.sqrt(np.diag(s))

x=np.linspace(0.3,5.2,1000)

#plt.plot(x,gauss(x,pcmu[0],pcmu[1],pcmu[2],pcmu[3]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper left")

errores=3*np.sqrt(NClMuH)*J/NShowH

for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
  
  ####################### CLUSTER Mx ############################ 

G=J*NClMxH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*NClMxH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,5)
print(integral)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta cluster mixto")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(0.2,10**(-4)*6,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-4),10**(-3)*1.25)

errores=3*np.sqrt(NClMxH)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

pcmx,s=curve_fit(gauss,EnePCRH,G,maxfev=1000000)
scmx= np.sqrt(np.diag(s))


pcmx,s=curve_fit(gauss,EnePCRH,G,maxfev=1000000,sigma=errores,p0=pcmu)
scmx= np.sqrt(np.diag(s))

x=np.linspace(0.3,5.2,1000)

#plt.plot(x,gauss(x,pcmx[0],pcmx[1],pcmx[2],pcmx[3]),label="ajuste gaussiano$\\cdot$parabola")
plt.legend(loc="upper left")

errores=3*np.sqrt(NClMxH)*J/NShowH

for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")


  ####################### 50 ############################ 

G=J*Ntel50H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Ntel50H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 50 y 100 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,0.55,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-3)*1.5,10**0*2.2)

errores=3*np.sqrt(Ntel50H)*J/(0.25*NShowH)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento50=1

Eneprime=EnePCRH-EnePCRH[desplazamiento50]

p50,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s50= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento50],5.2-EnePCRH[desplazamiento50],1000)

plt.plot(x+EnePCRH[desplazamiento50],gauss(x,p50[0],p50[1],p50[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Ntel50H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
G50=G.copy()
  
  ####################### 100 ############################ 

G=J*Nte100H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte100H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 100 y 150 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,0.2,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-3),10**0)

errores=3*np.sqrt(Nte100H)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento100=1

Eneprime=EnePCRH-EnePCRH[desplazamiento100]

p100,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s100= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento100],5.2-EnePCRH[desplazamiento100],1000)

plt.plot(x+EnePCRH[desplazamiento100],gauss(x,p100[0],p100[1],p100[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte100H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
G100=G.copy()
  
  ####################### 150 ############################ 

G=J*Nte150H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte150H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 150 y 200 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,0.15,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-3),10**0*1/1.5)

errores=3*np.sqrt(Nte150H)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento150=1

Eneprime=EnePCRH-EnePCRH[desplazamiento150]

p150,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s150= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento150],5.2-EnePCRH[desplazamiento150],1000)

plt.plot(x+EnePCRH[desplazamiento150],gauss(x,p150[0],p150[1],p150[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte150H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
G150=G.copy()  
  
    ####################### 200 ############################ 

G=J*Nte200H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte200H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 200 y 300 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,0.15,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-4)*2,10**(0))

errores=3*np.sqrt(Nte200H)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento200=1

Eneprime=EnePCRH-EnePCRH[desplazamiento200]

p200,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s200= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento200],5.2-EnePCRH[desplazamiento200],1000)

plt.plot(x+EnePCRH[desplazamiento200],gauss(x,p200[0],p200[1],p200[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte200H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")

G200=G.copy()
  
    ####################### 300 ############################ 

G=J*Nte300H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte300H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 300 y 500 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,0.15,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-2)*2,10**0**1/1.5)

errores=3*np.sqrt(Nte300H)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento300=1

Eneprime=EnePCRH-EnePCRH[desplazamiento300]

p300,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s300= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento300],5.2-EnePCRH[desplazamiento300],1000)

plt.plot(x+EnePCRH[desplazamiento300],gauss(x,p300[0],p300[1],p300[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte300H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")

G300=G.copy()  

      ####################### 500 ############################ 

G=J*Nte500H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte300H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 0.5 y 1 GeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,0.15,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-2)*2,10**0*1/1.8)

errores=3*np.sqrt(Nte500H)*J/(NShowH*0.25)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento500=1

Eneprime=EnePCRH-EnePCRH[desplazamiento500]

p500,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s500= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento500],5.2-EnePCRH[desplazamiento500],1000)

plt.plot(x+EnePCRH[desplazamiento500],gauss(x,p500[0],p500[1],p500[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte500H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
G500=G.copy()
  
      ####################### 1k ############################ 

G=J*Ntel1KH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Ntel1KH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 1 y 2 GeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,10**(-1)*(1/2),"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-2),2*10**(-1))

errores=3*np.sqrt(Ntel1KH)*J/(NShowH*0.24)

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

#p=[ 5.92950336,  2.24212364,  -1.46579629, -2.25103015]
 
desplazamiento1K=3

Eneprime=EnePCRH-EnePCRH[desplazamiento1K]

p1K,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s1K= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento1K],5.2-EnePCRH[desplazamiento1K],1000)

plt.plot(x+EnePCRH[desplazamiento1K],gauss(x,p1K[0],p1K[1],p1K[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Ntel1KH)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")

G1K=G.copy()

    ####################### 2k ############################ 

G=J*Ntel2KH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Ntel2KH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,3)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.semilogy(EnePCRH,G,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones con 2 GeV o más")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.25,2*10**(-2)*1.1,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)
plt.ylim(10**(-2)*1/1.7,10**(-1)*1/1.9)

errores=3*np.sqrt(Ntel2KH)*J/NShowH

numbers_sort = sorted(enumerate(errores), key=itemgetter(1),  reverse=True)

values=np.zeros(len(EnePCRH))

for i in range(len(EnePCRH)):
    index,values[i]=numbers_sort[i]
    
for i in range(len(EnePCRH)):
    if values[len(EnePCRH)-i-1]!=0:
        valor=values[len(EnePCRH)-i-1]
        break

for i in range(len(errores)):
    if errores[i]==0:
        errores[i]=valor

desplazamiento2K=3

Eneprime=EnePCRH-EnePCRH[desplazamiento2K]

p2K,s=curve_fit(gauss,Eneprime,G,maxfev=1000000)
s2K= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamiento2K],5.2-EnePCRH[desplazamiento2K],1000)

plt.plot(x+EnePCRH[desplazamiento2K],gauss(x,p2K[0],p2K[1],p2K[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Ntel2KH)*J/(NShowH*0.25)

for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G[i],yerr=errores[i],ecolor="k")
  
G2K=G.copy() 
  
################# Grafica conjunta

plt.figure(figsize=[11,3])
ax = plt.subplot(111)

plt.title("Función de respuesta para electrones")

plt.semilogy(EnePCRH,G50,"r.",label="50-100 KeV")
plt.semilogy(EnePCRH,G100,"v",label="100-150 KeV",color="orange")
plt.semilogy(EnePCRH,G150,"y^",label="150-200 KeV")
plt.semilogy(EnePCRH,G200,"gs",label="200-300 KeV")
plt.semilogy(EnePCRH,G300,"cp",label="300-500 KeV")
plt.semilogy(EnePCRH,G500,"bH",label="500-1000 KeV")
plt.semilogy(EnePCRH,G1K,"p",label="1000-2000 KeV",color="darkviolet")
plt.semilogy(EnePCRH,G2K,"k*",label="2000 y más KeV")

xd=np.linspace(0-EnePCRH[desplazamiento50],5.2-EnePCRH[desplazamiento50],1000)

plt.plot(xd+EnePCRH[desplazamiento50],gauss(xd,p50[0],p50[1],p50[2]),"r-")
plt.plot(xd+EnePCRH[desplazamiento100],gauss(xd,p100[0],p100[1],p100[2]),"-",color="orange")
plt.plot(xd+EnePCRH[desplazamiento150],gauss(xd,p150[0],p150[1],p150[2]),"y-")
plt.plot(xd+EnePCRH[desplazamiento200],gauss(xd,p200[0],p200[1],p200[2]),"g-")
plt.plot(xd+EnePCRH[desplazamiento300],gauss(xd,p300[0],p300[1],p300[2]),"c-")
plt.plot(xd+EnePCRH[desplazamiento500],gauss(xd,p500[0],p500[1],p500[2]),"b-")
plt.plot(xd+EnePCRH[desplazamiento1K],gauss(xd,p1K[0],p1K[1],p1K[2]),"-",color="darkviolet")
plt.plot(xd+EnePCRH[desplazamiento2K],gauss(xd,p2K[0],p2K[1],p2K[2]),"k-")

plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.ylim(10**(-3)*2,10**(0)*2)
plt.xlim(0,4.4)


# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))