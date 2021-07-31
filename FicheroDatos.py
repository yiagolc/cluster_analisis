# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:36:33 2021

@author: yagol
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,fsolve,minimize
from operator import itemgetter


################# lectura del fichero ##########################

file=open("of_clustp_H1.txt","r")
lineasH=file.readlines()
nH=len(lineasH)
file.close()

corr_ang=0.5611079105900089

# para (0-25) es 0.5611079105900089, para (25-37) es 0.5767182701600874,
# para (37-46) es 0.4877901472720794 y para (46-53) es 0.3781501448175211

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
def gauss(x,sigma1,media1,a):
    y=1/(sigma1*np.sqrt(2*np.pi))*np.e**(-(x-media1)**2/(2*sigma1**2))*(a*x)
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


nelectrones=8
nparametros=9+3+1

G=np.zeros([nelectrones,nparametros])

######################### 50 #################################

G50=J*Ntel50H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Ntel50H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G50,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 50 y 100 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,1,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(-8,8)

errores=3*np.sqrt(Ntel50H)*J/NShowH

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

p50,s=curve_fit(gauss,Eneprime,G50,maxfev=1000000)
s50= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento50],8-EnePCRH[desplazamiento50],1000)

plt.plot(x+EnePCRH[desplazamiento50],gauss(x,p50[0],p50[1],p50[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Ntel50H)*J/NShowH


for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G50[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[0,i+6]=round(p50[i],3)
    
    
def f50(x):
    return(-1/(p50[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento50])-p50[1])**2/(2*p50[0]**2))*(p50[2]*(x[0]-EnePCRH[desplazamiento50])))

x=minimize(f50,1.1)
maximo50=-x.fun
print(maximo50)

def f501(x):
    return(1/(p50[0]*np.sqrt(2*np.pi))*np.e**(-(x-p50[1])**2/(2*p50[0]**2))*(p50[2]*x)-[maximo50/2])



solucion1=fsolve(f501,float(x.x)-EnePCRH[desplazamiento50]-0.5)+EnePCRH[desplazamiento50]
solucion2=fsolve(f501,float(x.x)-EnePCRH[desplazamiento50]+0.5)+EnePCRH[desplazamiento50]
FWHM=solucion2-solucion1

G[0,4]=round(float(FWHM),3)
G[0,3]=round(float(x.x),3)
G[0,2]=round(float(maximo50),3)
G[0,5]=integral
G[0,-1]=desplazamiento50


for i in range(len(s50)):
    G[0,9+i]=round(s50[i],3)

######################### 100 #################################

G100=J*Nte100H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte100H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G100,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 100 y 150 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,0.5,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(Nte100H)*J/NShowH

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

p100,s=curve_fit(gauss,Eneprime,G100,maxfev=1000000)
s100= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento100],8-EnePCRH[desplazamiento100],1000)

plt.plot(x+EnePCRH[desplazamiento100],gauss(x,p100[0],p100[1],p100[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte100H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G100[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[1,i+6]=round(p100[i],3)
    
    
def f100(x):
    return(-1/(p100[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento100])-p100[1])**2/(2*p100[0]**2))*(p100[2]*(x[0]-EnePCRH[desplazamiento100])))

x=minimize(f100,1.1)
maximo=-x.fun

def f1001(x):
    return(1/(p100[0]*np.sqrt(2*np.pi))*np.e**(-(x-p100[1])**2/(2*p100[0]**2))*(p100[2]*x)-[maximo/2])


solucion1=fsolve(f1001,float(x.x)-EnePCRH[desplazamiento100]-0.5)+EnePCRH[desplazamiento100]
solucion2=fsolve(f1001,float(x.x)-EnePCRH[desplazamiento100]+0.5)+EnePCRH[desplazamiento100]

FWHM=solucion2-solucion1
G[1,4]=round(float(FWHM),3)
G[1,3]=round(float(x.x),3)
G[1,2]=round(float(maximo),3)
G[1,5]=integral
G[1,-1]=desplazamiento100


for i in range(len(s50)):
    G[1,9+i]=round(s100[i],3)

######################### 150 #################################

G150=J*Nte150H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte150H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G150,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 150 y 200 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log_{10}(E / GeV)$")
plt.text(3.25,0.3,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(Nte150H)*J/NShowH

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

p150,s=curve_fit(gauss,Eneprime,G150,maxfev=1000000)
s150= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento150],8-EnePCRH[desplazamiento150],1000)

plt.plot(x+EnePCRH[desplazamiento150],gauss(x,p150[0],p150[1],p150[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte150H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G150[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[2,i+6]=round(p150[i],3)
    
    
def f150(x):
     return(-1/(p150[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento150])-p150[1])**2/(2*p150[0]**2))*(p150[2]*(x[0]-EnePCRH[desplazamiento150])))

x=minimize(f150,1.1)
maximo=-x.fun

def f1501(x):
    return(1/(p150[0]*np.sqrt(2*np.pi))*np.e**(-(x-p150[1])**2/(2*p150[0]**2))*(p150[2]*x)-[maximo/2])


solucion1=fsolve(f1501,float(x.x)-EnePCRH[desplazamiento150]-0.5)+EnePCRH[desplazamiento150]
solucion2=fsolve(f1501,float(x.x)-EnePCRH[desplazamiento150]+0.5)+EnePCRH[desplazamiento150]

FWHM=solucion2-solucion1
G[2,4]=round(float(FWHM),3)
G[2,3]=round(float(x.x),3)
G[2,2]=round(float(maximo),3)
G[2,5]=integral
G[2,-1]=desplazamiento150

for i in range(len(s50)):
    G[2,9+i]=round(s150[i],3)

######################### 200 #################################

G200=J*Nte200H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte200H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G200,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 200 y 300 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,0.3,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(Nte200H)*J/NShowH

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

p200,s=curve_fit(gauss,Eneprime,G200,maxfev=1000000)
s200= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento200],8-EnePCRH[desplazamiento200],1000)

plt.plot(x+EnePCRH[desplazamiento200],gauss(x,p200[0],p200[1],p200[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte200H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G200[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[3,i+6]=round(p200[i],3)
    
    
def f200(x):
    return(-1/(p200[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento200])-p200[1])**2/(2*p200[0]**2))*(p200[2]*(x[0]-EnePCRH[desplazamiento200])))

x=minimize(f200,1.1)
maximo=-x.fun

def f2001(x):
    return(1/(p200[0]*np.sqrt(2*np.pi))*np.e**(-(x-p200[1])**2/(2*p200[0]**2))*(p200[2]*x)-[maximo/2])


solucion1=fsolve(f2001,float(x.x)-EnePCRH[desplazamiento200]-0.5)+EnePCRH[desplazamiento200]
solucion2=fsolve(f2001,float(x.x)-EnePCRH[desplazamiento200]+0.9)+EnePCRH[desplazamiento200]

FWHM=solucion2-solucion1

G[3,4]=round(float(FWHM),3)
G[3,3]=round(float(x.x),3)
G[3,2]=round(float(maximo),3)
G[3,5]=integral
G[3,-1]=desplazamiento200


for i in range(len(s200)):
    G[3,9+i]=round(s200[i],3)
    
######################### 300 #################################

G300=J*Nte300H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte300H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G300,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 300 y 500 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,0.3,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(Nte300H)*J/NShowH

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

p300,s=curve_fit(gauss,Eneprime,G300,maxfev=1000000)
s300= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento300],8-EnePCRH[desplazamiento300],1000)

plt.plot(x+EnePCRH[desplazamiento300],gauss(x,p300[0],p300[1],p300[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte300H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G300[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[4,i+6]=round(p300[i],3)
    
    
def f300(x):
    return(-1/(p300[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento300])-p300[1])**2/(2*p300[0]**2))*(p300[2]*(x[0]-EnePCRH[desplazamiento300])))

x=minimize(f300,1.1)
maximo=-x.fun

def f3001(x):
    return(1/(p300[0]*np.sqrt(2*np.pi))*np.e**(-(x-p300[1])**2/(2*p300[0]**2))*(p300[2]*x)-[maximo/2])


solucion1=fsolve(f3001,float(x.x)-EnePCRH[desplazamiento300]-0.5)+EnePCRH[desplazamiento300]
solucion2=fsolve(f3001,float(x.x)-EnePCRH[desplazamiento300]+0.5)+EnePCRH[desplazamiento300]

FWHM=solucion2-solucion1

G[4,4]=round(float(FWHM),3)
G[4,3]=round(float(x.x),3)
G[4,2]=round(float(maximo),3)
G[4,5]=integral
G[4,-1]=desplazamiento300

for i in range(len(s300)):
    G[4,9+i]=round(s300[i],3)
    
######################### 500 #################################

G500=J*Nte500H/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Nte500H[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G500,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 500 y 1000 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,0.2,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(Nte500H)*J/NShowH

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

p500,s=curve_fit(gauss,Eneprime,G500,maxfev=1000000)
s500= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento500],8-EnePCRH[desplazamiento500],1000)

plt.plot(x+EnePCRH[desplazamiento500],gauss(x,p500[0],p500[1],p500[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Nte500H)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G500[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[5,i+6]=round(p500[i],3)
    
def f500(x):
    return(-1/(p500[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento500])-p500[1])**2/(2*p500[0]**2))*(p500[2]*(x[0]-EnePCRH[desplazamiento500])))

x=minimize(f500,1.1)
maximo=-x.fun

def f5001(x):
    return(1/(p500[0]*np.sqrt(2*np.pi))*np.e**(-(x-p500[1])**2/(2*p500[0]**2))*(p500[2]*x)-[maximo/2])


solucion1=fsolve(f5001,float(x.x)-EnePCRH[desplazamiento500]-0.6)+EnePCRH[desplazamiento500]
solucion2=fsolve(f5001,float(x.x)-EnePCRH[desplazamiento500]+0.6)+EnePCRH[desplazamiento500]

FWHM=solucion2-solucion1

G[5,4]=round(float(FWHM),3)
G[5,3]=round(float(x.x),3)
G[5,2]=round(float(maximo),3)
G[5,5]=integral
G[5,-1]=desplazamiento500


for i in range(len(s500)):
    G[5,9+i]=round(s500[i],3)
    
######################### 1000 #################################

G1K=J*Ntel1KH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Ntel1KH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G1K,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones entre 1000 y 2000 KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,0.075,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(Ntel1KH)*J/NShowH

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


desplazamiento1K=3

Eneprime=EnePCRH-EnePCRH[desplazamiento1K]

p1K,s=curve_fit(gauss,Eneprime,G1K,maxfev=1000000)
s1K= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento1K],8-EnePCRH[desplazamiento1K],1000)

plt.plot(x+EnePCRH[desplazamiento1K],gauss(x,p1K[0],p1K[1],p1K[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Ntel1KH)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G1K[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[6,i+6]=round(p1K[i],3)
    
    
def f1K(x):
    return(-1/(p1K[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento1K])-p1K[1])**2/(2*p1K[0]**2))*(p1K[2]*(x[0]-EnePCRH[desplazamiento1K])))

x=minimize(f1K,1.6)
maximo1k=-x.fun

def f1K1(x):
    return(1/(p1K[0]*np.sqrt(2*np.pi))*np.e**(-(x-p1K[1])**2/(2*p1K[0]**2))*(p1K[2]*x)-[maximo1k/2])

b=x.x
solucion1=fsolve(f1K1,float(x.x)-EnePCRH[desplazamiento1K]-0.2)+EnePCRH[desplazamiento1K]
solucion2=fsolve(f1K1,float(x.x)-EnePCRH[desplazamiento1K]+0.2)+EnePCRH[desplazamiento1K]

FWHM=solucion2-solucion1
G[6,4]=round(float(FWHM),3)
G[6,3]=round(float(x.x),3)
G[6,2]=round(float(maximo1k),3)
G[6,5]=integral
G[6,-1]=desplazamiento1K


for i in range(len(s50)):
    G[6,9+i]=round(s1K[i],3)
    
    
######################### 2000 #################################

G2K=J*Ntel2KH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*Ntel2KH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,G2K,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones con 2000 o mas KeV")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E / GeV)$")
plt.text(3.25,0.03,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

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

p2K,s=curve_fit(gauss,Eneprime,G2K,maxfev=1000000)
s2K= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(-8-EnePCRH[desplazamiento2K],8-EnePCRH[desplazamiento2K],1000)

plt.plot(x+EnePCRH[desplazamiento2K],gauss(x,p2K[0],p2K[1],p2K[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

errores=3*np.sqrt(Ntel2KH)*J/NShowH
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],G2K[i],yerr=errores[i],ecolor="k")
  
for i in range(len(p50)):
    G[7,i+6]=round(p2K[i],3)
    
    
def f2K(x):
    return(-1/(p2K[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamiento2K])-p2K[1])**2/(2*p2K[0]**2))*(p2K[2]*(x[0]-EnePCRH[desplazamiento2K])))

x=minimize(f2K,1.6)
maximo=-x.fun

def f2K1(x):
    return(1/(p2K[0]*np.sqrt(2*np.pi))*np.e**(-(x-p2K[1])**2/(2*p2K[0]**2))*(p2K[2]*x)-[maximo/2])


solucion1=fsolve(f2K1,float(x.x)-EnePCRH[desplazamiento2K]-0.2)+EnePCRH[desplazamiento2K]
solucion2=fsolve(f2K1,float(x.x)-EnePCRH[desplazamiento2K]+0.2)+EnePCRH[desplazamiento2K]

FWHM=solucion2-solucion1

G[7,4]=round(float(FWHM),3)
G[7,3]=round(float(x.x),3)
G[7,2]=round(float(maximo),3)
G[7,5]=integral
G[7,-1]=desplazamiento2K

for i in range(len(s2K)):
    G[7,9+i]=round(s2K[i],3)


#aqui añadiriamos las energias minima y maxima a la tabla del txt

a=[5,1,1.5,2,3,5,1,2,4]


for i in range(len(a)-1):
    G[i,0]=a[i]
    G[i,1]=a[i+1]
    
 
    
 ################# Grafica conjunta

plt.figure(figsize=[11,3])
ax = plt.subplot(111)

plt.title("Función de respuesta para electrones en distintas energías")

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
plt.xlabel("$log(E / GeV)$")
plt.ylim(10**(-3)*4,10**0*3)
plt.xlim(0,4.4)


# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


################# Guardamos en el fichero


file=open("DatosElectrones25-37.txt","w")
a=["#EMin ","EMax","FRMax ","E_FRMax","FWHM","Int","Gauss_E ","Gauss_Sigma ","a ","Inc_G_E","Inc_G_S","Inc_a","DAT_In"]

for i in range(len(a)):
    file.write(a[i])
    file.write("\t")

file.write("\n")
file.close()

file=open("datosElectrones25-37.txt","a")

np.savetxt(file,G,fmt='%10.3f')

file.close()





################# Creamos otro fichero para las que faltan

####################### MUONES ############################

GMu=J*NMuH/(NShowH*0.25)

integral=0

for i in range(nH):
    integral=integral+0.25*NMuH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")

integral=round(integral,4)
integral=str(integral)

plt.figure(figsize=[8,3])
#plt.text(3.2,30,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.plot(EnePCRH,GMu,"r.",label="Datos simulados")
plt.title("Función de respuesta de muones")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")

plt.xlim(0,4.4)


errores=3*np.sqrt(NMuH)*J/NShowH

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

desplazamientomu=1

Eneprime=EnePCRH-EnePCRH[desplazamientomu]

pmu,s=curve_fit(gauss,Eneprime,GMu,maxfev=1000000)
smu= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamientomu],5.2-EnePCRH[desplazamientomu],1000)

#plt.plot(x+EnePCRH[desplazamientomu],gauss(x,pmu[0],pmu[1],pmu[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")

for i in range(1,nH):
  plt.errorbar(EnePCRH[i],GMu[i],yerr=errores[i],ecolor="k")

def fMu(x):
    return(-1/(pmu[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamientomu])-pmu[1])**2/(2*pmu[0]**2))*(pmu[2]*(x[0]-EnePCRH[desplazamientomu])))

x=minimize(fMu,1.6)
maximo=-x.fun

def fMu1(x):
    return(1/(pmu[0]*np.sqrt(2*np.pi))*np.e**(-(x-pmu[1])**2/(2*pmu[0]**2))*(pmu[2]*x)-[maximo/2])


solucion1=fsolve(fMu1,float(x.x)-EnePCRH[desplazamientomu]-0.3)+EnePCRH[desplazamientomu]
solucion2=fsolve(fMu1,float(x.x)-EnePCRH[desplazamientomu]+0.3)+EnePCRH[desplazamientomu]

FWHM=solucion2-solucion1



G2=np.zeros([2,nparametros-1])

G2[0,3]=round(float(FWHM),3)
G2[0,2]=round(float(x.x),3)
G2[0,1]=round(float(maximo),3)
G2[0,4]=integral
G2[0,-1]=desplazamientomu

for i in range(len(s2K)):
    G2[0,8+i]=round(smu[i],3)
    
for i in range(len(p50)):
    G2[0,i+5]=round(pmu[i],3)
    
G2[0,0]=0
    
####################### ELECTRONES ############################ 

GE=J*NteleH/(NShowH*0.25)
integral=0

for i in range(nH):
    integral=integral+0.25*NteleH[i]/NShowH[i]*J[i]
    
#integral=format(integral,".2E")  para escribir en not cientifica, el numero es el numero de decimales

integral=round(integral,4)
integral=str(integral)    

plt.figure(figsize=[8,3])
plt.plot(EnePCRH,GE,"r.",label="Datos simulados")
plt.title("Función de respuesta de electrones")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot GeV$")
plt.xlabel("$log(E/ GeV)$")
plt.text(3.2,2.9,"$integral=$"+integral,bbox={'alpha': 0.5, 'pad': 10})
plt.xlim(0,4.4)

errores=3*np.sqrt(NteleH)*J/NShowH

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


desplazamientoele=1

Eneprime=EnePCRH-EnePCRH[desplazamientoele]

pe,s=curve_fit(gauss,Eneprime,GE,maxfev=1000000)
se= np.sqrt(np.diag(s))


#pmu,s=curve_fit(gauss,Eneprime,G,maxfev=1000000,sigma=errores,p0=pmu)
#cmu= np.sqrt(np.diag(s))

x=np.linspace(0-EnePCRH[desplazamientoele],5.2-EnePCRH[desplazamientoele],1000)

plt.plot(x+EnePCRH[desplazamientoele],gauss(x,pe[0],pe[1],pe[2]),label="ajuste gaussiano$\\cdot$recta")
plt.legend(loc="upper right")
for i in range(1,nH):
  plt.errorbar(EnePCRH[i],GE[i],yerr=errores[i],ecolor="k")
 
  
def fe(x):
    return(-1/(pe[0]*np.sqrt(2*np.pi))*np.e**(-((x[0]-EnePCRH[desplazamientoele])-pe[1])**2/(2*pe[0]**2))*(pe[2]*(x[0]-EnePCRH[desplazamientoele])))

x=minimize(fe,1.6)
maximo=-x.fun

def fe1(x):
    return(1/(pe[0]*np.sqrt(2*np.pi))*np.e**(-(x-pe[1])**2/(2*pe[0]**2))*(pe[2]*x)-[maximo/2])


solucion1=fsolve(fe1,float(x.x)-EnePCRH[desplazamientoele]-0.8)+EnePCRH[desplazamientoele]
solucion2=fsolve(fe1,float(x.x)-EnePCRH[desplazamientoele]+0.8)+EnePCRH[desplazamientoele]

FWHM=solucion2-solucion1 
G2[1,3]=round(float(FWHM),3)
G2[1,2]=round(float(x.x),3)
G2[1,1]=round(float(maximo),3)
G2[1,4]=integral
G2[0,-1]=desplazamientoele
  
for i in range(len(s2K)):
    G2[1,8+i]=round(se[i],3)
    
for i in range(len(p50)):
    G2[1,i+5]=round(pe[i],3)
    
G2[1,0]=1

file=open("DatosMuEle25-37.txt","w")
a=["#EMin ","EMax","FRMax ","E_FRMax","FWHM","Int","Gauss_Sigma ","Gauss_E ","a ","Inc_G_S","Inc_G_E","Inc_a","DAT_In"]

for i in range(len(a)):
    file.write(a[i])
    file.write("\t")

file.write("\n")
file.close()

file=open("DatosMuEle25-37.txt","a")

np.savetxt(file,G2,fmt='%10.3f')

file.close()
