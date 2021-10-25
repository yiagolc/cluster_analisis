# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:55:54 2021

@author: yagol
"""

import numpy as np
import matplotlib.pyplot as plt


angulos=np.array([0,25,37,46,53,90])
valores=[]
for i in range(len(angulos)-2):
    valores.append((angulos[i+1]+angulos[i])/2)


integralMu=np.array([10.332,10.025,6.328,2.208])

plt.figure(figsize=[5,6])
plt.title("Integral frente al ángulo, muones")
plt.plot(valores,integralMu,"bo")
plt.xticks(angulos,angulos)
plt.ylim(0,11)
plt.xlim(0,90)
plt.xlabel("Ángulo de incidencia del primario /grados")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot$")



integralEle=np.array([1.142,0.964,0.555,0.183])

plt.figure(figsize=[5,6])
plt.title("Integral frente al ángulo, electrones")
plt.plot(valores,integralEle,"ro")
plt.xticks(angulos,angulos)
plt.xlim(0,90)
plt.ylim(0,1.2)
plt.xlabel("Ángulo de incidencia del primario /grados")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot$")


integralCEle=np.array([0.045,0.032,0.020,0.005])

plt.figure(figsize=[5,6])
plt.title("Integral frente al ángulo, cluster electrones")
plt.plot(valores,integralCEle,"co")
plt.xticks(angulos,angulos)
plt.xlim(0,90)
plt.ylim(0,0.047)
plt.xlabel("Ángulo de incidencia del primario /grados")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot$")


integralCMu=np.array([0.000054,0.000034,0.000012,0.00000])

plt.figure(figsize=[6,6])
plt.title("Integral frente al ángulo, cluster muones")
plt.plot(valores,integralCMu,"yo")
plt.xticks(angulos,angulos)
plt.xlim(0,90)

plt.xlabel("Ángulo de incidencia del primario /grados")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot$")


integralCMx=np.array([0.00052,0.00039,0.00029,0])

plt.figure(figsize=[5,6])
plt.title("Integral frente al ángulo, cluster mixto")
plt.plot(valores,integralCMx,"ko")
plt.xticks(angulos,angulos)
plt.xlim(0,90)

plt.xlabel("Ángulo de incidencia del primario /grados")
plt.ylabel("$N/s\\cdot m^{2}\\cdot sr\\cdot$")

