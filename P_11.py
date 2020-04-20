"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving initial value problem using Runge kutta method

"""

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

# Solving the problem using variable saperation 
A=np.array([[1,2,-1],[1,0,1],[1,2,0]])    # Matrix of coeffients of ui on right hand side
b=np.array([1,1,1])                       # Vector of coeffients of ui' on left hand side
c1=np.array([1,-2,1])                     # Coeffients of e^-t on right hand side
u0=np.array([3,-1,1])                     # Initial values of ui
ei=lin.eig(A)                             # Finding the eigen space of A 
e=ei[0]                                   # Defining e as eigen numbers 
v=ei[1]                                   # Defining v as eigen vector matrix
c=np.dot(v,c1)                            # finding coeffients of e^-t in diagonal space

t0=0                                      # defining limits of t and step-size h
tn=1
h=0.01

hi=np.linspace(t0,tn,int((tn-t0)/h))      # Creating mesh of t for t0-tn
y=np.zeros([len(hi),3])                   # Initializing y to zeros
y[0,:]=np.dot(v,u0)                       # Setting initial values in diagonal space
for i in range(len(hi)-1):                # Solving using 4th order runge kutta method
    k1=h*((e[:]*y[i,:])+(c[:]*np.exp(hi[i])))
    k2=h*((e[:]*(y[i,:]+(0.5*k1[:])))+(c[:]*np.exp(hi[i]+(h*0.5))))
    k3=h*((e[:]*(y[i,:]+(0.5*k2[:])))+(c[:]*np.exp(hi[i]+(h*0.5))))
    k4=h*((e[:]*(y[i,:]+(k3[:])))+(c[:]*np.exp(hi[i]+h)))
    y[i+1,:]=y[i,:]+(1/6*(k1[:]+(2*k2[:])+(2*k3[:])+k4[:]))

u=np.transpose(np.matmul(lin.inv(v),np.transpose(y)))   # Finding the values of ui in original space


# Plotting the solutions of ui
plt.plot(hi,u[:,0],'r',label="u1")
plt.plot(hi,u[:,1],'k',label="u2")
plt.plot(hi,u[:,2],'b',label="u3")
plt.xlabel("t")
plt.ylabel("u")
plt.xlim([0,1])
plt.title("Plots of u vs t")
plt.legend()
plt.show()
