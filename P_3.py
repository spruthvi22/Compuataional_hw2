"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving second order initial value problem using Runga-Kutta method

"""

import numpy as np
import matplotlib.pyplot as plt

def dk(k,t):
    return((t*np.exp(t))-t+k)

def dy(y,t,k):
    return(k+y)

def cu_r4(f1,t0,tn,y0,h,k=[]):
    """
    Integrates the initial value problem using 4th order runga kutta method 
    Paramaters:
    f1: Callable, the function to be integrated (defined in y,t,k)
    t0: Double, the lover bound (initial value) of the paramater t
    tn: Double, the upper bound of the paramater t
    y0: Double, the initial value of the solution y
    h: Double, the step size chosen
    k: ndarray of doubles, Extra paramater in f, which is a defined from t0-tn at h/2 stepsize
    Returns
    y: ndarray of doubles, the value of y at evenly spaved points between t0,tn
    """
    hi=np.linspace(t0,tn,1+int((tn-t0)/h))     # defines mesh of t between t0,tn
    if k==[]:                                  # defining dummy variable k when k is not required
        f=lambda y,t,k:f1(y,t)
        k=np.zeros(2*len(hi)-1)
    else:
        f=lambda y,t,k:f1(y,t,k)
    y=np.zeros(len(hi))                        # setting initial value of y to y0
    y[0]=y0                                    # setting initial value of y to y0
    for i in range(len(hi)-1):                 # using 4th order runga kurta to iterate for y[i+1]
        k1=h*f(y[i],hi[i],k[2*i])
        k2=h*f((y[i]+(0.5*k1)),(hi[i]+(h*0.5)),k[(2*i)+1])
        k3=h*f((y[i]+(0.5*k2)),(hi[i]+(h*0.5)),k[(2*i)+1])
        k4=h*f((y[i]+(k3)),(hi[i]+h),k[(2*i)+2])
        y[i+1]=y[i]+(1/6*(k1+(2*k2)+(2*k3)+k4))
    return(y)

def cu_2d_r4(dy,dk,t0,tn,y0,k0,h):
    """
    Integrates pair of initial value problem using 4th order runga kutta method 
    Paramaters:
    dy: Callable, the function to be integrated (defined in y,t,k), will be integrated second
    dk: Callable, the function to be integrated first (defined in y,t)
    t0: Double, the lover bound (initial value) of the paramater t
    tn: Double, the upper bound of the paramater t
    y0: Double, the initial value of the solution y
    k0: Double, the initial value of the solution k
    h: Double, the step size chosen
    Returns
    y: ndarray of doubles, the value of y at evenly spaved points between t0,tn
    Notes:
    look at the written part pdf file for better understanding
    """
    return(cu_r4(dy,t0,tn,y0,h,k=cu_r4(dk,t0,tn,k0,h/2)))

# Solving by converting the second order differntial equation into pair of first order differential equations
# Defining the problem paramaters and 
h=0.01
t0=0
tn=1
y0=0
k0=0
hi=np.linspace(t0,tn,1+int((tn-t0)/h))
y=cu_2d_r4(dy,dk,t0,tn,y0,k0,h) # calling the defined function to solve the initial value problem

# Plotting the solution of the initial value problem
plt.plot(hi,y,'b')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Plot of y(x)")
plt.show()
