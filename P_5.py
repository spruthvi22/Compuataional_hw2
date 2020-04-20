"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving second order boundary value problem using shooting method

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def dk(k,t):          # Defining k'(k,t) where y'=k
    return(-10)

def dy(y,t,k):        # Defining y'(y,t,k)=k
    return(k)

def y_sol(t):         # Defining the analytical solution fo the problem       
    return((-5*(t**2))+50*t)

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

# Defining the problem paramaters
h=0.01
t0=0
tn=10
y0=0
yn=0
hi=np.linspace(t0,tn,1+int((tn-t0)/h))
n=len(hi)
k0=opt.newton(lambda x:(cu_2d_r4(dy,dk,t0,tn,y0,x,h))[n-1],0)  # Solving y[n]=yn taking k0 as variable using newton raphson
y=cu_2d_r4(dy,dk,t0,tn,y0,k0,h)                 # Solving the solution for at k0 solved above 

# Plotting the solution using newton raphson
plt.plot(hi,y,'y',label="Using Newton Raphson")
plt.xlabel("t")
plt.ylabel("x") 

# Solving using argmin function of numpy
x=np.linspace(40,60,11)                         # Taking k0 in steps of 2 from 40 to 60 
yn_err=np.zeros(len(x))
for i in range(len(x)):
    y_arn=cu_2d_r4(dy,dk,t0,tn,y0,x[i],h)       # Finding the solutions for diferrent k0
    plt.plot(hi,y_arn,'b-')
    yn_n=y_arn[n-1]
    yn_err[i]=np.abs(yn-yn_n)                  

ot=np.argmin(yn_err)                            # Finding the minimum value of y[n]-yn using argmin 
y_ar=cu_2d_r4(dy,dk,t0,tn,y0,x[ot],h)           # Finding the solution corrosponding to min argument

# Plotting the solution using argmin and analytical solution
plt.plot(hi,y_ar,'r',label="Using argmin")
plt.plot(hi,y_sol(hi),'k',label="Analytical Solution")
plt.title("Boundary value problem using Shooting method")
plt.legend()
plt.show()
