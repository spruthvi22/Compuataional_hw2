"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving initial value problem using adaptive step size with Runge Kutta method

"""

import numpy as np
import matplotlib.pyplot as plt

def f(y,t):                 # Defining function f(y,t)=y'
    return(1/((t**2)+(y**2)))

def r4_1(f,y,hi,h):
    """
    Gives y(i+1) after one step of Runge kutta
    Paramaters:
    f: Callable, the function to be integrated
    y: Double, the value of y(i)
    hi: Double, the value of t(i)
    h: Double, the step size chosen
    Returns
    y: double, the value of y(i+1) 
    """
    k1=h*f(y,hi)
    k2=h*f((y+(0.5*k1)),(hi+(h*0.5)))
    k3=h*f((y+(0.5*k2)),(hi+(h*0.5)))
    k4=h*f((y+(k3)),(hi+h))
    return(y+(1/6*(k1+(2*k2)+(2*k3)+k4)))

def cu_r4_adt(f,t0,tn,y0,h,ab_err=10**-6,sft=1):
    """
    Gives Solction to initial value problem using adaprive step size
    Paramaters:
    f: Callable, the function to be integrated
    t0: Double, the lover bound (initial value) of the paramater t
    tn: Double, the upper bound of the paramater t
    y0: Double, the initial value of the solution y
    h: Double, the step size chosen
    ab_err: Double, the required limit on error, default= e-6
    Returns
    [y,t]: ndarry of double, the values of y, t at mesh points  
    """
    y=[y0]        # Setting y0, t0
    t=[t0]
    i=0
    while(t[i]<tn):                      # Limit till t reaches tn
        y1=r4_1(f,y[i],t[i],2*h)         # Calculating y(t+2h) in single step
        yi=r4_1(f,y[i],t[i],h)           # Calculating y(t+h)
        y2=r4_1(f,yi,(t[i]+h),h)         # Calculating y(t+2h) for y(t+h) in two steps
        err=np.abs(y2-y1)                # Defining error
        hopt=h*sft*(((30*h*ab_err)/(err))**(0.25))      # Finding the optimal h
        # if optimal hopt is larger than current h append the value of y(t+h) to y, as it has better accuracy
        if hopt>h:
            y.append(yi)                 
            t.append(t[i]+h)
            # if hopt is smaller than current h, then recaliculate y(i+1) at hopt and then append it to y  
        else:
            yi=r4_1(f,y[i],t[i],hopt)
            y.append(yi)
            t.append(t[i]+hopt)
        h=hopt                           # Setting h=hopt for next iteration
        i=i+1
    # Changing the size of last step so we dont go past tn (t[n]=tn)
    yn=r4_1(f,y[i-1],t[i-1],tn-t[i-1])   
    t[i]=tn
    y[i]=yn
    return([np.array(y),np.array(t)])


# Defining the problem paramaters and initial h
t0=0
tn=3.5*(10**6)
y0=1
h=0.1
y=cu_r4_adt(f,t0,tn,y0,h,ab_err=10**-4)  # Calling the defined function for absolute error e-4
print("the value of x at t=3.5 e6 = ",y[0][len(y[0])-1]) 
