"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving initial value problem using euler method and comparing it to actual solution
"""

import numpy as np
import matplotlib.pyplot as plt

def f(y,t):         # Defining the function f(y,t) =y'
    return((y/t)-(y/t)**2)

#using euler
def cu_euler(f,t0,tn,y0,h):
    """
    Integrates the initial value problem using backward euler method
    Paramaters:
    f: Callable, the function to be integrated
    t0: Double, the lover bound (initial value) of the paramater t
    tn: Double, the upper bound of the paramater t
    y0: Double, the initial value of the solution y
    h: Double, the step size chosen
    Returns
    [y,hi]: 2 ndarrays of doubles, the value of y and t at evenly spaved points between t0,tn
    """
    hi=np.linspace(t0,tn,1+int((tn-t0)/h))      # defines mesh of t between t0,tn
    y=np.zeros(len(hi))                       # initializes y array to zeros
    y[0]=y0                                   # setting initial value of y to y0
    for i in range(len(hi)-1):
        y[i+1]=y[i]+(h*f(y[i],hi[i]))         # finding y[i+1] using euler method
    return([y,hi])


# Solving the initial value problem using defined function
[y,hi]=cu_euler(f,1,2,1,0.1)     # Solving for t0=1, tn=1, y0=1, h=0.1
y_s=hi/(1+np.log(hi))            # Evaluating the analytical solution at t=hi 

# Plotting the analytical solution and numerical solution 
plt.subplot(2,1,1,title="Solutions")
plt.plot(hi,y,'r',label="Euler Method")
plt.plot(hi,y_s,'b',label="Analytical Solution")
plt.legend()
plt.xlabel("t")
plt.ylabel("y")

# Plotting the absolute and relative error as subplot
plt.subplot(2,1,2,title="Error")
plt.plot(hi,np.absolute(y_s-y),'r',label="Absolute Error")        # Plotting absolute error =|y_act - y_num|
plt.plot(hi,np.absolute((y_s-y)/y_s),'b',label="Relative Error")  # Plotting relative error =|y_act - y_num|/y_act
plt.legend()
plt.xlabel("t")
plt.ylabel("Relative / Absolute Error")
plt.subplots_adjust(wspace=0.4,hspace=0.5)
plt.show()

