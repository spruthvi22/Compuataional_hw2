"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving second order differential equation using euler method

"""

import numpy as np
import matplotlib.pyplot as plt

# Defining the second order differnetial equation as a pair of first order differential equations
# as y'=k, k'=2k/t - 2y/t^2 + t*ln(t)
def f(y,t):                  
    return(np.array([y[1],(2*y[1]/t)-(2*y[0]/(t**2))+(np.log(t)*t)]))

# Defining analytical solution of y
def y_act(t):       
    return((7*t/4)+((0.5*t**3)*np.log(t))-(3/4*t**3))


def cu_euler(f,t0,tn,y0,h):
    """
    Integrates the pair of initial value problems using euler method
    Paramaters:
    f: Callable, the function to be integrated, f should return n sized array, where n is number of DEs 
    t0: Double, the lover bound (initial value) of the paramater t
    tn: Double, the upper bound of the paramater t
    y0: ndarray, the initial value of the solution y of size n
    h: Double, the step size chosen
    Returns
    y: ndarray of doubles, the values of yi at evenly spaved points between t0,tn
    """
    hi=np.linspace(t0,tn,1+int((tn-t0)/h))
    y=np.zeros([len(y0),len(hi)])
    y[:,0]=y0
    for i in range(len(hi)-1):
        y[:,i+1]=y[:,i]+(h*f(y[:,i],hi[i]))
    return(y)

# Defining problem paramaters
t0=1
tn=2
y0=np.array([1,0])
h=0.001
hi=np.linspace(t0,tn,1+int((tn-t0)/h))
y=cu_euler(f,t0,tn,y0,h)[0,:]     # Evaluating y using fnction defined above

# Plotting the solutions
plt.plot(hi,y,'r',label="Numerical Solution")
plt.plot(hi,y_act(hi),'k',label="Analytical Solution")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Plot of y vs t")
plt.legend()
plt.show()
