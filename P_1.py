"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving initial value problem using euler backward integration

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time

def f1(y,t):     # Definition of the first function 
    return(-9*y)

def f2(y,t):     # Definition of the second function
    return((-20*(y-t)**2)+2*t)



#using backward euler
def cu_backward_euler(f,t0,tn,y0,h):
    """
    Integrates the initial value problem using backward euler method
    Paramaters:
    f: Callable, the function to be integrated
    t0: Double, the lover bound (initial value) of the paramater t
    tn: Double, the upper bound of the paramater t
    y0: Double, the initial value of the solution y
    h: Double, the step size chosen
    Returns
    y: ndarray of doubles, the value of y at evenly spaved points between t0,tn
    """
    hi=np.linspace(t0,tn,1+int((tn-t0)/h))   # defines mesh of t between t0,tn
    y=np.zeros(len(hi))                    # initializes y array to zeros
    y[0]=y0                                # setting initial value of y to y0
    for i in range(len(hi)-1):
        k=opt.newton(lambda x:(h*f(x,hi[i+1]))-x+y[i],y[i])  # solving the backward integral equation using newton raphson method with y[i] as initial value
        y[i+1]=k                           # updating the y[i+1] form the solution of above
    return(y)



# Defining problem paramaters for first and second function
h=0.01
y01=np.exp(1)  #initial value for function 1
y02=1/3        #initial value for function 2
t0=0
tn=1
# Solving first function using the defined function and measuring the time
strt_time=time.time()
y1=cu_backward_euler(f1,t0,tn,y01,h)
end_time=time.time()
print("Time using Nweton-Raphson method for function 1", end_time-strt_time)

# Solving first function using known solution of iteration equation and measuring time
strt_time=time.time()
hi=np.linspace(t0,tn,1+int((tn-t0)/h))
y=np.zeros(len(hi))
y[0]=y01
for i in range(len(hi)-1):    # Appling euler method using known iteration solution
    y[i+1]=(y[i]/(1-(h*9)))
end_time=time.time()
print("Time using the known iteration solution for function 1",end_time-strt_time)

# Solving the second function using the defined function 
y2=cu_backward_euler(f2,t0,tn,y02,h)

# Plotting solutions to both functions
plt.subplot(1,2,1)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.plot(hi,y1,'r',label="y' = -9y")
plt.plot(hi,y2,'b',label="y' = -20(y-t)^2 + 2t")
plt.title("Plot of y vs x")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Plotting Log(y) vs x as subplot
plt.subplot(1,2,2)
plt.plot(hi,np.log(y1),'r',label="y' = -9y")
plt.plot(hi,np.log(y2),'b',label="y' = -20(y-t)^2 + 2t")
plt.title("Plot of log(y) vs x")
plt.xlabel("X")
plt.ylabel("log(Y)")
plt.legend()
plt.show()
