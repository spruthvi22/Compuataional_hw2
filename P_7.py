"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving initial value problem using solve_ivp

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

# Solving the first problem
def f1(t,y): return((t*np.exp(3*t))-(2*y))             # Defining the function y'=f(t,y)
y1=inte.solve_ivp(f1,[0,1],[0],dense_output=True)      # Solving using solve_ivp over range [0,1] and y0=0
plt.subplot(2,2,1,title="1")
t1=np.linspace(0,1,101)                                # Creating a mesh of t for plotting
# Plotting the Numerical and Analytical solutions
plt.plot(t1,y1.sol(t1).T,'r',label="Numerical Solution")
def y1_sol(t): return((1/25*np.exp(-2*t))*(1-np.exp(5*t)+(5*t*np.exp(5*t))))
plt.plot(t1,y1_sol(t1),'b',label="Analytical Solution")
plt.legend()

# Solving the second problem
# For this problem plotting is only done till 2.99 as the solution is divergent at t=3 
def f2(t,y): return(1-(t-y)**2)
y2=inte.solve_ivp(f2,[2,3],[1],dense_output=True)
plt.subplot(2,2,2,title="2")
t2=np.linspace(2,3,101)
plt.plot(t2[0:99],y2.sol(t2[0:99]).T,'r',label="Numerical Solution")
def y2_sol(t): return((1-(3*t)+(t**2))/(-3+t))
plt.plot(t2[0:99],y2_sol(t2[0:99]),'b',label="Analytical Solution")
plt.legend()

# Solving the third problem
def f3(t,y): return(1+(y/t))
y3=inte.solve_ivp(f3,[1,2],[2],dense_output=True)
plt.subplot(2,2,3,title="2")
t3=np.linspace(1,2,101)
plt.plot(t3,y3.sol(t3).T,'r',label="Numerical Solution")
def y3_sol(t): return((2*t)+(t*np.log(t)))
plt.plot(t3,y3_sol(t3),'b',label="Analytical Solution")
plt.legend()

# Solving the third Problem
def f4(t,y): return(np.cos(2*t)+np.sin(3*t))
y4=inte.solve_ivp(f4,[0,1],[1],dense_output=True)
plt.subplot(2,2,4,title="2")
t4=np.linspace(0,1,101)
plt.plot(t4,y4.sol(t4).T,'r',label="Numerical Solution")
def y4_sol(t): return((1/6)*(8-(2*np.cos(3*t))+(3*np.sin(2*t))))
plt.plot(t4,y4_sol(t4),'b',label="Analytical Solution")
plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.legend()
plt.show()

""" 
All the solutions match closely with the avalytical solutions
