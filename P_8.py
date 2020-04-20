"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving boundary value problems using solve_bvp

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

# Solving for first function
def f1(t,y):                     # Defining second order DE as a pair of first order DE
    return(np.vstack([y[1],-1*np.exp(-2*y[0])]))
def bc1(ya,yb):                  # Defining the boundary conditions as y(a)-ya (or y'(a)-y'a ), similary at y=b
    return(np.array([ya[0],yb[0]-np.log(2)]))
t1=np.linspace(1,2,20)           # Defining mesh of t paramater form a to b
y1_a=np.zeros([2,len(t1)])       # Initializing the value of y to zeros
y1=inte.solve_bvp(f1,bc1,t1,y1_a)  # solving using solve_bvp
t1_p=np.linspace(1,2,200)          # Defining mesh of t for plotting
# Plotting the numerical and actual solutions
plt.subplot(2,2,1,title="1")      
plt.plot(t1_p,y1.sol(t1_p)[0],'r',label="Numerical Solution")
if y1.success == True:
    print(" The solution of function 1 is converged to desired accuracy")
else:
    print(" The solution of function 1 has not converged to desired accuracy")
# Mathematica Was unable to solve for the analytical solution
plt.legend()

# Solving in a similar manner for second function
def f2(t,y):
    return(np.vstack([y[1],((y[1]*np.cos(t))-(y[0]*np.log(y[0])))]))
def bc2(ya,yb):
    return(np.array([ya[0]-1,yb[0]-np.exp(1)]))
t2=np.linspace(0,0.5*np.pi,20)
y2_a=np.ones([2,len(t2)])
y2=inte.solve_bvp(f2,bc2,t2,y2_a)
t2_p=np.linspace(0,0.5*np.pi,200)
plt.subplot(2,2,2,title="2")
plt.plot(t2_p,y2.sol(t2_p)[0],'r',label="Numerical Solution")
if y2.success == True:
    print(" The solution of function 2 is converged to desired accuracy")
else:
    print(" The solution of function 2 has not converged to desired accuracy")
# Mathematica Was unable to solve for the analytical solution
plt.legend()


# Solcing for third function
def f3(t,y):
    return(np.vstack((y[1],(-1/np.cos(t))*((2*(y[1]**3))+((y[0]**2)*y[1])))))
def bc3(ya,yb):
    return(np.array([ya[0]-2**(-1/4),yb[0]-((12**(1/4))/2)]))
t3=np.linspace(np.pi/4,np.pi/3,20)
y3_a=np.zeros([2,len(t3)])
y3=inte.solve_bvp(f3,bc3,t3,y3_a)
t3_p=np.linspace(np.pi/4,np.pi/3,200)
plt.subplot(2,2,3,title="3")
plt.plot(t3_p,y3.sol(t3_p)[0],'r',label="Numerical Solution")
if y3.success == True:
    print(" The solution of function 3 is converged to desired accuracy")
else:
    print(" The solution of function 3 has not converged to desired accuracy")
# Mathematica Was unable to solve for the analytical solution
plt.legend()


# Solving for fourth function
def f4(t,y):
    return(np.vstack([y[1],0.5-0.5*y[1]**2-(y[0]*np.sin(t/2))]))
def bc4(ya,yb):
    return(np.array([ya[0]-2,yb[0]-2]))
t4=np.linspace(0,np.pi,20)
y4_a=np.zeros([2,len(t3)])
y4=inte.solve_bvp(f4,bc4,t4,y4_a)
t4_p=np.linspace(np.pi/4,np.pi/3,200)
plt.subplot(2,2,4,title="4")
plt.plot(t4_p,y4.sol(t4_p)[0],'r',label="Numerical Solution")
if y4.success == True:
    print(" The solution of function 4 is converged to desired accuracy")
else:
    print(" The solution of function 4 has not converged to desired accuracy")
# Mathematica Was unable to solve for the analytical solution
plt.legend()


plt.show()

