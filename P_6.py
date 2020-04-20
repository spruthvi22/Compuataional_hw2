"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Solving second order differential equation using Runga-Kutta method

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
from scipy.sparse import csc_matrix
import numpy.linalg as lin

def f(y,t,dy):
    return(-10.0)


# Defning analytical solution  
def y_sol(t):      
    return((-5*(t**2))+50*t)

# Defining Problem paramaters
h=0.01
t0=0
tn=10
y0=0
yn=0
hi=np.linspace(t0,tn,1+int((tn-t0)/h))   # Creating t mesh in range t0,tn 
n=1+int((tn-t0)/h)

# Defining the relaxation problem as A*x=b
A=np.identity(n-2)                       
I=np.identity(n-1)
A=(-2*A)+I[1:n-1,0:n-2]+I[0:n-2,1:n-1]
A=A/h**2
A=csc_matrix(A,dtype=float)
# Defining b=f(y,t,y')=-10 , * if f is a function of y,t,y' then we have to iterate using jacobi or other methods
b=-10*np.ones(n-2)
b[0]=b[0]-(y0/h**2)
b[n-3]=b[n-3]-(yn/h**2)
# Initializing solution y as zeros
y=np.zeros(n)           
y[1:n-1]=sp.spsolve(A,b)       # Solving using sparce solve
y[0]=y0
y[n-1]=yn
y_sp=y

"""
Solving using Jacobi method
Using Jacobi method beacuse the solutions using other methods will show non symmetric possible sloutions which eventually converge to actual solution 

"""
h=1.0 
t0=0
tn=10
y0=0
yn=0

hi1=np.linspace(t0,tn,1+int((tn-t0)/h))   # Creating t mesh in range t0,tn 
n=1+int((tn-t0)/h)

# Defining the relaxation problem as A*x=b
A=np.identity(n-2)                       
I=np.identity(n-1)
A=(-2*A)+I[1:n-1,0:n-2]+I[0:n-2,1:n-1]
A=A/h**2
# Defining b=f(y,t,y')=-10 , * if f is a function of y,t,y' then we have to iterate using jacobi or other methods
y=np.zeros(n)
y[0]=y0
y[n-1]=yn
x=np.zeros(n-2)
xi=x
plt.subplot(2,1,1,title="Possible solutions using Jacobi for 20 iterations")
for i in range(20):
    dx=(y[2:n-1]-y[0:n-3])/(2*h)   
    b[:]=f(x[:],hi1[1:n-1],dx[:])        # Defining in such a way that it can be used for any function 
    b[0]=b[0]-(y0/h**2)
    b[n-3]=b[n-3]-(yn/h**2)
    for j in range(n-2):
        d=b[j]                 # Taking a variable d which equals b(j)-(A(j,0)*x(0) + A(j,1)*x(1) ...... + A(j,n)x(n)) 
        for k in range(n-2):
            if(j!=k):
                d=d-(A[j][k]*x[k])
        xi[j]=(d/A[j][j])      # finding updated jth component of x using  x(j)i+1 = d/A(j,j)
    x=xi
    y[1:n-1]=x[:]
    plt.plot(hi1,y,'b')        # The solution using Jacobi method slowly converges to the actual slution


plt.plot(hi,y_sp,'y',label="Solving Using scipy.sparce")
plt.plot(hi,y_sol(hi),'k',label="Analytical Solution")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()

"""
Solving using Relaxation method 
"""
y=np.zeros(n)
y[0]=y0
y[n-1]=yn
x=np.zeros(n-2)
w=1.25
plt.subplot(2,1,2,title="Possible solutions using Reaxation method for 20 iterations")
for i in range(20):
    dx=(y[2:n-1]-y[0:n-3])/(2*h)   
    b[:]=f(x[:],hi1[1:n-1],dx[:])        # Defining in such a way that it can be used for any function 
    b[0]=b[0]-(y0/h**2)
    b[n-3]=b[n-3]-(yn/h**2)
    for j in range(n-2):
        d=b[j]
        for k in range(n-2):
            if(j!=k):
                d=d-(A[j][k]*x[k])
        x[j]=(w*(d/A[j][j]))+(x[j]*(1-w))   # Adding a extra relaxation to Gauss Sidel
    y[1:n-1]=x[:]
    plt.plot(hi1,y,'g')        # The solution using Jacobi method slowly converges to the actual slution


plt.plot(hi,y_sp,'y',label="Solving Using scipy.sparce")
plt.plot(hi,y_sol(hi),'k',label="Analytical Solution")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.subplots_adjust(wspace=0.4,hspace=0.5)
plt.show()
