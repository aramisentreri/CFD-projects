import numpy as np
import matplotlib.pyplot as plt
import math
import plot_exact
from plot_airfoil import plot_airfoil
from pylab import *

m = float(50)
n = float(30)
a = float(1)
b = float(5)
e = -0.1 
mu = 0.1
alpha = 5*2*math.pi/360
Uinf = 200
Gamma = 4*math.pi*Uinf*a*np.sin(alpha)
B = np.sqrt(a**2 - mu**2) + e
dtheta = 2*math.pi/m
dr = (b-a)/n

## The parametrization of the Joukowiski airfoil in the real variables 

Theta = np.arange(0,2*math.pi,dtheta)


xcirc = e + a*np.cos(Theta)
ycirc = mu + a*np.sin(Theta)

Xairfoil = xcirc*(1+B**2/(xcirc**2 + ycirc**2)) 
Yairfoil = ycirc*(1-B**2/(xcirc**2 + ycirc**2))

## Grid in the circle plane

R = np.arange(a,b+dr/2,dr)


[theta, r] = np.meshgrid(Theta, R)


## Grid in cartesian

x = e + r*np.cos(theta)
y = mu + r*np.sin(theta)

## Iteration scheme 
#####################################################################
def invert_polar_laplacian(Zbar,r,theta,dr,dtheta):
	
	Z = np.zeros((n+1,m))
	
	## Dirichlet boundary conditions inner and outter circles
	Z[0,:] = Zbar[0,:]
	Z[n,:] = Zbar[n,:]	
	

	for j in range(1,int(m-1)): ## iteration on the angle without the extremes	
		A = np.zeros((n+1,n+1))
		ff = np.zeros((n+1,1))		
		for i in range(1,int(n)): ## iteration on the radius
			A[i,i+1] = 1/dr**2 + 1/(2*r[i,j]*dr)
			A[i,i] = -2*(1/dr**2 + 1/((dtheta**2)*(r[i,j]**2)))
			A[i,i-1] = 1/(dr**2) - 1/(2*r[i,j]*dr)
			ff[i] = - (Zbar[i,j+1]+Zbar[i,j-1])/((r[i,j]**2)*(dtheta**2))
		A[0,0] = 1
		A[n,n] = 1
		ff[0] = Zbar[0,j]
		ff[n] = Zbar[n,j] 

		## Solve for u
		Z[:,j] = linalg.lstsq(A,ff)[0].reshape(n+1)

	## For j=0
		A0 = np.zeros((n+1,n+1))
		ff0 = np.zeros((n+1,1))	
		for i in range(1,int(n)):
		 	A0[i,i+1] = 1/dr**2 + 1/(2*r[i,0]*dr)
			A0[i,i] = -2*(1/dr**2 + 1/((dtheta**2)*(r[i,0]**2)))
			A0[i,i-1] = 1/dr**2 - 1/(2*r[i,0]*dr)
			ff0[i] = - (Zbar[i,1]+Zbar[i,m-1])/((r[i,0]**2)*(dtheta**2))
		A0[0,0] = 1
		A0[n,n] = 1
		ff0[0] = Zbar[0,0]
		ff0[n] = Zbar[n,0] #make sure Zbar has the right boundary condition at infinity

		## Solve for u
		Z[:,0] = linalg.lstsq(A0,ff0)[0].reshape(n+1)


	## For j=m-1

		Am1 = np.zeros((n+1,n+1))
		ffm1 = np.zeros((n+1,1))		
		for i in range(1,int(n)): ## iteration on the radius
			Am1[i,i+1] = 1/dr**2 + 1/(2*r[i,m-1]*dr)
			Am1[i,i] = -2*(1/dr**2 + 1/((dtheta**2)*(r[i,m-1]**2)))
			Am1[i,i-1] = 1/dr**2 - 1/(2*r[i,m-1]*dr)
			ffm1[i] = - (Zbar[i,0]+Zbar[i,m-2])/((r[i,m-1]**2)*(dtheta**2))
		Am1[0,0] = 1
		Am1[n,n] = 1
		ffm1[0] = Zbar[0,m-1]
		ffm1[n] = Zbar[n,m-1] #make sure Zbar has the right boundary condition at infinity

		## Solve for Z
		Z[:,m-1] = linalg.lstsq(Am1,ffm1)[0].reshape(n+1)

		return Z

###########################################################



Xexact = x*(1+B**2/(x**2+y**2))
Yexact = y*(1-B**2/(x**2+y**2))

X = Xexact #Perfect initial guess
Y = Yexact

## Boundary condition on the circle

for j in range(0,int(m)): 
	X[0,j] = x[0,j]*(1+B**2/(x[0,j]**2 + y[0,j]**2)) 
	Y[0,j] = y[0,j]*(1-B**2/(x[0,j]**2 + y[0,j]**2)) 


## Boundary conditions far away	and initial guess

X[n,:] = x[n,:] # Define as the identity far away
Y[n,:] = y[n,:]

## Iteration scheme
for k in range(1,500):
	X = invert_polar_laplacian(X,r,theta,dr,dtheta)
	Y = invert_polar_laplacian(Y,r,theta,dr,dtheta)


## Now that we have X and Y we can call the plotting of the exact solutions
	
[fig2, fig3, fig4, fig5] = plot_exact.plot_exact(x,y,X,Y,r,theta,m,n,a,b,B,e,mu,Uinf,alpha)
plt.show()
