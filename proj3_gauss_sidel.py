import numpy as np
import matplotlib.pyplot as plt
import math
import plot_exact
from plot_airfoil import plot_airfoil
from pylab import *

m = float(30)
n = float(20)
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
def gauss_sidel_iteration(Z,r,theta,dr,dtheta):
	
	Znew = Z ## initial guess	
		
	for j in range(1,int(m-1)):
		for i in range(1,int(n)):
			Znew[i,j] = -((Z[i,j+1] + Z[i,j-1])/((r[i,j]**2)*(dtheta**2)) + Z[i-1,j]*(1/dr**2 - 1/(2*r[i,j]*dr)) + Z[i+1,j]*(1/dr**2 + 1/(2*r[i,j]*dr)))/(-2/dr**2 -2/((r[i,j]**2)*(dtheta**2)))
	
	
	## For j=m-1 and j=0
		for i in range(1,int(n)):	
			Znew[i,m-1] = -((Z[i,0] + Z[i,m-2])/((r[i,m-1]**2)*(dtheta**2)) + Z[i-1,m-1]*(1/dr**2 - 1/(2*r[i,m-1]*dr)) + Z[i+1,m-1]*(1/dr**2 + 1/(2*r[i,m-1]*dr)))/(-2/dr**2 -2/((r[i,m-1]**2)*(dtheta**2)))
			Znew[i,0] = -((Z[i,1] + Z[i,m-1])/((r[i,0]**2)*(dtheta**2)) + Z[i-1,0]*(1/dr**2 - 1/(2*r[i,0]*dr)) + Z[i+1,0]*(1/dr**2 + 1/(2*r[i,0]*dr)))/(-2/dr**2 -2/((r[i,0]**2)*(dtheta**2)))
			
	return Znew

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
for k in range(1,50):
	X = gauss_sidel_iteration(X,r,theta,dr,dtheta)
	Y = gauss_sidel_iteration(Y,r,theta,dr,dtheta)


## Now that we have X and Y we can call the plotting of the exact solutions
	
[fig2, fig3, fig4, fig5] = plot_exact.plot_exact(x,y,X,Y,r,theta,m,n,a,b,B,e,mu,Uinf,alpha)
plt.show()
