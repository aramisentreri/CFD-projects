import numpy as np
import matplotlib.pyplot as plt
import math 
from plot_airfoil import plot_airfoil

def plot_exact(x,y,X,Y,r,theta,m,n,a,b,B,e,mu,Uinf,alpha):
	
	dr = (b-a)/n
	dtheta = 2*math.pi/m
	Gamma = 4*math.pi*Uinf*a*np.sin(alpha) #Kutta condition	
	# Vr, Vt and Cp in cilindrical coords
	Vr = Uinf*(1-B**2/r**2)*np.cos(theta-alpha)
	Vt = -Uinf*(1+B**2/r**2)*np.sin(theta-alpha) - Gamma/	(2*math.pi*r)

	Cp = 1 - (Vr**2+Vt**2)/Uinf**2

	# Velocity in cartesian 	
	
	U = Vr*np.cos(theta) - Vt*np.sin(theta)
	V = Vr*np.sin(theta) + Vt*np.cos(theta)
	
	## Jacobian of the given mapping X,Y
	
	drdx = x/r
	drdy = y/r
	dthetadx = -y/r**2
	dthetady = x/r**2

	dXdr = np.zeros((n+1,m))
	dXdtheta = np.zeros((n+1,m))
	dYdr = np.zeros((n+1,m))
	dYdtheta = np.zeros((n+1,m))

	for j in range(1,int(m-1)):
		for i in range(1,int(n)):
			dXdr[i,j] = (X[i+1,j] - X[i-1,j])/(2*dr)
			dXdtheta[i,j] = (X[i,j+1] - X[i,j-1])/(2*dtheta)
			dYdr[i,j] = (Y[i+1,j] - Y[i-1,j])/(2*dr)
			dYdtheta[i,j] = (Y[i,j+1] - Y[i,j-1])/(2*dtheta)
		dXdr[0,j] = (X[1,j] - X[0,j])/dr
		dXdr[n,j] = (X[n,j] - X[n-1,j])/dr
		dXdtheta[0,j] = (X[0,j+1] - X[0,j-1])/(2*dtheta)
		dXdtheta[n,j] = (X[n,j+1] - X[n,j-1])/(2*dtheta)
		dYdr[0,j] = (Y[1,j] - Y[0,j])/dr
		dYdr[n,j] = (Y[n,j] - Y[n-1,j])/dr
		dYdtheta[0,j] = (Y[0,j+1] - Y[0,j-1])/(2*dtheta)
		dYdtheta[n,j] = (Y[n,j+1] - Y[n,j-1])/(2*dtheta)

	## For j=0
	for i in range(1,int(n)):
		dXdr[i,0] = (X[i+1,0] - X[i-1,0])/(2*dr)
		dXdtheta[i,0] = (X[i,1] - X[i,m-1])/(2*dtheta)
		dYdr[i,0] = (Y[i+1,0] - Y[i-1,0])/(2*dr)
		dYdtheta[i,0] = (Y[i,1] - Y[i,m-1])/(2*dtheta)
	dXdr[0,0] = (X[1,0] - X[0,0])/dr
	dXdr[n,0] = (X[n,0] - X[n-1,0])/dr
	dXdtheta[0,0] = (X[0,1] - X[0,m-1])/(2*dtheta)
	dXdtheta[n,0] = (X[n,1] - X[n,m-1])/(2*dtheta)
	dYdr[0,0] = (Y[1,0] - Y[0,0])/dr
	dYdr[n,0] = (Y[n,0] - Y[n-1,0])/dr
	dYdtheta[0,0] = (Y[0,1] - Y[0,m-1])/(2*dtheta)
	dYdtheta[n,0] = (Y[n,1] - Y[n,m-1])/(2*dtheta)
	

	## For j=m-1
	for i in range(1,int(n)):
		dXdr[i,m-1] = (X[i+1,m-1] - X[i-1,m-1])/(2*dr)
		dXdtheta[i,m-1] = (X[i,0] - X[i,m-2])/(2*dtheta)
		dYdr[i,m-1] = (Y[i+1,m-1] - Y[i-1,m-1])/(2*dr)
		dYdtheta[i,m-1] = (Y[i,0] - Y[i,m-2])/(2*dtheta)
	dXdr[0,m-1] = (X[1,m-1] - X[0,m-1])/dr
	dXdr[n,m-1] = (X[n,m-1] - X[n-1,m-1])/dr
	dXdtheta[0,m-1] = (X[0,0] - X[0,m-2])/(2*dtheta)
	dXdtheta[n,m-1] = (X[n,0] - X[n,m-2])/(2*dtheta)
	dYdr[0,m-1] = (Y[1,m-1] - Y[0,m-1])/dr
	dYdr[n,m-1] = (Y[n,m-1] - Y[n-1,m-1])/dr
	dYdtheta[0,m-1] = (Y[0,0] - Y[0,m-2])/(2*dtheta)
	dYdtheta[n,m-1] = (Y[n,0] - Y[n,m-2])/(2*dtheta)

	## Now the Jacobian in cartesian 
	
	dXdx = dXdr*drdx + dXdtheta*dthetadx
	dXdy = dXdr*drdy + dXdtheta*dthetady
	dYdx = dYdr*drdx + dYdtheta*dthetadx
	dYdy = dYdr*drdy + dYdtheta*dthetady
	
	## Velocity fields in the real domain using Jacobian mapping

	Ureal = U*dXdx + V*dXdy
	Vreal = U*dYdx + V*dYdy
	
	## Plotting in the real plane
	
	fig2 = plt.figure(2)
	plt.contour(X,Y,Cp,24)
	plot_airfoil(e,mu,a)
	plt.title('Pressure coefficient')
	plt.axis('equal')
	
	fig3 = plt.figure(3)
	plt.quiver(X,Y,Ureal,Vreal)
	plot_airfoil(e,mu,a)
	plt.title('Velocity field')
	plt.axis('equal')
	
	## Plotting the grids

	fig4 = plt.figure(4)
	plt.plot(x,y,'b.')
	circle3 = plt.Circle((e,mu),radius=a,fill=False)
	plt.gcf().gca().add_artist(circle3)
	plt.title('Cylinder plane grid')
	plt.plot(x[1,:],y[1,:],'r.')
	plt.plot(x[2,:],y[2,:],'g.')

	fig5 = plt.figure(5)
	plt.plot(X,Y,'b.')
	plot_airfoil(e,mu,a)
	plt.title('Real plane grid')
	plt.plot(X[1,:],Y[1,:],'r.')
	plt.plot(X[2,:],Y[2,:],'g.')	
	
	return [fig2, fig3, fig4, fig5]	
	
