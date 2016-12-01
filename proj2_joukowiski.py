import numpy as np
import matplotlib.pyplot as plt
import math 

def plot_airfoil(e,mu,a):

	b = np.sqrt(a**2 - mu**2) + e
	
	theta = np.arange(0,2*math.pi,0.1)
	
	# Parametrization of the circle
	x = e + a*np.cos(theta)
	y = mu + a*np.sin(theta)
	
	# Joukowski transformation
	X = x*(1+b**2/(x**2+y**2))
	Y = y*(1-b**2/(x**2+y**2))

	fig = plt.plot(X,Y)

	return fig

## Different airfoils plots
plt.figure(1)
plot_airfoil(0,0,1)
plt.axis('equal')
#plt.gca().set_aspect('equal',adjustable='box')

plot_airfoil(0,0.1,1)
plt.axis('equal')

plot_airfoil(-0.1,0,1)
plt.axis('equal')

plot_airfoil(-0.1,0.1,1)
plt.axis('equal')


### Exact solutions of a flow over an airfoil
m=float(50)
n=float(20)
a=float(1)
b=float(3)
alpha = 5*2*math.pi/360 #5 degrees
e = -0.1 # Flat plate
mu = 0.1
B = np.sqrt(a**2 - mu**2) + e
Uinf = 200
dtheta = 2*math.pi/m
dr = (b-a)/n

#Gamma = 4*math.pi*Uinf*a*np.sin(alpha) ## Kutta-Joukowski condition??
Gamma = 30

## Cilindrical grid on the cilinder space

Theta = np.arange(0,2*math.pi,dtheta)
R = np.arange(a,b+dr,dr)
[theta, r] = np.meshgrid(Theta,R)

# Cartesian variables
x = e + r*np.cos(theta)
y = mu + r*np.sin(theta)
# Transformed variables
X = x*(1+B**2/(x**2+y**2))
Y = y*(1-B**2/(x**2+y**2))

# Vr, Vt and Cp in cilindrical coords
Vr = Uinf*(1-B**2/r**2)*np.cos(theta-alpha)
Vt = -Uinf*(1+B**2/r**2)*np.sin(theta-alpha) - Gamma/(2*math.pi*r)

Cp = 1 - (Vr**2+Vt**2)/Uinf**2

# Velocity in cartesian 

U = Vr*np.cos(theta) - Vt*np.sin(theta)
V = Vr*np.sin(theta) + Vt*np.cos(theta)

## Velocity fields in the real domain using Jacobian mapping

dXdx = 1 + B**2/(x**2+y**2) - 2*(x**2)*B**2/(x**2+y**2)**2
dXdy = -2*y*x*B**2/(x**2+y**2)**2
dYdx = 2*x*y*B**2/(x**2+y**2)**2
dYdy = 1 - B**2/(x**2+y**2) + 2*(y**2)*B**2/(x**2+y**2)**2

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

plt.show()


