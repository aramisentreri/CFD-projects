import numpy as np
from laplace_iteration2_r import laplace_iteration2_r
from laplace_iteration2_t import laplace_iteration2_t
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

close('all')

n = float(20)
m=float(40)
G=float(50) ##Gamma circulation on the cilinder
## fluid velocity
Uinf=float(50)
a=float(1) ##internal radius
b=float(2) ## external radius
#### generation of the polar mesh
dr = (b-a)/n
dtheta = 2*math.pi/m
Theta= np.arange(0,2*math.pi,dtheta)
R= np.arange(a,b+dr,dr)

[theta,r] = np.meshgrid(Theta,R)
## cut r and theta 
print theta.shape
print r.shape
#r=r[0:n+1][:,0:m]
#theta=theta[0:n+1][:,0:m]


#### Exact solution

U_r = Uinf*(1-1/(r**2))*np.cos(theta)
U_t = -Uinf*(1+1/(r**2))*np.sin(theta)-G/(2*math.pi*r)

#### plotting exact solutions
x=r*np.cos(theta)
y=r*np.sin(theta)

U=U_r*np.cos(theta) - U_t*np.sin(theta)
V= U_r*np.sin(theta) + U_t*np.cos(theta)
plt.figure(1)
plt.quiver(x,y,U,V)
circle1 = plt.Circle((0,0),radius=1,fill=False)
#fig2.add_subplot(circle1)
plt.gcf().gca().add_artist(circle1)
plt.title('Velocity of the exact solution')
plt.grid(True)
#plt.show()

fig2 = plt.figure(2)
Cp = 1-(U**2+V**2)/Uinf**2
plt.contour(x,y,Cp,50)
circle2 = plt.Circle((0,0),radius=1,fill=False)
#fig2.add_subplot(circle1)
plt.gcf().gca().add_artist(circle2)
plt.title('Pressure coefficient of the exact solution')
#plt.axis square
#plt.show()

## Computation of the numerical Approximation

#### Iteration scheme
## we will run the iteration 100 times to obtain the solutions
ru_r = Uinf*r*np.random.rand(n+1,m) ## initial guess is random
ru_t = Uinf*r*np.random.rand(n+1,m)

for k in range(1,200):
	ru_r = laplace_iteration2_r(ru_r,r,U_r,dr,dtheta,n,m)
	ru_t = laplace_iteration2_t(ru_t,r,U_t,dr,dtheta,n,m)


u_r=ru_r/r
u_t=ru_t/r
#### Plotting approx solutions

u=u_r*np.cos(theta) - u_t*np.sin(theta)
v= u_r*np.sin(theta) + u_t*np.cos(theta)
plt.figure(3)
plt.quiver(x,y,u,v)
circle3 = plt.Circle((0,0),radius=1,fill=False)
plt.gcf().gca().add_artist(circle3)
plt.title('Velocity of the numerical solution')
plt.xlabel('x axis') 
plt.ylabel('y axis') 
#plt.axis square 


#### Plotting the difference between the numerical and exact solutions

plt.figure(4)
plt.quiver(x,y,U-u,V-v)
circle4 = plt.Circle((0,0),radius=1,fill=False)
plt.gcf().gca().add_artist(circle4)
plt.title('Difference between the exact and approx solutions')
xlabel('x')
plt.ylabel('y') 
#plt.axis square 


#### Plotting the pressure function 
plt.figure(5)
Cp_approx = 1-(u_r**2 + u_t**2)/Uinf**2
circle5 = plt.Circle((0,0),radius=1,fill=False)
plt.gcf().gca().add_artist(circle5)

# hold on
plt.contour(x,y,Cp_approx,50)
plt.title('Pressure coefficient')
#plt.axis square

## show all the figures
plt.show()
