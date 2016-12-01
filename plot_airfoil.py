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

