import numpy
from matplotlib import pyplot
##import time, sys

nx = 41
deltax = 2.0/(nx-1)
nt = 25
dt = .025
c = 1

u = numpy.ones(nx)
u[.5/deltax : (1/deltax)+1] = 2

pyplot.plot(numpy.linspace(0,2,nx),u);
pyplot.show(1)

un = numpy.ones(nx) #initialize a temporary array

for n in range(nt):  #loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() ##copy the existing values of u into un
    for i in range(1,nx): ## you can try commenting this line and...
    #for i in range(nx): ## ... uncommenting this line and see what happens!
        u[i] = un[i]-c*(dt/deltax)*(un[i]-un[i-1])
    pyplot.hold(True)
    pyplot.plot(numpy.linspace(0,2,nx),u);
pyplot.show()
