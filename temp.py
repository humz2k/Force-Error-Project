import numpy as np
import matplotlib.pyplot as plt

from calculations import *
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

# %% codecell

def plot_r_potential(density=None,n_particles=None,radius=None,file=None,p=None,bins=5,eps=0):

    min = radius
    max = 0

    rs,phis = get_program_for_particles(density=density,radius=radius,n_particles=n_particles,eps=eps)
    cont_r = list(range(floor(np.min(rs)),ceil(np.max(rs))+1))
    if np.min(rs) < min:
        min = np.min(rs)
    if np.max(rs) > max:
        max = np.max(rs)

    plt.scatter(rs,phis,zorder=0,s=0.5,label="(n="+str(n_particles)+")")

    cont_r = list(range(floor(min),ceil(max)+1))
    analytic = [get_phi(density=density,radius=radius,point=r/radius) for r in cont_r]
    plt.plot(cont_r,analytic,zorder=1,color="blue",alpha=0.5,label="theoretical")


    if p != None:
        vol = (4/3) * math.pi * (radius ** 3)
        mass_particles = (density * vol)/n_particles
        radiuses = np.arange(0,radius+1)
        max_r = p(radiuses,radius,1) * radius
        print(max_r)
        print(radiuses)
        pot = ((-1) * constants.G * (mass_particles)/max_r) * (n_particles-1)
        plt.plot(radiuses,pot,label=str(1.))

    plt.xlabel("radius of particle")
    plt.ylabel("potential at particle")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius))
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
    else:
        plt.show()

z = np.array([3,0,0])
f = np.poly1d(z)
def p(start,prob,change=0.001,up=True,f=f):
    integ = f.integ()
    if up:
        b = start+change
        while b < 1 and integ(b) - integ(start) < prob:
            b += change
        if b > 1:
            b = 1.
        b = b
    else:
        b = start - change
        while b > 0 and integ(start) - integ(b) < prob:
            b -= change
    if b < 0:
        return 0.
    return b

def get_max_dist(r,a,prob,func=p):
    ups = func(r/a,prob) - r/a
    downs = -(func(r/a,prob,up=False) - r/a)
    if ups > downs:
        return ups
    return downs

test = np.vectorize(get_max_dist)

plot_r_potential(density=10,n_particles=1000,radius=30,p=test)

'''
a = 30
p = 10
n = 1000
vol = (4/3) * math.pi * (a ** 3)
mass_particles = (p * vol)/n

radiuses = np.arange(0,a+1)

for prob in np.arange(0.2,1.2,0.2):
    max_r = test(radiuses,a,prob)
    pot = ((-1) * constants.G * (mass_particles)/max_r) * (n-1)
    plt.plot(radiuses,pot,label=str(prob))
plt.legend()
plt.show()
'''
'''
test = np.vectorize(p)
radiuses = np.arange(0,1,0.01)
vmin = 0
vmax = 5
norm = plt.Normalize(vmin, vmax)
cm = plt.cm.rainbow

for i in range(vmin,vmax+1):
    ups = test(radiuses,i/vmax) - radiuses
    downs = test(radiuses,i/vmax,up=False) - radiuses
    max_dist = np.maximum(ups,-downs)
    #print((i/vmax))
    #print(ups)
    #print(downs)
    #np.max(ups,downs)
    #print(radiuses)
    #plt.plot(radiuses,ups,color=cm(norm(i)),label=str((i)/vmax))
    #plt.plot(radiuses,-downs,color=cm(norm(i)))
    plt.plot(radiuses,max_dist,label=str(i/vmax))
plt.legend()
plt.show()
'''

#print(p(0.5,0.1),p(0.5,0.1,up=False))
