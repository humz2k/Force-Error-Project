from particles_generator import *
from theoretical_stuff import *
import numpy as np
import math
from scipy import spatial
from scipy import constants
import matplotlib.pyplot as plt

def get_programmatic(n_particles,density):
    radius,particles = get_particles(n_particles,density)
    total = 0
    start_point = np.array([[radius,0,0]])
    #print(particles)
    #print(start_point)
    dist = -constants.G/spatial.distance.cdist(particles,start_point)
    return np.sum(dist)
    '''
    for i in range(particles.shape[0]-1):
        start = np.array([particles[i]])
        rest = particles[i+1:]
        dist = -constants.G/spatial.distance.cdist(rest,start)
        total += np.sum(dist)

        for j in rest:
            dist = np.linalg.norm(start-j)
            total += -(constants.G / dist)
    '''
    #return total

def get_diff(n_particles,density):
    theory = get_phi(n_particles,density)
    programmatic = get_programmatic(n_particles,density)
    return theory-programmatic,theory,programmatic

#print(get_programmatic(10000,100))
diffs = []
theorys = []
programs = []
xs = []
n = 100000
start = 1
repeats = 5
for i in range(start,n,1000):
    print((i-start)/(n-start))
    xs.append(i)
    theorys.append(get_phi(i,0.01))
    diff = 0
    programmatics = 0
    for j in range(repeats):
        temp,theory,programmatic = get_diff(i,0.01)
        programmatics += programmatic
        diff += temp
    programmatics = programmatics/repeats
    programs.append(programmatics)
    diff = abs(diff/repeats)
    diffs.append(diff)

#print(to_plot)
#plt.plot(xs,diffs)
plt.plot(theorys)
plt.plot(programs)
plt.show()
#print(diff,theory,programmatic)
