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
n = 20000
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

fig, ((theory_plot,program_plot),(overlay_plot,diff_plot)) = plt.subplots(2,2)

theory_plot.plot(xs,theorys)
theory_plot.set_xlabel('N Particles')
theory_plot.set_ylabel('Potential')
theory_plot.set_title("Theoretical Gravitational Potential",pad=12)

program_plot.plot(xs,programs)
program_plot.set_xlabel('N Particles')
program_plot.set_ylabel('Potential')
program_plot.set_title('Calculated Gravitational Potential',pad=12)

overlay_plot.plot(xs,programs,alpha=0.4, label = "Calculated")
overlay_plot.plot(xs,theorys,alpha=0.4, label = "Theoretical")
overlay_plot.set_xlabel('N Particles')
overlay_plot.set_ylabel('Potential')
overlay_plot.legend(loc ="upper right")
overlay_plot.set_title('Theoretical/Calculated Overlay',pad=12)

diff_plot.plot(xs,diffs)
diff_plot.set_xlabel('N Particles')
diff_plot.set_ylabel('Delta Potential')
diff_plot.set_title('Theoretical/Calculated Difference',pad=12)

fig.tight_layout()
plt.show()

'''
theory_plot = plt.figure(1)
plt.title('Theoretical Gravitational PE')
plt.plot(xs,theorys)
theory_plot.show()

program_plot = plt.figure(2)
plt.title('Calculated Gravitational PE')
plt.plot(xs,programs)
program_plot.show()

overlay_plot = plt.figure(3)
plt.title('Theoretical/Calculated Overlay')
plt.plot(xs,programs,alpha=0.5)
plt.plot(xs,theorys,alpha=0.5)
overlay_plot.show()

diff_plot = plt.figure(4)
plt.title('Theoretical/Calculated Difference')
plt.plot(xs,diffs)
diff_plot.show()

input()
'''
#print(diff,theory,programmatic)
