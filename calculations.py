# %% codecell
from particles_generator import *
from theoretical_stuff import *
import numpy as np
import math
from scipy import spatial
from scipy import constants
import matplotlib.pyplot as plt
# %% codecell

def get_programmatic(density=None,radius=None,n_particles=None,point="radius"):
    radius,particles = get_particles(n_particles=n_particles,density=density,radius=radius)

    mass_particles = (density * (radius ** 3))/n_particles

    #print("MASS",mass_particles**2)
    #print(particles,radius)

    total = 0
    if point == "radius":
        start_point = np.array([[radius,0,0]])
    elif point == "center":
        start_point = np.array([[0,0,0]])
    #print(particles)
    #print(start_point)
    dist = -constants.G * (mass_particles**2)/spatial.distance.cdist(particles,start_point)
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
# %% codecell

def get_diff(n_particles,density,radius,point="radius"):
    theory = get_phi(n_particles,density,radius)
    programmatic = get_programmatic(n_particles=n_particles,density=density,radius=radius,point=point)
    return theory-programmatic,theory,programmatic

# %% codecell

if __name__ == "__main__":
    diffs = []
    theorys = []
    programs = []
    xs = []
    n = 2000
    start = 10
    repeats = 1
    for i in range(start,n,10):
        xs.append(i)
        theorys.append(get_phi(n_particles=i,density=100,radius=100))
        diff = 0
        programmatics = 0
        for j in range(repeats):
            temp,theory,programmatic = get_diff(n_particles=i,density=100,radius=100)
            programmatics += programmatic
            diff += temp
        programmatics = programmatics/repeats
        programs.append(programmatics)
        diff = abs(diff/repeats)
        diffs.append(diff)

    fig, (overlay_plot,diff_plot) = plt.subplots(2,1)

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

# %% codecell
