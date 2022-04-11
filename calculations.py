# %% codecell
from particles_generator import *
from theoretical_stuff import *
import numpy as np
import math
from scipy import spatial
from scipy import constants
import matplotlib.pyplot as plt
# %% codecell

def get_programmatic(density=None,radius=None,n_particles=None,point=1,eps=0):
    particles = get_particles(n_particles=n_particles,radius=radius)
    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density * vol)/n_particles

    start_point = np.array([[int(radius*point),0,0]])
    if eps == 0:
        r_mul = 1/spatial.distance.cdist(particles,start_point)
    else:
        r_mul = 1/((spatial.distance.cdist(particles,start_point)**2 + eps**2)**(1/2))
    dist = (-1) * constants.G * (mass_particles) * r_mul
    return np.sum(dist)


def get_program_for_particles(density=None,radius=None,n_particles=None,eps=0):
    particles = get_particles(n_particles=n_particles,radius=radius)
    #print("p",particles)
    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density * vol)/n_particles
    rs = []
    phis = []
    for idx,i in enumerate(particles):
        these_particles = np.vstack([particles[idx+1:],particles[:idx]])
        #print(np.linalg.norm(these_particles[0] - i))
        #print(i,these_particles[0])
        #print(spatial.distance.cdist(these_particles,[i]))


        if eps == 0:
            r_mul = spatial.distance.cdist(these_particles,[i])
        else:
            r_mul = (spatial.distance.cdist(these_particles,[i])**2 + eps**2)**(1/2)

        #print("mul",r_mul)

        potentials = (-1) * constants.G * (mass_particles**1)/r_mul

        #print("po",potentials)

        phi = np.sum(potentials)
        r = spatial.distance.cdist([i],[np.zeros(3)])[0][0]
        rs.append(r)
        phis.append(phi)
    return rs,phis

#rs,phis = get_program_for_particles(density=30,radius=10,n_particles=2)
#print(phis)
#print([get_phi(density=30,radius=10,point=i/10) for i in rs])

# %% codecell

def get_diff(n_particles,density,radius,point="radius"):
    theory = get_phi(n_particles,density,radius)
    programmatic = get_programmatic(n_particles=n_particles,density=density,radius=radius,point=point)
    return theory-programmatic,theory,programmatic

# %% codecell

if __name__ == "__main__":


    #print(get_program_for_particles(density=100,radius=10,n_particles=10))






    '''
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
    '''

# %% codecell
