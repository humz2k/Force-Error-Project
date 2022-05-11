# %% codecell
from particles_generator import *
from theoretical_stuff import *
import numpy as np
import math
from scipy import spatial
from scipy import constants
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
    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density * vol)/n_particles
    rs = []
    phis = []
    spatials = []
    for idx,i in enumerate(particles):
        these_particles = np.vstack([particles[idx+1:],particles[:idx]])
        spatials.append(spatial.distance.cdist(these_particles,[i]))
    average_r = np.mean(np.array(spatials).flatten())
    #print(average_r)
    #print(eps/average_r)
    for i,dist in zip(particles,spatials):
        #these_particles = np.vstack([particles[idx+1:],particles[:idx]])

        if eps == 0:
            r_mul = dist
        else:
            r_mul = (dist**2 + (eps/average_r)**2)**(1/2)

        potentials = (-1) * constants.G * (mass_particles**1)/r_mul

        phi = np.sum(potentials)
        r = spatial.distance.cdist([i],[np.zeros(3)])[0][0]
        rs.append(r)
        phis.append(phi)
    return rs,phis

def get_program_eps(density=None,radius=None,n_particles=None,eps=[0]):
    particles = get_particles(n_particles=n_particles,radius=radius)
    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density * vol)/n_particles
    spatials = []

    for idx,i in enumerate(particles):
        these_particles = np.vstack([particles[idx+1:],particles[:idx]])
        spatials.append(spatial.distance.cdist(these_particles,[i]))

    average_r = (vol/n_particles)**(1/3)
    print("stuff")
    print(vol,n_particles)
    print(vol/n_particles)
    print(average_r)
    separations = []
    for i in spatials:
        separations.append(np.min(i))
    print("ah",np.mean(np.array(separations)))

    #print(average_r)
    #print(eps/average_r)


    out = {}
    for ep in eps:

        print(ep/average_r)

        rs = []
        phis = []
        for i,dist in zip(particles,spatials):
            #these_particles = np.vstack([particles[idx+1:],particles[:idx]])

            if ep == 0:
                r_mul = dist
            else:
                r_mul = (dist**2 + (ep/average_r)**2)**(1/2)

            potentials = (-1) * constants.G * (mass_particles**1)/r_mul

            phi = np.sum(potentials)
            r = spatial.distance.cdist([i],[np.zeros(3)])[0][0]
            rs.append(r)
            phis.append(phi)
        out[ep] = (np.array(rs),np.array(phis))
    #print(out)
    return out

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
