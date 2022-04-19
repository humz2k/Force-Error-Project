from calculations import *
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

plt.rcParams['figure.figsize'] = [8, 8]

def plot_r_potential(density=None,n_particles=None,radius=None,file=None,eps=0):

    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    min = radius
    max = 0
    for idx,n in enumerate(n_particles):

        rs,phis = get_program_for_particles(density=density,radius=radius,n_particles=n,eps=eps)
        cont_r = list(range(floor(np.min(rs)),ceil(np.max(rs))+1))
        if np.min(rs) < min:
            min = np.min(rs)
        if np.max(rs) > max:
            max = np.max(rs)

        plt.scatter(rs,phis,zorder=0,s=0.5,label="(n="+str(n)+")")

    cont_r = list(range(floor(min),ceil(max)+1))
    analytic = [get_phi(density=density,radius=radius,point=r/radius) for r in cont_r]
    plt.plot(cont_r,analytic,zorder=1,color="blue",alpha=0.5,label="theoretical")

    plt.xlabel("radius of particle")
    plt.ylabel("potential at particle")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius)+",eps=" + str(eps),pad=20)
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
    else:
        plt.show()

def plot_scatter_r_potential(density=None,n_particles=None,radius=None,file=None,eps=0):

    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    for n in n_particles:

        rs,phis = get_program_for_particles(density=density,radius=radius,n_particles=n,eps=eps)

        analytic = np.array([get_phi(density=density,radius=radius,point=r/radius) for r in rs])

        plt.scatter(rs,phis-analytic - 1.,s=2,alpha=0.9,label="(n="+str(n)+")")

    plt.xlabel("radius of particle")
    plt.ylabel("ratio of calculated to theoretical")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius)+",eps=" + str(eps),pad=20)
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
    else:
        plt.show()

def save_scatter_plot(p,n,a,eps):
    file = "plots/scatter_p" + str(p) + "n" + str(n) + "a" + str(a) + "eps" + str(eps).replace(".",",")
    plot_scatter_r_potential(density=p,n_particles=n,radius=a,file=file,eps=eps)

def save_r_potential_plot(p,n,a,eps):
    file = "plots/potential_p" + str(p) + "n" + str(n) + "a" + str(a) + "eps" + str(eps).replace(".",",")
    plot_r_potential(density=p,n_particles=n,radius=a,file=file,eps=eps)

#save_scatter_plot(10,[1000,2000,3000],100,2)

#save_r_potential_plot(500,[100,1000,3000],100,10)
save_r_potential_plot(500,[500,1000,1500],100,10)
