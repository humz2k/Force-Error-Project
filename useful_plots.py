# %% codecell

from calculations import *
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

# %% codecell

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
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius))
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
    else:
        plt.show()

'''
p = 30
n = [100,1000,3000]
#n = 100
a = 10
file = "plots/p" + str(p) + "n" + str(n) + "a" + str(a)
plot_r_potential(density=p,n_particles=n,radius=a,eps=0,file=file)
'''


def plot_radius_potential(density=None,n_particles=None,point=1,show_theory=False,step=10,repeats=1,start=10,upper_limit=2000):
    if not (type(density) is list or isinstance(density, np.ndarray)):
        density = [density]

    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    if not (type(point) is list or isinstance(point, np.ndarray)):
        point = [point]
    legend = True
    fig, (overlay_plot) = plt.subplots(1,1)
    for r in point:
        for n in n_particles:
            for p in density:
                programs = []
                xs = range(start,upper_limit,step)
                theorys = []
                for i in xs:
                    if show_theory:
                        theorys.append(get_phi(density=p,radius=i,point=r))
                    programmatics = 0
                    for j in range(repeats):
                        programmatic = get_programmatic(density=p,radius=i,n_particles=n,point=r)
                        programmatics += programmatic
                    programmatics /= repeats
                    programs.append(programmatics)

                label = "("
                if len(n_particles) > 1:
                    label += "n=" + str(n)
                    if len(density) > 1:
                        label += ","
                if len(density) > 1:
                    label += "p=" + str(p)
                    if len(point) > 1:
                        label += ","
                if len(point) > 1:
                    label += "r="+str(r) +"a"
                label += ")"
                if len(n_particles) == 1 and len(density) == 1 and len(point) == 1:
                    label = ""
                    legend = False

                if show_theory:
                    mctslabel = "MCTS" + label
                else:
                    mctslabel = label
                overlay_plot.plot(xs,programs,alpha=0.8, label = mctslabel,zorder=0)
                if show_theory:
                    overlay_plot.plot(xs,theorys,alpha=0.8, label = "THRY" + label,zorder=0)

    if show_theory:
        legend = True
    overlay_plot.set_xlabel('Radius')
    overlay_plot.set_ylabel('Potential')
    if legend:
        overlay_plot.legend(loc ="lower left")
    title = 'varying a,sims=' + str(repeats)
    if len(n_particles) == 1:
        title += ",n=" + str(n_particles[0])
    if len(density) == 1:
        title += ",p=" + str(density[0])
    if len(point) == 1:
        title += ',r=' + str(point[0]) + 'a'
    overlay_plot.set_title(title,fontsize=10, pad=12)

    fig.suptitle('Radius/Potential',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_n_potential(density=None,radius=None,point=1,show_theory=False,repeats=1,step=10,start=10,upper_limit=3000):
    if not (type(density) is list or isinstance(density, np.ndarray)):
        density = [density]

    if not (type(radius) is list or isinstance(radius, np.ndarray)):
        radius = [radius]

    if not (type(point) is list or isinstance(point, np.ndarray)):
        point = [point]

    fig, (overlay_plot) = plt.subplots(1,1)

    for r in point:
        for a in radius:
            for p in density:
                programs = []
                xs = range(start,upper_limit,step)
                if show_theory:
                    theorys = [get_phi(density=p,radius=a,point=r)] * len(xs)
                for n in xs:
                    programmatics = 0
                    for j in range(repeats):
                        programmatic = get_programmatic(density=p,radius=a,n_particles=n,point=r)
                        programmatics += programmatic
                    programmatics /= repeats
                    programs.append(programmatics)

                label = "("
                if len(radius) > 1:
                    label += "a=" + str(a)
                    if len(density) > 1:
                        label += ","
                if len(density) > 1:
                    label += "p=" + str(p)
                    if len(point) > 1:
                        label += ","
                if len(point) > 1:
                    label += "r="+str(r) +"a"
                label += ")"
                if len(radius) == 1 and len(density) == 1 and len(point) == 1:
                    label = ""

                if show_theory:
                    mctslabel = "MCTS" + label
                else:
                    mctslabel = label
                overlay_plot.plot(xs,programs,alpha=0.8, label = mctslabel)
                if show_theory:
                    overlay_plot.plot(xs,theorys,alpha=0.8, label = "THRY" + label)

    overlay_plot.set_xlabel('N Particles')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="lower right")
    title = 'varying n,sims=' + str(repeats)
    if len(radius) == 1:
        title += ",a=" + str(radius[0])
    if len(density) == 1:
        title += ",p=" + str(density[0])
    if len(point) == 1:
        title += ',r=' + str(point[0]) + 'a'
    overlay_plot.set_title(title, pad=12)

    fig.suptitle('N Particles/Potential',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_particles(radius,n_particles,scale=20):
    particles = get_particles(radius=radius,n_particles=n_particles)

    x = particles[:,0]
    y = particles[:,1]
    z = particles[:,2]

    fig = plt.figure()
    plot3d = fig.add_subplot(2,2,1, projection='3d')

    plot3d.scatter(x, y, z, c=z+radius/2)
    plot3d.set_xlabel('x')
    plot3d.set_ylabel('y')
    plot3d.set_zlabel('z')
    plot3d.set_xlim(-radius,radius)
    plot3d.set_ylim(-radius,radius)
    plot3d.set_zlim(-radius,radius)

    plotabove = fig.add_subplot(2,2,2)
    circle = plt.Circle(( 0 , 0 ), radius,fill=False,zorder=0)
    plotabove.add_artist(circle)
    plotabove.set_xlim(-radius,radius)
    plotabove.set_ylim(-radius,radius)
    plotabove.set(adjustable='box', aspect='equal')
    plotabove.scatter(x, y,s = (((z+radius/2)/radius)**2)*scale,c=z+radius/2,zorder=1)
    plotabove.set_xlabel('x')
    plotabove.set_ylabel('y')

    plotside = fig.add_subplot(2,2,3)
    circle = plt.Circle(( 0 , 0 ), radius,fill=False,zorder=0)
    plotside.add_artist(circle)
    plotside.set_xlim(-radius,radius)
    plotside.set_ylim(-radius,radius)
    plotside.set(adjustable='box', aspect='equal')
    plotside.scatter(x, z, c=z+radius/2,s = (((y+radius/2)/radius)**2)*scale,zorder=1)
    plotside.set_xlabel('x')
    plotside.set_ylabel('z')

    plotside2 = fig.add_subplot(2,2,4)
    circle = plt.Circle(( 0 , 0 ), radius,fill=False,zorder=0)
    plotside2.add_artist(circle)
    plotside2.set_xlim(-radius,radius)
    plotside2.set_ylim(-radius,radius)
    plotside2.set(adjustable='box', aspect='equal')
    plotside2.scatter(y, z, c=z+radius/2,s = (((x+radius/2)/radius)**2)*scale,zorder=1)
    plotside2.set_xlabel('x')
    plotside2.set_ylabel('z')

    fig.suptitle("Particles Generated by r = " + str(radius) + " & n = " + str(n_particles))
    fig.tight_layout(w_pad=5)

    plt.show()
