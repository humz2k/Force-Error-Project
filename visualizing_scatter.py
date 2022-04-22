from calculations import *
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

plt.rcParams['figure.figsize'] = [8, 8]

def plot_r_potential(density=None,n_particles=None,radius=None,file=None,eps=0):
    if not (type(eps) is list or isinstance(eps, np.ndarray)):
        eps = [eps]
    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    min = radius
    max = 0
    for ep in eps:
        for idx,n in enumerate(n_particles):

            rs,phis = get_program_for_particles(density=density,radius=radius,n_particles=n,eps=ep)
            cont_r = list(range(floor(np.min(rs)),ceil(np.max(rs))+1))
            if np.min(rs) < min:
                min = np.min(rs)
            if np.max(rs) > max:
                max = np.max(rs)

            plt.scatter(rs,phis,zorder=0,s=0.5,label="(n="+str(n)+",eps="+str(ep)+")")

    cont_r = list(range(floor(min),ceil(max)+1))
    analytic = [get_phi(density=density,radius=radius,point=r/radius) for r in cont_r]
    plt.plot(cont_r,analytic,zorder=1,color="blue",alpha=0.5,label="theoretical")

    plt.xlabel("radius of particle")
    plt.ylabel("potential at particle")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius)+",eps=" + str(eps),pad=20)
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

def plot_scatter_r_potential(density=None,n_particles=None,radius=None,file=None,eps=0):
    if not (type(eps) is list or isinstance(eps, np.ndarray)):
        eps = [eps]

    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    for ep in eps:
        for n in n_particles:

            rs,phis = get_program_for_particles(density=density,radius=radius,n_particles=n,eps=ep)

            analytic = np.array([get_phi(density=density,radius=radius,point=r/radius) for r in rs])

            plt.scatter(rs,phis-analytic - 1.,s=2,alpha=0.9,label="(n="+str(n)+",eps="+str(ep)+")")

    plt.xlabel("radius of particle")
    plt.ylabel("ratio of calculated to theoretical")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius)+",eps=" + str(eps),pad=20)
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

def plot_scatter_pdf(density=None,n_particles=None,radius=None,file=None,eps=0,runs=1):
    if not (type(eps) is list or isinstance(eps, np.ndarray)):
        eps = [eps]

    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    for idx,ep in enumerate(eps):
        for n in n_particles:

            phis = []
            rs = []
            for run in range(runs):
                temp_rs,temp_phis = get_program_for_particles(density=density,radius=radius,n_particles=n,eps=ep)
                phis += temp_phis
                rs += temp_rs

            print(len(phis))
            phis = np.array(phis)
            rs = np.array(rs)

            analytic = np.array([get_phi(density=density,radius=radius,point=r/radius) for r in rs])

            width = np.max(phis) - np.min(phis)
            bin_size = 0.07/120

            plt.hist(phis-analytic - 1.,bins=int(width/bin_size),alpha=0.5,label="(n="+str(n)+",eps="+str(ep)+")",zorder=len(eps)-idx)

            #plt.scatter(rs,phis-analytic - 1.,s=2,alpha=0.9,label="(n="+str(n)+",eps="+str(ep)+")")

    plt.xlabel("calculated-theoretical -1.")
    #plt.ylabel("ratio of calculated to theoretical")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius)+",eps=" + str(eps),pad=20)
    plt.legend(prop={'size': 6})
    if file != None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

def save_scatter_plot(p,n,a,eps):
    file = "plots/scatter_p" + str(p) + "n" + str(n) + "a" + str(a) + "eps" + str(eps).replace(".",",")
    plot_scatter_r_potential(density=p,n_particles=n,radius=a,file=file,eps=eps)

def save_r_potential_plot(p,n,a,eps):
    file = "plots/potential_p" + str(p) + "n" + str(n) + "a" + str(a) + "eps" + str(eps).replace(".",",")
    plot_r_potential(density=p,n_particles=n,radius=a,file=file,eps=eps)

#save_scatter_plot(10,[1000],100,eps=[0,10,50])
#save_scatter_plot(20,[2000],100,eps=[0,10,50])
#save_scatter_plot(30,[100,300,500],100,eps=[10])

#save_r_potential_plot(500,[1500],1000,[0,500])
#save_r_potential_plot(1000,[500,1000,1500],100,[0])
#save_r_potential_plot(30,[10],100,[0,10,100])
#plot_scatter_pdf(density=500,n_particles=[5000],radius=1000,eps=[0,500],runs=1)

n_particles = 5000
density = 500
radius = 100
eps = [0,500,1000,2000,5000,20000]
nsims = 5
nbins = 300
outs = []
min = np.inf
max = -np.inf
for i in range(nsims):
    outs.append(get_program_eps(density=density,n_particles=n_particles,radius=radius,eps=eps))
    for ep in eps:
        rs,phis = outs[-1][ep]
        analytic = np.array([get_phi(density=density,radius=radius,point=r/radius) for r in rs])
        data = phis-analytic -1.
        if np.min(data) < min:
            min = np.min(data)
        if np.max(data) > max:
            max = np.max(data)
print(min,max)
#print([outs[i] for i in outs.keys()])
#print(outs[0])
#exit(1)
for ep in outs[0].keys():
    rs = []
    phis = []
    for out in outs:
        temp_rs,temp_phis = out[ep]
        rs += temp_rs.tolist()
        phis += temp_phis.tolist()
    phis = np.array(phis)
    print(len(phis))
    rs = np.array(rs)
    width = np.max(phis) - np.min(phis)
    #bin_size = 0.07/((5000*120)/n_particles)
    analytic = np.array([get_phi(density=density,radius=radius,point=r/radius) for r in rs])
    plt.hist(phis-analytic - 1.,bins=nbins,range=(min,max),alpha=0.8,label=str(ep),zorder=ep,histtype='step')

plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius)+",eps=" + str(eps) + ",nsims=" + str(nsims),pad=20)
plt.xlabel("calculated - theoretical - 1.")
plt.ylabel("particle density")
plt.legend()
plt.show()
