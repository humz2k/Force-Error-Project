import numpy as np
import math
from scipy import spatial
from scipy import constants
import matplotlib.pyplot as plt

class Analytic(object):
    class Uniform(object):
        @staticmethod
        def phi(sim_args,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            relative = pos_r/sim_args["distargs"]["r"]
            if relative == 1:
                return (-4/3) * math.pi * constants.G * sim_args["p"] * (sim_args["distargs"]["r"] ** 2)
            elif relative < 1:
                return (-2) * math.pi * constants.G * sim_args["p"] * ((sim_args["distargs"]["r"] ** 2) - ((1/3) * ((pos_r)**2)))
            else:
                return (-4/3) * math.pi * constants.G * sim_args["p"] * ((sim_args["distargs"]["r"] ** 3)/(pos_r))
    
    class NFW(object):
        @staticmethod
        def phi(sim_args,pos,**kwargs):

            Rs = sim_args["distargs"]["r"]/kwargs["c"]

            #max_mass = 4*np.pi*(Rs**3)*((np.log(1+kwargs["c"]))-(kwargs["c"]/(1+kwargs["c"])))
            max_mass = 4*np.pi*(Rs**3)*(np.log((Rs+sim_args["distargs"]["r"])/Rs)+(Rs/(Rs+sim_args["distargs"]["r"]))-1)
            actual_mass = sim_args["p"] * sim_args["vol"]
            p0 = actual_mass/max_mass
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            
            if pos_r == 0:
                return -4*np.pi*p0*(Rs**2) * constants.G
            return -1 * ((4*np.pi*p0*(Rs**3)*constants.G)/pos_r) * np.log(1+(pos_r/Rs))


def args(r,n,p):
    vol = (4/3) * math.pi * (r ** 3)
    particle_mass = (p * vol)/n
    average_separation = (vol/n)**(1/3)
    return {"distargs":{"r":r,"n":n},"p":p,"vol":vol,"particle_mass":particle_mass,"average_separation":average_separation}

class Distribution:
    def __init__(self,particle_generator,analytic,**kwargs):
        self.particle_generator = particle_generator
        self.analytic = analytic
        self.kwargs = kwargs

    def __call__(self,r,n):
        return self.particle_generator(r,n,**self.kwargs)

class ParticleGenerator(object):
    @staticmethod
    def Uniform(r,n):
        phi = np.random.uniform(low=0,high=2*math.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        particle_r = r * ((np.random.uniform(low=0,high=1,size=n))**(1/3))
        x = particle_r * np.sin(theta) * np.cos(phi)
        y = particle_r * np.sin(theta) * np.sin(phi)
        z = particle_r * np.cos(theta)
        return np.column_stack([x,y,z])
    
    def NFW(r,n,c=10,nbins = 10000):
        def massDist(Rmax,Rvir,c):
            Rs = Rvir/c
            max_mass = 4*np.pi*(Rs**3)*((np.log(1+c))-(c/(1+c)))
            return (4*np.pi*(Rs**3)*((np.log((Rs+Rmax)/Rs))+(Rs/(Rs+Rmax))-1))/max_mass
        def sampleRadius(Rvir,c,nBins,nParticles):
            radiuses = np.linspace(0,Rvir,nBins)
            low = massDist(radiuses[:-1],Rvir,c)
            high = massDist(radiuses[1:],Rvir,c)
            dists = high-low
            mul = (1/np.min(dists))
            nRadiuses = np.round(dists * mul).astype(int)
            delta = radiuses[1]
            rSamples = []
            for n,low in zip(nRadiuses,radiuses[:-1]):
                rSamples += list(np.random.random(n)*delta+low)
            rSamples = np.array(rSamples)
            return np.random.choice(rSamples,size=nParticles,replace=False)
        particle_r = sampleRadius(r,c,nbins,n)
        phi = np.random.uniform(low=0,high=2*math.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        x = particle_r * np.sin(theta) * np.cos(phi)
        y = particle_r * np.sin(theta) * np.sin(phi)
        z = particle_r * np.cos(theta)
        return np.column_stack([x,y,z])

class Ray:
    def __init__(self,vector=None,length=None):
        assert length != None or vector != None
        self.has_vector = False
        if isinstance(vector,np.ndarray):
            self.vector = np.reshape(vector/np.linalg.norm(vector),(1,) + vector.shape)
            self.has_vector = True
        self.length = length

    def __call__(self,nsteps=25):
        if self.has_vector:
            rs = np.reshape(np.linspace(0,self.length,nsteps),(1,nsteps)).T
            points = rs * self.vector
            return points
        else:
            return self.rs(nsteps)

    def rs(self,nsteps=25):
        return np.linspace(0,self.length,nsteps)

    def analytic_phis(self,sim,nsteps=25,**kwargs):
        rs = np.reshape(np.linspace(0,self.length,nsteps),(1,nsteps)).T
        points = rs * np.array([[1,0,0]])
        analytic = np.zeros(points.shape[0],dtype=float)
        for idx,pos in enumerate(points):
            analytic[idx] = sim.analytic.phi(pos,**kwargs)
        return analytic

def DefaultRay(length=None):
    return Ray(vector=np.array([1,0,0]),length=length)

class __analytic__:
    def __init__(self,sim_args,distribution):
        self.sim_args = sim_args
        self.distribution = distribution

    def phi(self,pos,**kwargs):
        return self.distribution.analytic.phi(self.sim_args,pos,**self.distribution.kwargs,**kwargs)

class Simulation:
    def __init__(self,sim_args,distribution):
        self.args = sim_args
        self.r = self.args["distargs"]["r"]
        self.n = self.args["distargs"]["n"]
        self.p = sim_args["p"]
        self.vol = sim_args["vol"]
        self.particle_mass = sim_args["particle_mass"]
        self.average_separation = sim_args["average_separation"]
        self.distribution = distribution
        self.particles = self.regen()
        self.analytic = __analytic__(sim_args,distribution)

    def regen(self):
        return self.distribution(**self.args["distargs"])

    def dists(self,pos):
        return spatial.distance.cdist(self.particles,np.reshape(pos,(1,)+pos.shape))

    def phi(self,pos,eps=0):
        dists = self.dists(pos);dists = dists[dists != 0]
        if eps == 0:
            potentials = (-1) * constants.G * (self.particle_mass)/dists
        else:
            potentials = (-1) * constants.G * (self.particle_mass)/((dists**2+eps**2)**(1/2))
        return np.sum(potentials)

    def phis(self,positions,eps=0):
        distribution = np.zeros(positions.shape[0],dtype=float)
        analytic = np.zeros(positions.shape[0],dtype=float)
        for idx,pos in enumerate(positions):
            distribution[idx] = self.phi(pos,eps)
            analytic[idx] = self.analytic.phi(pos)
        return distribution,analytic

def plot_ray(sim,vector,length,nsteps=25,label=False,eps=0):
    my_ray = Ray(vector,length)
    dist,ana = sim.phis(my_ray(nsteps),eps=eps)
    rs = my_ray.rs(nsteps)
    if label:
        plt.scatter(rs,dist,label="Vector"+str(my_ray.vector[0])+"n"+str(sim.r),s=10,zorder=1)
    else:
        plt.scatter(rs,dist,s=10,zorder=1)

def angles2vectors(alphas,betas):
    x = np.cos(alphas) * np.cos(betas)
    z = np.sin(alphas) * np.cos(betas)
    y = np.sin(betas)
    return np.column_stack([x,y,z])

def randangles(size=10):
    return np.random.uniform(0,2*np.pi,size=size),np.random.uniform(0,2*np.pi,size=size)

if __name__ == "__main__":
    sim1 = Simulation(args(r=100,n=100,p=1000),Distribution(ParticleGenerator.Uniform,Analytic.Uniform))
    vecs = angles2vectors(*randangles())
    default_ray = Ray(vector=np.array([1,0,0]),length=200)
    for vec in vecs:
        plot_ray(sim1,vec,200)
    plt.plot(default_ray.rs()/sim1.r,default_ray.analytic_phis(sim1),label="Analytic",zorder=0,color="red")
    plt.xlabel("R")
    plt.ylabel("Phi")
    plt.show()
