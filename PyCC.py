import numpy as np
import pandas as pd
from scipy import spatial
from scipy import constants
import time
import treecode

def evaluate(file=None,outfile=None,df=None,evaluate_at = None,algo = "directsum",steps=1,delta=0,save=True,**kwargs):
    if type(df) == type(None):
        a = pd.read_csv(file)
    else:
        a = df
    particles = a.loc[:,["x","y","z"]].to_numpy(dtype=float)
    masses = a.loc[:,"mass"].to_numpy(dtype=float)
    if type(evaluate_at) == type(None):
        evaluate_at = particles
    else:
        evaluate_at = evaluate_at.loc[:,["x","y","z"]].to_numpy(dtype=float)
    first = time.perf_counter()
    stats = {}
    if algo == "directsum":
        phis = DirectSum.phis(evaluate_at,particles,masses,**kwargs)
    if algo == "treecode":
        tree_build_time_1 = time.perf_counter()
        tree = treecode.Tree(particles,masses)
        tree.build_tree()
        tree_build_time_2 = time.perf_counter()
        phis,stats = tree.evaluate_phis(evaluate_at,**kwargs)
        stats["tree_build_time"] = tree_build_time_2-tree_build_time_1
    second = time.perf_counter()
    eval_time = second-first
    stats.update({"eval_time":eval_time})
    phis = pd.DataFrame(np.reshape(phis,(1,)+phis.shape).T,columns=["phi"])
    positions = pd.DataFrame(evaluate_at,columns=["x","y","z"])
    out = pd.concat((positions,phis),axis=1)
    if save:
        if file == None:
            outfile = file.split(".")[0]+"_out"+".csv"
        out.to_csv(outfile,index=False)
    return out,stats

class DirectSum(object):
    @staticmethod
    def dists(pos,particles):
        return spatial.distance.cdist(particles,np.reshape(pos,(1,)+pos.shape))

    @staticmethod
    def phi(pos,particles,masses,eps=0):
        dists = DirectSum.dists(pos,particles).flatten()
        masses = masses[dists != 0]
        dists = dists[dists != 0]
        if eps == 0:
            potentials = (-1) * constants.G * (masses)/dists
        else:
            potentials = (-1) * constants.G * (masses)/((dists**2+eps**2)**(1/2))
        return np.sum(potentials)
    
    @staticmethod
    def phis(positions,particles,masses,eps=0):
        distribution = np.zeros(positions.shape[0],dtype=float)
        for idx,pos in enumerate(positions):
            distribution[idx] = DirectSum.phi(pos,particles,masses,eps)
        return distribution

class Distributions(object):
    @staticmethod
    def Uniform(r,n,p,file=None):
        phi = np.random.uniform(low=0,high=2*np.pi,size=n)
        theta = np.arccos(np.random.uniform(low=-1,high=1,size=n))
        particle_r = r * ((np.random.uniform(low=0,high=1,size=n))**(1/3))
        x = particle_r * np.sin(theta) * np.cos(phi)
        y = particle_r * np.sin(theta) * np.sin(phi)
        z = particle_r * np.cos(theta)
        vol = (4/3) * np.pi * (r ** 3)
        particle_mass = (p * vol)/n
        masses = pd.DataFrame(np.full((1,n),particle_mass).T,columns=["mass"])
        particles = pd.DataFrame(np.column_stack([x,y,z]),columns=["x","y","z"])
        df = pd.concat((particles,masses),axis=1)
        if file != None:
            df.to_csv(file,index=False)
        return df
        
class Analytic(object):
    @staticmethod
    def Uniform(r,p,positions):
        positions = positions.loc[:,["x","y","z"]].to_numpy()
        def phi(r,p,pos):
            pos_r = spatial.distance.cdist(np.array([[0,0,0]]),np.reshape(pos,(1,)+pos.shape)).flatten()[0]
            relative = pos_r/r
            if relative == 1:
                return (-4/3) * np.pi * constants.G * p * (r ** 2)
            elif relative < 1:
                return (-2) * np.pi * constants.G * p * ((r ** 2) - ((1/3) * ((pos_r)**2)))
            else:
                return (-4/3) * np.pi * constants.G * p * ((r ** 3)/(pos_r))
        out = np.zeros((len(positions)),dtype=float)
        for idx,pos in enumerate(positions):
            out[idx] = phi(r,p,pos)
        return out

def angles2vectors(alphas,betas):
    x = np.cos(alphas) * np.cos(betas)
    z = np.sin(alphas) * np.cos(betas)
    y = np.sin(betas)
    return np.column_stack([x,y,z])

def randangles(size=10):
    return np.random.uniform(0,2*np.pi,size=size),np.random.uniform(0,2*np.pi,size=size)

def random_vectors(size=1):
    return angles2vectors(*randangles(size))

def ray(vector,length,nsteps,file=None):
    vector = np.reshape(vector/np.linalg.norm(vector),(1,) + vector.shape)
    rs = np.reshape(np.linspace(0,length,nsteps),(1,nsteps)).T
    points = rs * vector
    df = pd.DataFrame(points,columns=["x","y","z"])
    if file != None:
        df.to_csv(file,index=False)
    return df

def points2radius(points):
    points = points.loc[:,["x","y","z"]].to_numpy()
    return spatial.distance.cdist(np.array([[0,0,0]]),points).flatten()
    