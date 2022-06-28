import numpy as np
import open3d as o3d
import PyCC
import time
from scipy import spatial
from astropy import constants
import treecode

df = PyCC.Distributions.Uniform(r = 100, p = 100, n = 5000)

points = df.loc[:,["x","y","z"]].to_numpy()
masses = df.loc[:,"mass"].to_numpy()

class Node:
    def __init__(self,particles=None,n_particles=0,masses=None,vol=None,pos=None,n_children=0):
        self.particles = particles
        self.n_particles = n_particles
        self.masses = masses
        self.vol = vol
        self.pos = pos
        self.children = []
        self.n_children = n_children

class Tree:
    def __init__(self,particles,masses):
        self.particles,self.masses = particles,masses
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.particles)
    
    def build_tree(self):
        first = time.perf_counter()
        self.tree = o3d.geometry.Octree(max_depth = 100)
        self.tree.convert_from_point_cloud(self.pcd)
        second = time.perf_counter()
        return second-first
    
    def evaluate_one(self,pos,eps=0,theta=1):
        self.phi = 0
        self.acc = np.zeros(3,dtype=float)
        def traverse(node,info):
            vol = info.size ** 3
            dist = np.linalg.norm(info.origin - pos)
            if len(node.indices) == 1:
                if dist > 0:
                    mass = self.masses[node.indices[0]]
                    particle = self.particles[node.indices[0]]
                    if eps == 0:
                        self.phi += (-1) * constants.G * (mass)/dist
                        self.acc += ((particle - pos) * (constants.G * ((mass) / (dist**3)))).value
                    else:
                        self.phi += (-1) * constants.G * (mass)/((dist**2+eps**2)**(1/2))
                        self.acc += ((particle - pos) * (constants.G * mass / (((dist**2+eps**2)**(1/2))**3))).value
                return True
            if (vol/dist) <= theta:
                if dist > 0:
                    mass = np.sum(np.take(self.masses,node.indices))
                    if eps == 0:
                        self.phi += (-1) * constants.G * (mass)/dist
                        self.acc += ((info.origin - pos) * (constants.G * ((mass) / (dist**3)))).value
                    else:
                        self.phi += (-1) * constants.G * (mass)/((dist**2+eps**2)**(1/2))
                        self.acc += ((info.origin - pos) * (constants.G * mass / (((dist**2+eps**2)**(1/2))**3))).value
                return True
            return False
        self.tree.traverse(traverse)
        return self.acc,self.phi.value
    
    def evaluate(self,evaluate_at,eps=0,theta=1):
        first = time.perf_counter()
        accs = np.zeros_like(evaluate_at,dtype=float)
        phis = np.zeros((len(evaluate_at)),dtype=float)
        for idx,pos in enumerate(evaluate_at):
            acc,phi = self.evaluate_one(pos,eps,theta)
            accs[idx] = acc
            phis[idx] = phi
        second = time.perf_counter()
        return accs,phis,second-first
    
    def convert_tree(self):
        first = time.perf_counter()
        base = Node(n_children=1)
        stack = [base]
        def traverse(node,info):
            if len(node.indices) == 1:
                particles = np.reshape(self.particles[node.indices[0]],(1,3))
                masses = np.array([self.masses[node.indices[0]]])
            else:
                particles = None
                masses = np.sum(np.take(self.masses,node.indices))
            n_children = str(node).split("with ")[1].split(" ")[0]
            try:
                n_children = int(n_children)
            except:
                n_children = 0
            if n_children == 0 and len(node.indices) > 1:
                print("YEET")
            new_node = Node(particles = particles,n_particles=len(node.indices),masses = masses,vol = info.size**3,pos = info.origin,n_children = n_children)
            stack[-1].children.append(new_node)
            stack.append(new_node)
            if len(node.indices) == 1:
                stack.pop(-1)
                while stack[-1].n_children == len(stack[-1].children):
                    stack.pop(-1)
                    if len(stack) == 0:
                        break
                return True
            return False
        self.tree.traverse(traverse)
        second = time.perf_counter()
        self.base_node = base.children[0]
        return second-first
        
    def evaluate_numpy(self,evaluate_at,eps=0,theta=1):
        self.convert_tree()
        #the output array for phis
        out = np.zeros(len(evaluate_at),dtype=float)
        acc = np.zeros((len(evaluate_at),3),dtype=float)

        #indexes of the phis
        indexes = np.arange(len(evaluate_at))

        stack = [self.base_node]
        positions = [indexes]

        truncations = 0
        direct = 0

        while len(stack) != 0:

            node = stack.pop()
            pos_indexes = positions.pop()

            pos = np.take(evaluate_at,pos_indexes,axis=0)

            if node.n_particles == 1:
                direct += len(pos)
                dists = spatial.distance.cdist(pos,node.particles).flatten()

                to_change = pos_indexes[dists != 0]
                parts = np.take(evaluate_at,to_change,axis=0)
                dists = dists[dists != 0]
                if len(dists) > 0:
                    if eps == 0:
                        delta_phi = (-1) * constants.G * (node.masses[0])/dists
                        muls = (constants.G * ((node.masses[0]) / (dists**3)))
                        accelerations = (node.particles - parts) * np.reshape(muls,(1,) + muls.shape).T
                    else:
                        delta_phi = (-1) * constants.G * (node.masses[0])/((dists**2+eps**2)**(1/2))
                        muls = (constants.G * node.masses[0] / (((dists**2+eps**2)**(1/2))**3))
                        accelerations = (node.particles - parts) * np.reshape(muls,(1,) + muls.shape).T
                    out[to_change] += np.array(delta_phi)
                    acc[to_change] += np.array(accelerations)
            else:
                dists = spatial.distance.cdist(pos,np.reshape(node.pos,(1,)+node.pos.shape)).flatten()
                check = ((node.vol/dists) <= theta)
                nexts = pos_indexes[np.logical_not(check)]
                finished = pos_indexes[check]
                if len(finished) != 0:
                    truncations += len(finished)
                    mass = np.sum(node.masses)
                    dists = dists[check]
                    to_change = finished[dists != 0]
                    parts = np.take(evaluate_at,to_change,axis=0)
                    dists = dists[dists != 0]
                    if eps == 0:
                        delta_phi = (-1) * constants.G * (mass)/dists
                        muls = (constants.G * ((mass) / (dists**3)))
                        accelerations = (-(parts - node.pos)) * np.reshape(muls,(1,) + muls.shape).T
                    else:
                        delta_phi = (-1) * constants.G * (mass)/((dists**2+eps**2)**(1/2))
                        muls = (constants.G * mass / (((dists**2+eps**2)**(1/2))**3))
                        accelerations = (-(parts - node.pos)) * np.reshape(muls,(1,) + muls.shape).T
                    out[to_change] += np.array(delta_phi)
                    acc[to_change] += np.array(accelerations)
                if len(nexts) > 0:
                    for child in node.children:
                        if child.n_particles > 0:
                            stack.append(child)
                            positions.append(nexts)
        return acc,out,{"truncations":truncations,"directs":direct}

first = time.perf_counter()
tree = Tree(points,masses)
tree.build_tree()
acc,out,stats = tree.evaluate_numpy(points,theta=1)
second = time.perf_counter()
print("NEW",(second-first))

first = time.perf_counter()
tree2 = treecode.Tree(points,masses)
tree2.build_tree()
acc2,temp_phi,stats = tree2.evaluate(points,theta=1)
second = time.perf_counter()
print("OLD",second-first)

print(np.mean(out - temp_phi))
#print(np.mean(acc2-acc))
