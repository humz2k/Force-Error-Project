import numpy as np
import open3d as o3d
import PyCC
import time
from scipy import spatial
from astropy import constants
import treecode

df = PyCC.Distributions.Uniform(r = 1000, p = 100, n = 1000)

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
        self.tree = o3d.geometry.Octree(max_depth = 50)
        self.tree.convert_from_point_cloud(self.pcd, size_expand=0.01)
        
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
        indexes = np.arange(len(evaluate_at))
        self.phi = np.zeros_like(indexes,dtype=float)
        self.acc = np.zeros_like(evaluate_at,dtype=float)
        self.stack = [indexes]
        self.evaluations = [0]
        self.correct_evaluations = []
        self.truncations = 0
        def traverse(node,info):
            #print(node.indices)
            if len(self.stack) == 0:
                return True
            
            n_children = str(node).split("with ")[1].split(" ")[0]
            try:
                n_children = int(n_children)
                if n_children == 1:
                    return False
                if n_children == 0:
                    n_children = 1
            except:
                n_children = 1
            
            self.correct_evaluations.append(n_children)

            indexes = self.stack[-1]

            particles = np.take(evaluate_at,indexes,axis=0)
            if len(node.indices) == 1:
                origin = self.particles[node.indices[0]]
            else:
                origin = info.origin+(info.size/2)
            dists = spatial.distance.cdist(particles,np.reshape(origin,(1,3))).flatten()
            good = dists != 0
            if np.sum(np.logical_not(good)) != 0:
                indexes = indexes[good]
                particles = particles[good]
                dists = dists[good]
            done = False

            if len(node.indices) == 1:
                next = np.array([],dtype=int)
                mass = self.masses[node.indices[0]]
                if len(dists) > 0:
                    if eps == 0:
                        delta_phi = (-1) * constants.G * (mass)/dists
                        muls = (constants.G * ((mass) / (dists**3)))
                        accelerations = (-(particles - origin)) * np.reshape(muls,(1,) + muls.shape).T
                    else:
                        delta_phi = (-1) * constants.G * (mass)/((dists**2+eps**2)**(1/2))
                        muls = (constants.G * mass / (((dists**2+eps**2)**(1/2))**3))
                        accelerations = (-(particles - origin)) * np.reshape(muls,(1,) + muls.shape).T
                    self.phi[indexes] += np.array(delta_phi)
                    self.acc[indexes] += np.array(accelerations)
                done = True
            else:
                check = (((info.size**3)/dists) <= theta)
                next = indexes[np.logical_not(check)]
                finished = indexes[check]
                if len(finished) > 0:
                    self.truncations += len(finished)
                    particles = np.take(evaluate_at,finished,axis=0)
                    dists = dists[check]
                    mass = np.sum(self.masses[node.indices])
                    if len(dists) > 0:
                        if eps == 0:
                            delta_phi = (-1) * constants.G * (mass)/dists
                            muls = (constants.G * ((mass) / (dists**3)))
                            accelerations = (-(particles - origin)) * np.reshape(muls,(1,) + muls.shape).T
                        else:
                            delta_phi = (-1) * constants.G * (mass)/((dists**2+eps**2)**(1/2))
                            muls = (constants.G * mass / (((dists**2+eps**2)**(1/2))**3))
                            accelerations = (-(particles - origin)) * np.reshape(muls,(1,) + muls.shape).T
                        self.phi[finished] += np.array(delta_phi)
                        self.acc[finished] += np.array(accelerations)
                if len(next) == 0:
                    self.evaluations[-1] = self.correct_evaluations[-1] - 1
                    done = True
                else:
                    self.evaluations.append(0)
                    self.stack.append(next)
                    return False

            last = next
            self.evaluations[-1] += 1
            
            while self.evaluations[-1] == self.correct_evaluations[-1]:
                last = self.stack.pop(-1)
                self.evaluations.pop(-1)
                self.correct_evaluations.pop(-1)
                if len(self.evaluations) == 0:
                    break
                self.evaluations[-1] += 1
                
            
            self.stack.append(last)
            self.evaluations.append(0)

            return done
        self.tree.traverse(traverse)
        second = time.perf_counter()
        return self.acc,self.phi,{"time":second-first,"truncations":self.truncations}
    
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

print("NEW")
tree = Tree(points,masses)
print("    BUILD",tree.build_tree())
acc,phi,stats = tree.evaluate(points,0,1)
print("TRUNC",stats["truncations"])
print("    EVAL",stats["time"])

print("OLD")
tree2 = treecode.Tree(points,masses)
first = time.perf_counter()
tree2.build_tree()
second = time.perf_counter()
print("    BUILD",second-first)
first = time.perf_counter()
acc2,phi2,stats = tree2.evaluate(points,0,1)
print("TRUNC",stats["truncations"])
second = time.perf_counter()
print("    EVAL",second-first)
print("")
print(np.mean(np.abs(phi2-phi)))
print(np.max((np.abs(phi2-phi))))
print(np.mean(np.abs((acc2-acc).flatten())))
print(np.max((np.abs((acc2-acc).flatten()))))

print("")
out,stats = PyCC.evaluate(save=False,df=df,eval_type="phi",algo="directsum")
print(np.mean(out.loc[:,"phi"].to_numpy()),np.mean(phi))
print(np.mean((out.loc[:,"phi"].to_numpy()-phi)))
print(np.mean((out.loc[:,"phi"].to_numpy()-phi2)))