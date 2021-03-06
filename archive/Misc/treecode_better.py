import numpy as np
import open3d as o3d
import PyCC
import time
from scipy import spatial
from astropy import constants
import treecode

class Node:
    def __init__(self,node,info):
        self.node,self.info = node,info
        self.children = []

    def child_points(self):
        return sum([len(child.node.indices) for child in self.children])

class Tree:
    def __init__(self,particles,masses):
        self.particles,self.masses = particles,masses
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.particles)

    def build_tree(self):
        first = time.perf_counter()
        self.tree = o3d.geometry.Octree(max_depth = 50)
        self.tree.convert_from_point_cloud(self.pcd, size_expand=0.01)
        self.stack = []
        def traverse(node,info):
            if len(self.stack) == 0:
                self.base_node = Node(node,info)
                self.stack.append(self.base_node)
            else:
                new_node = Node(node,info)
                self.stack[-1].children.append(new_node)
                self.stack.append(new_node)

            if len(node.indices) == 1:
                self.stack.pop(-1)
                for i in range(len(self.stack))[::-1]:
                    if self.stack[i].child_points() == len(self.stack[i].node.indices):
                        self.stack.pop(i)
                    else:
                        break
                return True
            return False
        self.tree.traverse(traverse)

        second = time.perf_counter()
        return second-first

    def evaluate(self,evaluate_at,eps=0,theta=1):
        #the output array for phis
        out = np.zeros(len(evaluate_at),dtype=float)
        acc = np.zeros((len(evaluate_at),3),dtype=float)

        #indexes of the phis
        indexes = np.arange(len(evaluate_at))

        stack = [self.base_node]
        positions = [indexes]

        truncations = 0
        direct = 0

        G = constants.G.value

        while len(stack) != 0:

            node = stack.pop()
            pos_indexes = positions.pop()

            pos = evaluate_at[pos_indexes]

            if len(node.node.indices) == 1:
                origin = np.reshape(self.particles[node.node.indices[0]],(1,3))
            else:
                origin = np.reshape(node.info.origin + node.info.size/2,(1,3))

            mass = np.sum(self.masses[node.node.indices])
            dists = spatial.distance.cdist(pos,origin).flatten()

            to_change = pos_indexes[dists != 0]
            parts = pos[dists != 0]
            dists = dists[dists != 0]

            if len(node.node.indices) == 1:
                check = np.ones_like(to_change)
                direct += np.sum(check)
            else:
                vol = node.info.size**3
                check = ((vol/dists) <= theta)
                truncations += np.sum(check)

            nexts = to_change[np.logical_not(check)]
            to_change = to_change[check]
            parts = parts[check]
            dists = dists[check]

            if len(dists) > 0:
                if eps == 0:
                    delta_phi = (-1) * G * (mass)/dists
                    muls = (G * ((mass) / (dists**3)))
                    accelerations = (origin - parts) * np.reshape(muls,(1,) + muls.shape).T
                else:
                    delta_phi = (-1) * G * (mass)/((dists**2+eps**2)**(1/2))
                    muls = (G * mass / (((dists**2+eps**2)**(1/2))**3))
                    accelerations = (origin - parts) * np.reshape(muls,(1,) + muls.shape).T
                out[to_change] += delta_phi
                acc[to_change] += accelerations

            if len(nexts) > 0:
                for child in node.children:
                    if len(child.node.indices) > 0:
                        stack.append(child)
                        positions.append(nexts)

        return acc,out,{"truncations":truncations,"directs":direct}
