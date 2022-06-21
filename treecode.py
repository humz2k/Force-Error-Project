import numpy as np
from scipy import spatial
from scipy import constants
import time

class Node:
    def __init__(self,particles,masses,vol,pos):
        self.particles = particles
        self.n_particles = len(self.particles)
        self.masses = masses
        self.vol = vol
        self.pos = pos
        self.children = []
        self.parent = None

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

class Tree:
    def __init__(self,particles,masses):
        self.particles = particles
        self.masses = masses
        self.base_node = None
        self.truncations = 0
        self.full = 0
        self.dist_calculations = 0
    
    def get_box(self):
        max_x = np.max(self.particles[:,0])
        min_x = np.min(self.particles[:,0])
        max_y = np.max(self.particles[:,1])
        min_y = np.min(self.particles[:,1])
        max_z = np.max(self.particles[:,2])
        min_z = np.min(self.particles[:,2])
        x = max([abs(max_x),abs(min_x)])
        y = max([abs(max_y),abs(min_y)])
        z = max([abs(max_z),abs(min_z)])
        size = max([x,y,z])
        return np.array([[-size,size],[-size,size],[-size,size]])
    
    def divide_box(self,box):
        new_boxes = [[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]]
        for i in range(0,4):
            new_boxes[i][0][0] = box[0][0]
            new_boxes[i][0][1] = sum(box[0])/2
        for i in range(4,8):
            new_boxes[i][0][0] = sum(box[0])/2
            new_boxes[i][0][1] = box[0][1]
        for i in range(0,4):
            new_boxes[i*2][1][0] = sum(box[1])/2
            new_boxes[i*2][1][1] = box[1][1]
        for i in range(0,4):
            new_boxes[i*2+1][1][0] = box[1][0]
            new_boxes[i*2+1][1][1] = sum(box[1])/2
        new_boxes[0][2][0] = box[2][0]
        new_boxes[0][2][1] = sum(box[2])/2
        new_boxes[1][2][0] = box[2][0]
        new_boxes[1][2][1] = sum(box[2])/2

        new_boxes[2][2][0] = sum(box[2])/2
        new_boxes[2][2][1] = box[2][1]
        new_boxes[3][2][0] = sum(box[2])/2
        new_boxes[3][2][1] = box[2][1]

        new_boxes[4][2][0] = box[2][0]
        new_boxes[4][2][1] = sum(box[2])/2
        new_boxes[5][2][0] = box[2][0]
        new_boxes[5][2][1] = sum(box[2])/2

        new_boxes[6][2][0] = sum(box[2])/2
        new_boxes[6][2][1] = box[2][1]
        new_boxes[7][2][0] = sum(box[2])/2
        new_boxes[7][2][1] = box[2][1]
        boxes = []
        for i in new_boxes:
            boxes.append(np.array(i))
        return boxes
    
    def build_tree(self):
        box = self.get_box()
        vol = abs(box[0][1] - box[0][0]) * abs(box[1][1] - box[1][0]) * abs(box[2][1] - box[2][0])
        self.base_node,_,_ = self.make_node(self.particles,self.masses,box,vol)
        return self.base_node
    
    def particle_in_box(self,particle,box):
        x = particle[0]
        y = particle[1]
        z = particle[2]
        if (x < box[0][0] or x > box[0][1]):
            return False
        if (y < box[1][0] or y > box[1][1]):
            return False
        if (z < box[2][0] or z > box[2][1]):
            return False
        return True

    def make_node(self,particles,particle_masses,box,vol):
        parts = []
        masses = []
        remaining_parts = []
        remaining_masses = []
        for pos,mass in zip(particles,particle_masses):
            if self.particle_in_box(pos,box):
                parts.append(pos)
                masses.append(mass)
            else:
                remaining_parts.append(pos)
                remaining_masses.append(mass)

        pos = np.array([box[0][1] - box[0][0],box[1][1] - box[1][0],box[2][1] - box[2][0]])

        parts = np.array(parts)
        masses = np.array(masses)
        node = Node(parts,masses,vol,pos)
        remaining_masses = np.array(remaining_masses)
        remaining_parts = np.array(remaining_parts)
        if len(parts) > 1:
            for subbox in self.divide_box(box):
                next_node,parts,masses = self.make_node(parts,masses,subbox,vol/8)
                node.children.append(next_node)
                next_node.parent = node
        return node,remaining_parts,remaining_masses

    def phis(self,evaluate_at,eps=0,theta=1):
        out = np.zeros((len(evaluate_at)),dtype=float)
        self.truncations = 0
        self.full = 0
        for idx,i in enumerate(evaluate_at):
            out[idx] = self.evaluate_phi(self.base_node,i,theta,eps,0,0)
        return out,{"truncations":self.truncations,"direct":self.full}
    
    def evaluate_phis(self,evaluate_at,eps=0,theta=1):
        out = np.zeros(len(evaluate_at),dtype=float)
        delta = np.zeros_like(out)
        stack = [self.base_node]
        indexes = np.arange(len(evaluate_at))
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
                dists = dists[dists != 0]
                if eps == 0:
                    delta_phi = (-1) * constants.G * (node.masses[0])/dists
                else:
                    delta_phi = (-1) * constants.G * (node.masses[0])/((dists**2+eps**2)**(1/2))
                out[to_change] += delta_phi
            else:
                dists = spatial.distance.cdist(pos,np.reshape(node.pos,(1,)+node.pos.shape))
                check = ((node.vol/dists) <= theta).flatten()
                nexts = pos_indexes[np.logical_not(check)]
                finished = pos_indexes[check]
                if len(finished) != 0:
                    truncations += len(finished)
                    mass = np.sum(node.masses)
                    dists = dists[check].flatten()
                    to_change = finished[dists != 0]
                    dists = dists[dists != 0]
                    if eps == 0:
                        delta_phi = (-1) * constants.G * (mass)/dists
                    else:
                        delta_phi = (-1) * constants.G * (mass)/((dists**2+eps**2)**(1/2))
                    out[to_change] += delta_phi
                if len(nexts) > 0:
                    for child in node.children:
                        if child.n_particles > 0:
                            stack.append(child)
                            positions.append(nexts)
        return out,{"truncations":truncations,"direct":direct}