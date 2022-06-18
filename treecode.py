import numpy as np
import matplotlib.pyplot as plt
import math

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

class Node:
    def __init__(self,particles,vol,pos):
        self.particles = particles
        self.n_particles = len(self.particles)
        self.vol = vol
        self.pos = pos
        self.children = []
        self.parent = None

class Tree:
    def __init__(self,particles):
        self.particles = particles
    
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
        return self.make_node(self.particles,box)
    
    def particle_in_box(self,particle,box):
        x = particle[0]
        y = particle[1]
        z = particle[2]
        if not (x > box[0][0] and x <= box[0][1]):
            return False
        if not (y > box[1][0] and y <= box[1][1]):
            return False
        if not (z > box[2][0] and x <= box[2][1]):
            return False
        return True

    def make_node(self,particles,box):
        parts = []
        for i in particles:
            if self.particle_in_box(i,box):
                parts.append(i)
        vol = abs(box[0][1] - box[0][0]) * abs(box[1][1] - box[1][0]) * abs(box[2][1] - box[2][0])
        pos = np.array([box[0][1] - box[0][0],box[1][1] - box[1][0],box[2][1] - box[2][0]])
        node = Node(parts,vol,pos)
        if len(parts) > 1:
            for subbox in self.divide_box(box):
                next_node = self.make_node(parts,subbox)
                node.children.append(next_node)
                next_node.parent = node
        return node

parts = ParticleGenerator.Uniform(100,100)
myTree = Tree(parts)
tree = myTree.build_tree()