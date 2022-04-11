import random
import numpy as np
import math


def get_particle(radius):
    phi = random.uniform(0,2*math.pi)
    cos_theta = random.uniform(-1,1)
    u = random.uniform(0,1)
    theta = math.acos(cos_theta)
    r = radius * (u**(1/3))
    x = r * math.sin( theta) * math.cos( phi )
    y = r * math.sin( theta) * math.sin( phi )
    z = r * math.cos( theta )

    return np.array([x,y,z])

    '''
    angle_x = math.radians(random.randint(0,360))
    angle_z = math.radians(random.randint(0,360))
    angle_y = math.radians(random.randint(0,360))
    r = random.randint(0,radius)
    pos = np.array([r,0,0]) * Ry(angle_y) * Rx(angle_x) * Rz(angle_z)
    return np.asarray(pos).flatten()
    '''

def get_particles(radius=None,n_particles=None):
    '''
    if radius == None:
        radius = ((3/4)*((n_particles/density)/math.pi))**(1/3)
    '''
    particles = []

    for i in range(n_particles):
        temp = get_particle(radius)
        particles.append(temp)

    return np.array(particles)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    

    '''
    for r in range(100,1000,100):
        n = 100000
        radius = r
        particles = get_particles(radius=radius,n_particles=n)

        bin_size = radius/100
        bins = list(range(0,radius,math.ceil(bin_size)))
        #print(bins)

        rs = np.linalg.norm(particles,axis=1)
        xs = [0] * len(rs)

        ys = []
        for i in bins:
            j = rs[rs >= i]
            j = j[j < i+bin_size]
            ys.append(len(j))

        plt.scatter(np.array(bins)/radius,ys,s=0.5)

    xs = np.array(bins)/radius
    ys = (xs*14.4)**3

    plt.plot(xs,ys)

    plt.show()
    '''
