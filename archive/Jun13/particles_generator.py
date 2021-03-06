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


def get_particles(radius=None,n_particles=None):
    particles = []

    for i in range(n_particles):
        temp = get_particle(radius)
        particles.append(temp)

    return np.array(particles)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 1000000
    radius = 1000
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

    plt.scatter(np.array(bins)/radius,100*np.array(ys)/n,s=0.5)

    z = np.polyfit(np.array(bins)/radius,np.array(ys)/n,2)
    print(type(z))
    print(z)
    z = np.array([3,0,0])
    p = np.poly1d(z)
    xs = np.array(bins)/radius
    ys = p(xs)

    print(p)

    plt.plot(xs,ys)
    #plt.yscale('log')
    plt.show()
