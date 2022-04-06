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
    radius,particles = get_particles(1000,100)
    print(radius)
    x = particles[:,0]
    y = particles[:,1]
    z = particles[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
