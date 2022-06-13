from calculations import *
import numpy as np
import matplotlib.pyplot as plt

z = [3,0,0]
f = np.poly1d(z)
def area(a,b,f):
    integ = f.integ()
    return integ(b) - integ(a)
def get_bins(nbins,f=f,a=0,b=1,step=0.01):
    bin_size = (b-a)/nbins
    bins = []
    temp = a
    for i in range(nbins):
        a = temp
        while True:
            temp += step
            if area(a,temp,f) >= bin_size or temp >= 1:
                break
        bins.append(temp)
    return np.array(bins)

def get_programmatic_average(density=None,radius=None,n_particles=None,eps=0):
    particles = get_particles(n_particles=n_particles,radius=radius)
    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density * vol)/n_particles

    dists = []
    means = []
    for point in np.arange(0,1,0.1):
        start_point = np.array([[int(radius*point),0,0]])
        distances = spatial.distance.cdist(particles,start_point)
        if eps == 0:
            r_mul = 1/distances
        else:
            r_mul = 1/((distances**2 + eps**2)**(1/2))
        dist = (-1) * constants.G * (mass_particles) * r_mul
        dists.append(np.sum(dist))
        means.append(np.mean(distances))
    return dists,means

def plot_calc_average():

    radius = 30
    density = 10

    bins = 10
    n = bins*2
    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density*vol)/n

    rs = get_bins(bins)
    xs = []
    ys = []
    for i in np.arange(0,1,0.1):

        particle_dists = np.abs(np.concatenate((rs-i,rs+i)))
        pots = ((-1) * constants.G * (mass_particles))/(particle_dists*radius)
        xs.append(np.mean(particle_dists))
        print(i,xs[-1])
        print(particle_dists)
        ys.append(np.sum(pots))
    print(xs)
    plt.scatter(xs,ys,zorder=100)

    step = 10
    for n in range(10,41,step):
        xs = []
        ys = []
        for i in range(100):
            y,x = get_programmatic_average(density,radius,n)
            xs += x
            ys += y

        print(len(xs)/10)

        plt.scatter(np.array(xs)/radius,ys,s=0.1,label=str(n),zorder=int(n/step))

    vol = (4/3) * math.pi * (radius ** 3)
    mass_particles = (density * vol)
    dists = np.arange(0.1,2,0.1)
    ys = ((-1) * constants.G * mass_particles)/(dists*radius)

    plt.plot(dists,ys,label="lowest possible")

    plt.legend()
    plt.xlabel("average r")
    plt.ylabel("calculated potential")
    plt.xlim(0,2)
    plt.tight_layout()
    plt.show()


plot_calc_average()

#print(get_bins(10))
