import matplotlib.pyplot as plt
import math

def potential_model(n_particles,density,radius,a=1.5e10,b=5):
    #const = a * n_particles
    inverse_density = (1/density)**2
    const1 = a * inverse_density
    const1 *= n_particles

    return -(radius**b)/const1
    #x = radius**(inverse_density*const)
    #return (-x)/5
    #return -radius**(((1/density)**2)*(a * n_particles))/b

def std_model(n_particles,density,radius,a=-0.0137,b=18,c=5):
    #const = a * n_particles
    const1 = math.e**(a*density+b)

    return ((radius**c)/const1)

if __name__ == "__main__":

    density = 500
    radius = 100
    ns = list(range(10,2000))

    model_data = [std_model(10,density,radius) for radius in range(1,1000)]

    plt.plot(range(1,1000),model_data)

    plt.show()
