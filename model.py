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
#a=2330964841,b=105760531823
def rms_model(n_particles,density,radius,a=2.5e9,b=1e11,c=5):

    #const = a * n_particles
    inverse_density = (1/density)**2
    const1 = inverse_density * a
    const2 = inverse_density * b

    const3 = const1*((n_particles-1)**2) + const2*(n_particles-1)

    return ((radius**c)/const3)

if __name__ == "__main__":

    density = 500
    #radius = 100
    ns = list(range(10,2000))

    model_data = [rms_model(10,density,radius) for radius in range(1,500)]

    plt.plot(range(1,500),model_data)

    plt.show()
