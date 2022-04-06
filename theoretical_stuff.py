import math
from scipy import constants

def get_phi(density=None,radius=None,point=1):
    '''
    if radius == None:
        radius = ((3/4)*((n_particles/density)/math.pi))**(1/3)
    '''
    if point == 1:
        return (-4/3) * math.pi * constants.G * density * (radius ** 2)
    elif point < 1:
        return (-2) * math.pi * constants.G * density * ((radius ** 2) - ((1/3) * ((point*radius)**2)))
    else:
        return (-4/3) * math.pi * constants.G * density * ((radius ** 3)/(point*radius))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    density = 100
    for radius in range(1,100,10):
        ys = [get_phi(density=density,radius=radius,point=i/100) for i in range(300)]
        print(ys)
        xs = [(i/100) for i in range(300)]
        plt.plot(xs,ys,label=str(density))
    plt.legend()
    plt.show()
