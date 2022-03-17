import math
from scipy import constants

def get_phi(n_particles,density):

    radius = ((3/4)*((n_particles/density)/math.pi))**(1/3)

    return (-4/3) * math.pi * constants.G * density * (radius ** 2)
