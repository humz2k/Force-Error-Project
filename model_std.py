from calculations import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_model_data(density,n_particles,radius,repeats=100):

    data = []
    for rep in range(repeats):
        temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=radius)
        data.append(programmatic)

    return np.array(data)

def get_average_std(density,n_particles,radius,repeats=100,average_of=10):
    data = np.array([np.std(get_model_data(density,n_particles,radius,repeats=repeats)) for i in range(average_of)])
    return np.mean(data)

def get_data_density(densities,n_particles,radius):
    return np.array([get_average_std(i,n_particles,radius) for i in densities])

def get_data_radius(density,n_particles,radiuses):
    return np.array([get_average_std(density,n_particles,radius) for radius in radiuses])

densities = np.array(list(range(1,1000,10)))
radiuses = np.array(list(range(1,1000,10)))

print("poo")
plt.plot(densities,get_data_radius(10,10,radiuses),label="Density 10")
print("poo")
plt.plot(densities,get_data_radius(50,10,radiuses),label="Density 50")
print("poo")
plt.plot(densities,get_data_radius(100,10,radiuses),label="Density 100")
#plt.plot(list(range(1,1000,10)),[np.std(get_model_data(i,20,100)) for i in range(1,1000,10)],label="Radius 20")
#plt.plot(list(range(1,1000,10)),[np.std(get_model_data(i,30,100)) for i in range(1,1000,10)],label="Radius 30")
#plt.plot(list(range(1,1000,10)),[np.std(get_model_data(i,40,100)) for i in range(1,1000,10)],label="Radius 40")

plt.legend(loc="upper left")
plt.show()
