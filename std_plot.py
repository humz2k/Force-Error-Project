import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %% codecell

with open("std_data.txt","r") as f:
    data = np.array([[float(i) for i in j.split(",")] for j in f.read().splitlines()])

# %% codecell

def objective(x,a,b):
    return a*(x**2) + b*x

xs = np.array(list(range(0,50)))

densities = []
a_data = []
b_data = []

for density in [100,150,200,250,300,500,700]:
#for density in [500]:

    #print(data[:,1][data[:,0] == density] - 1)

    popt, _ = curve_fit(objective,data[:,1][data[:,0] == density] - 1,data[:,2][data[:,0] == density])
    a,b = popt

    print(density,a,b)

    densities.append(density)
    a_data.append(a)
    b_data.append(b)

    plt.plot(xs,objective(xs,a,b),label="model"+str(density))

    plt.plot(data[:,1][data[:,0] == density] - 1,data[:,2][data[:,0] == density],alpha=0.5,label="data"+str(density))

plt.plot([1,1],[0,1e9])
plt.legend()
plt.show()

# %% codecell

def objective2(x,a):
    return ((1/x)**2)*a

base = 1e9
mul = ((1/densities[0])**2)*base
a = a_data[0]/mul
a *= base
print(a)

y_new = [objective2(x,a) for x in densities]

plt.plot(densities,y_new,label="model")

plt.plot(densities,a_data)

plt.legend()
plt.show()

# %% codecell

base = 1e7
mul = ((1/densities[0])**2)*base
a = b_data[0]/mul
a *= base
print(a)

y_new = [objective2(x,a) for x in densities]

plt.plot(densities,y_new,label="model")

plt.plot(densities,b_data)

plt.legend()
plt.show()
