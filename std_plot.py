import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

with open("std_data.txt","r") as f:
    data = np.array([[float(i) for i in j.split(",")] for j in f.read().splitlines()])

def objective(x,a,b):
    return (x*a) + b

popt, _ = curve_fit(objective,data[:,0],data[:,1])
a,b = popt

print(a,b)

plt.plot(data[:,0],objective(data[:,0],a,b))

plt.plot(data[:,0],data[:,1])

plt.show()
