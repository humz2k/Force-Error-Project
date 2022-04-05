from calculations import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import *
import numpy as np

def get_model_data(density,n_particles,radius,repeats=500):

    expected = potential_model(n_particles,density,radius)

    data = []
    for rep in range(repeats):
        temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=radius)
        data.append(programmatic)

    return expected,np.array(data)

def find_outliers(data,m=1.5):
    mean,std = np.mean(data),np.std(data)
    return abs(data - mean) < abs(std*m)

def get_rms(data):
    squares = (data-expected)**2
    rm = np.sum(squares)/len(squares)
    return np.sqrt(rm)

radiuses = list(range(10,500,10))

with open("std_data.txt","w") as f:

    for n in range(10,51,5):

        total_data = []
        for i in range(2):
            rms = []
            means = []
            for radius in radiuses:
                print(radius)

                expected,data = get_model_data(100,n,radius)

                data = data[find_outliers(data)]

                means.append(np.mean(data))

                rms.append(get_rms(data))

            total_data.append(rms)
            plt.plot(radiuses,rms,zorder=0,alpha=0.2)
            #plt.plot(radiuses,means,label=str(i))

        best = np.zeros_like(np.array(total_data[0]))
        for i in total_data:
            best += np.array(i)
        best = best / len(total_data)

        def objective(x,a):
            return (x**5)/a

        popt, _ = curve_fit(objective,np.array(radiuses),best)
        a, = popt
        b = 5

        f.write(str(n) + "," + str(a) + "," + str(b) + "\n")

        y_new = [(i**5)/a for i in radiuses]

        #plt.plot(radiuses,best,label="avg" + str(n),zorder=1)
        plt.plot(radiuses,y_new,label="model" + str(n),zorder=1)

plt.legend()
plt.show()
