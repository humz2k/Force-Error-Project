import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
with open("model_data_density_1000.txt","r") as f:
    data_1000 = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

with open("model_data_density_500.txt","r") as f:
    data_500 = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

with open("model_data_density_100.txt","r") as f:
    data_100 = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

with open("model_data_density_10.txt","r") as f:
    data_10 = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

ns = [10,100,500,1000]
a_data = [np.mean(data_10[:,1]),np.mean(data_100[:,1]),np.mean(data_500[:,1]),np.mean(data_1000[:,1])]
b_data = [np.mean(data_10[:,2]),np.mean(data_100[:,2]),np.mean(data_500[:,2]),np.mean(data_1000[:,2])]
'''

with open("test_model_data.txt","r") as f:
    data = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

print(data)

plt.plot(data[:,0],data[:,1])
#plt.yscale('symlog')

def objective(x):
    return (1/(x*0.1))*1.5e8

#popt, _ = curve_fit(objective,data[:,0],data[:,1])

#c, = popt
#print(c)

y_new = objective(data[:,0])

plt.plot(data[:,0],y_new)

plt.show()

'''
fig, ((a_plot_10,b_plot_10),(a_plot_100,b_plot_100)) = plt.subplots(2,2)
a_plot_10.plot(data_10[:,0],data_10[:,1])
a_plot_10.set_title("Density 10, A")
a_plot_10.plot(data_10[:,0],[np.mean(data_10[:,1])] * len(data_10[:,0]))
b_plot_10.plot(data_10[:,0],data_10[:,2])
b_plot_10.set_title("Density 10, B")
b_plot_10.plot(data_10[:,0],[np.mean(data_10[:,2])] * len(data_10[:,0]))

a_plot_100.plot(data_100[:,0],data_100[:,1])
a_plot_100.set_title("Density 100, A")
a_plot_100.plot(data_100[:,0],[np.mean(data_100[:,1])] * len(data_100[:,0]))
b_plot_100.plot(data_100[:,0],data_100[:,2])
b_plot_100.set_title("Density 100, B")
b_plot_100.plot(data_100[:,0],[np.mean(data_100[:,2])] * len(data_100[:,0]))

plt.show()
'''
