import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from math import exp

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

expo = 2

asss = []

x = np.array(list(range(10,8000)))

def y_func(x,a):
    return ((1/x)**expo)*a

#((1/x)**expo)*a = a_actual

with open("test_model_data_20_10.txt","r") as f:
    data = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

plt.plot(data[:,0],data[:,1],label="20")

a = data[:,1][0]/((1/10)**expo)

asss.append(a)

y = y_func(x,a)

plt.plot(x,y,label="1/x",alpha=0.5)

with open("test_model_data_10_10.txt","r") as f:
    data = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

plt.plot(data[:,0],data[:,1],label="10")

a = data[:,1][0]/((1/10)**expo)

asss.append(a)

y = y_func(x,a)

plt.plot(x,y,label="1/x",alpha=0.5)

with open("test_model_data_50_10.txt","r") as f:
    data = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

plt.plot(data[:,0],data[:,1],label="50")

a = data[:,1][0]/((1/10)**expo)

asss.append(a)

y = y_func(x,a)

plt.plot(x,y,label="1/x",alpha=0.5)

with open("test_model_data_100_10.txt","r") as f:
    data = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

plt.plot(data[:,0],data[:,1],label="100")

a = data[:,1][0]/((1/10)**expo)

asss.append(a)

y = y_func(x,a)

plt.plot(x,y,label="1/x",alpha=0.5)

print(np.mean(np.array(asss)))

#plt.semilogy(base=10)


#popt, _ = curve_fit(y_func,data[:,0],data[:,1])

#a = popt

#print(data[:,0][0],data[:,1][0])

#print()

#plt.semilogy(base=10)

#def objective(x,a,b,c,d):
#    return a*x**3+b*x**2+c*x+d

#popt, _ = curve_fit(objective,data[:,0],np.log10(data[:,1]))

#a,b,c,d = popt

#print(b,c,d)

#y_new = objective(np.array(list(range(0,700))),a,b,c,d)

#plt.plot(list(range(0,700)),y_new,alpha=0.5)
plt.legend(loc = "upper right")
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
