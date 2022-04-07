import numpy as np
import matplotlib.pyplot as plt

with open("temp.txt","r") as f:
    data = np.array([[float(j) for j in i.split(" ")] for i in f.read().splitlines()])

'''
x = data[:,0][data[:,0] <= 1]
y_data = data[:,1][data[:,0] <= 1]

z = np.polyfit(1-x, y_data, 2)
print(z)
f = np.poly1d(z)
print(f)

def model(x,a=-0.5,b=1,c=1):
    return a*(1-x)**2 + b*(1-x) + c

plt.plot(1-x,y_data,label="data")
plt.plot(x,f(x),label="fit")
#plt.plot(x,model(x),label="model")
plt.legend()
plt.show()
'''
x = data[:,0][data[:,0] >= 1]
y_data = data[:,1][data[:,0] >= 1]

def model(x):
    return 1/x

plt.plot(x,y_data,label="data")
#plt.plot(x,f(x),label="fit")
plt.plot(x,model(x),label="model")
plt.legend()
plt.show()
