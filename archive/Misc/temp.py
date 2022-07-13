import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

xs = np.array([100,250,500])
ys = np.array([17508850,2732132,73695])

def objective(x,a,b):
    return a*x + b

popt, _ = curve_fit(objective,xs,np.log(ys))
a,b = popt
print(a,b)

a = -0.013755428548299697
b = 18
ynew = np.e**objective(xs,a,b)

plt.plot(xs,ynew,label="model")
plt.plot(xs,ys)
plt.legend()
plt.show()
