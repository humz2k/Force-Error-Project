import PyCC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_errors(thetas,r,n,p,repeats = 2):
    outs = np.zeros((len(thetas),6*repeats),dtype=float)
    for repeat in range(repeats):
        for idx,theta in enumerate(thetas):
            distribution = PyCC.Distributions.Uniform(r=r,n=n,p=p)
            dist = r/np.tan(theta/2)
            pos = pd.DataFrame(np.array([[dist,0,0],[-dist,0,0],[0,dist,0],[0,-dist,0],[0,0,dist],[0,0,-dist]]),columns=["x","y","z"])
            direct,stats = PyCC.evaluate(df=distribution,evaluate_at=pos,save=False,algo="directsum")
            direct = direct.loc[:,"phi"].to_numpy()
            tree,stats = PyCC.evaluate(df=distribution,evaluate_at=pos,save=False,algo="treecode",theta=1)
            tree = tree.loc[:,"phi"].to_numpy()
            frac = np.abs(direct-tree)/np.abs(tree)
            offset = repeat * 6
            for i in range(6):
                outs[idx][i+offset] = frac[i]
    return outs

thetas = [1,0.66,1/2]
r = 1
p = 100
ns = [2,8,32,128,512,2048,8192,32768]
colors = ["red","green","blue"]
means = np.zeros((len(ns),len(thetas)),dtype=float)
medians = np.zeros((len(ns),len(thetas)),dtype=float)
for idx,n in enumerate(ns):
    print(n)
    out = get_errors(thetas=thetas,r=r,n=n,p=p,repeats=10)
    means[idx] = np.mean(out,axis=1)
    medians[idx] = np.median(out,axis=1)

for i in range(3):
    plt.scatter([0],[0],c=colors[i],label=thetas[i])
for idx,theta in enumerate(thetas):
    plt.scatter(ns,means[:,idx],c=colors[idx],marker="x")
for idx,theta in enumerate(thetas):
    plt.scatter(ns,medians[:,idx],c=colors[idx],marker="2")
plt.xscale('log')
plt.yscale('log')
plt.yticks([.03,.01,.003,.001,.0003,.0001])
plt.legend()
plt.show()