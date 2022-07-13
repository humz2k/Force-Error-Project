import PyCC
import numpy as np
import matplotlib.pyplot as plt

r = 100
n = 500
p = 100

m_to_kpc = 3.24078e-20

test_dist = PyCC.Distributions.NFW(Rvir=r,p0=p,c=10,n=n)
#test_dist = PyCC.Distributions.Uniform(r=r,n=n,p=p)
out,stats = PyCC.evaluate(df=test_dist,save=False,algo="directsum",eval_type="both",steps=100,dt=100,eps=5)
particles = {}
for i in range(n):
    raw = out[out['id'] == i]
    data = {}
    data["pos"] = raw.loc[:,["x","y","z"]].to_numpy()
    data["acc"] = raw.loc[:,["ax","ay","az"]].to_numpy()
    data["vel"] = raw.loc[:,["vx","vy","vz"]].to_numpy()
    data["phi"] = raw.loc[:,["phi"]].to_numpy()
    particles[i] = data

fig = plt.figure()
ax = plt.axes(projection="3d")
for i in range(n):
    ax.plot(particles[i]["pos"][:,0],particles[i]["pos"][:,1],particles[i]["pos"][:,2])
plt.show()
