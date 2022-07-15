import PyCC
import numpy as np
import matplotlib.pyplot as plt

M = 1
G = 1
a = 1
n = 100

df = PyCC.Distributions.Plummer(n=n,M=M,G=G,a=a)

out,stats = PyCC.evaluate(df,steps=0,G=G,precision="single",dt=1/64)
print(stats)

print(stats["batch_times"] * stats["n_batches"])
print(stats["yeehaw"] * stats["n_batches"])

out,stats = PyCC.evaluate(df,steps=0,G=G,precision="double",dt=1/64)
print(stats)
print(out)

exit()

def e_phis(outs):
    steps = np.unique(outs.loc[:,"step"].to_numpy())
    phis = np.zeros((len(steps)),dtype=float)
    for step in steps:
        phis[step] = np.sum(out[out["step"] == step].loc[:,"phi"].to_numpy())/2
    return phis

def e_kin(outs,df):
    steps = np.unique(outs.loc[:,"step"].to_numpy())
    energies = np.zeros((len(steps)),dtype=float)
    for step in steps:
        energies[step] = np.sum(0.5 * df.loc[:,"mass"].to_numpy() * np.linalg.norm(out[out["step"] == step].loc[:,["vx","vy","vz"]].to_numpy(),axis=1)**2)
    return energies

print(e_phis(out))
print(e_kin(out,df))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for id in range(n):
    pos = out[out["id"] == id].loc[:,["x","y","z"]].to_numpy()
    plt.plot(pos[:,0],pos[:,1],pos[:,2])

plt.show()