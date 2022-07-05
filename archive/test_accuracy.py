import PyCC
import numpy as np
import matplotlib.pyplot as plt

thetas = [0.25,0.5,0.75,1]
ns = [10,100,1000,10000]

mean_errors = {}
max_errors = {}
for theta in thetas:
    mean_errors[theta] = []
    max_errors[theta] = []

for n in ns:
    test_dist = PyCC.Distributions.Uniform(r=10,n=100,p=10)
    direct,direct_stats = PyCC.evaluate(df=test_dist,save=False,algo="directsum")

    for theta in thetas:
        treecode,stats = PyCC.evaluate(df=test_dist,save=False,algo="treecode",theta=theta)
        diff = treecode.loc[:,"phi"].to_numpy() - direct.loc[:,"phi"].to_numpy()
        abs_diff = np.abs(diff)
        mean_errors[theta].append(np.mean(abs_diff))
        max_errors[theta].append(np.max(abs_diff))

fig, axs = plt.subplots(2)
fig.suptitle('Treecode vs Direct Sum')

for theta in thetas:
    axs[0].scatter(ns,mean_errors[theta],label=r"$\theta = " + str(theta) + r"$",s=10,zorder=1)
    axs[1].scatter(ns,max_errors[theta],label=r"$\theta = " + str(theta) + r"$",s=10,zorder=1)
    axs[0].plot(ns,mean_errors[theta],alpha=0.5,zorder=0)
    axs[1].plot(ns,max_errors[theta],alpha=0.5,zorder=0)

axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[1].legend(loc='upper right', bbox_to_anchor=(1.1, 1.2))
axs[0].set_ylabel("Mean Error")
axs[1].set_ylabel("Max Error")
axs[0].set_xlabel("N Particles")
axs[1].set_xlabel("N Particles")
plt.show()