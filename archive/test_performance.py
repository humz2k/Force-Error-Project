import PyCC
import numpy as np
import matplotlib.pyplot as plt


theta_1_ys = []
theta_2_ys = []
theta_05_ys = []
xs = []
direct_ys = []
mean_error = []
max_error = []
truncs = []

ray = PyCC.ray(np.array([1,0,0]),20,25,None)

max_n = 5000
for n in range(2000,max_n,2000):
    print(n/max_n)
    xs.append(n)

    test_dist = PyCC.Distributions.Uniform(r=10,n=n,p=10)

    theta_1,theta_1_stats = PyCC.evaluate(df=test_dist,save=False,algo="treecode",theta=1)
    theta_1_ys.append(theta_1_stats["eval_time"])

    truncs.append(theta_1_stats["truncations"]/(theta_1_stats["truncations"] + theta_1_stats["directs"]))

    #theta_2,theta_2_stats = PyCC.evaluate(df=test_dist,evaluate_at=ray,save=False,algo="treecode",theta=2)
    #theta_2_ys.append(theta_1_stats["eval_time"])

    #theta_05,theta_05_stats = PyCC.evaluate(df=test_dist,evaluate_at=ray,save=False,algo="treecode",theta=0.5)
    #theta_05_ys.append(theta_05_stats["eval_time"])

    direct,direct_stats = PyCC.evaluate(df=test_dist,save=False,algo="directsum")
    direct_ys.append(direct_stats["eval_time"])

    diff = theta_1.loc[:,"phi"].to_numpy() - direct.loc[:,"phi"].to_numpy()
    abs_diff = np.abs(diff)
    mean_error.append(np.mean(abs_diff))
    max_error.append(np.max(abs_diff))


fig, axs = plt.subplots(2)
fig.suptitle('Treecode vs Direct Sum')

axs[0].plot(xs,direct_ys,label="directsum")
axs[0].plot(xs,theta_1_ys,label="treecode, " + r"$\theta = 1$")
#plt.plot(xs,theta_2_ys,label="treecode, " + r"$\theta = 2$")
#plt.plot(xs,theta_05_ys,label="treecode, " + r"$\theta = 0.5$")
axs[0].set_ylabel("Time")
axs[0].set_xlabel("N Particles")
axs[0].legend()

axs[1].plot(xs,mean_error,label="mean")
axs[1].plot(xs,max_error,label="max")
axs[1].set_xlabel("N Particles")
axs[1].set_ylabel(r"$|\phi_{sum} - \phi_{treecode}|$")
axs[1].set_yscale('log')
axs[1].legend()

plt.tight_layout()

plt.show()


