import PyCC
import numpy as np
import matplotlib.pyplot as plt

n = 100
steps = 200
df = PyCC.Distributions.Uniform(r=1000000000000,p=100,n=n)

for dt in [1,10,100]:
    print(dt)
    out_sum,stats = PyCC.evaluate(df=df,save=False,dt=dt,steps=steps,algo="directsum",eps=5)
    print("Sum",stats)
    out_tree,stats = PyCC.evaluate(df=df,save=False,dt=dt,steps=steps,algo="treecode",theta=1,eps=5)
    print("Tree",stats)
    error = np.zeros(steps+1,dtype=float)

    for id in range(n):
        acc_sum = out_sum[out_sum["id"] == id].loc[:,["ax","ay","az"]].to_numpy()
        acc_tree = out_tree[out_tree["id"] == id].loc[:,["ax","ay","az"]].to_numpy()
        error += np.mean(np.abs(acc_sum - acc_tree),axis=1)

    plt.plot((error/(steps+1))[:-1],label="dt="+str(dt))
plt.yscale('log')
plt.legend()
plt.show()
