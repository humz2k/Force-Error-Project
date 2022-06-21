import PyCC
import numpy as np

test_dist = PyCC.Distributions.Uniform(r=10,n=10000,p=10)
print(test_dist)
treecode,stats = PyCC.evaluate(df=test_dist,save=False,algo="treecode",theta=1)
treecode = treecode.loc[:,"phi"].to_numpy()
print(stats)

direct,stats = PyCC.evaluate(df=test_dist,save=False,algo="directsum")
direct = direct.loc[:,"phi"].to_numpy()
print(stats)

diff = np.abs(treecode-direct)
print(np.mean(diff),np.max(diff))