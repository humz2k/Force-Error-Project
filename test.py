import PyCC
import numpy as np
import matplotlib.pyplot as plt

df = PyCC.Distributions.Uniform(r=10000,n=10000,p=0.1)
ray = PyCC.ray(np.array([1,0,0]),100,25)

out,eval_out,stats = PyCC.evaluate(df,evaluate_at=ray,steps=1,precision="double")
print(stats)

true_val = eval_out.loc[:,["ax","ay","az","phi"]].to_numpy()

out2,eval_out2,stats = PyCC.evaluate(df,evaluate_at=ray,steps=1,precision="single")
print(stats)

next_val = eval_out2.loc[:,["ax","ay","az","phi"]].to_numpy()

print(np.mean(np.abs((true_val - next_val))))

for step in range(2):
    true_out = out[out["step"] == step].loc[:,["x","y","z"]].to_numpy()
    next_out = out2[out2["step"] == step].loc[:,["x","y","z"]].to_numpy()
    print(np.mean(np.abs(true_out-next_out)))
    

