# %% codecell

from calculations import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_model_data(density,n_particles,rs=range(1,3000),repeats=100):

    data = []
    xs = []
    for radius in rs:
        temp_data = []
        temp_xs = []
        for rep in range(repeats):
            temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=radius)
            temp_data.append(programmatic)
            temp_xs.append(radius)
        data.append(temp_data)
        xs.append(temp_xs)


    return np.array(xs),np.array(data)

def find_outliers(data,m=1.5):
    mean,std = np.mean(data),np.std(data)
    return abs(data - mean) < abs(std*m)

def remove_outliers(model_data):
    x,y = model_data
    out_x = []
    out_y = []
    total = 0
    for i in range(x.shape[0]):
        outliers = find_outliers(y[i])
        total += len(outliers) - np.count_nonzero(outliers)
        out_y += y[i][outliers].tolist()
        out_x += x[i][outliers].tolist()

    return np.array(out_x),np.array(out_y)

def get_error_mean(model_data,m=1.5):
    x,y = model_data[0].flatten(),model_data[1].flatten()
    xs = np.unique(x)
    line1 = []
    line2 = []
    line3 = []
    for i in xs:
        mean,std = np.mean(y[x == i]),np.std(y[x == i])
        line1.append(mean + m*std)
        line2.append(mean - m*std)
        line3.append(mean)
    return (xs,np.array(line1)),(xs,np.array(line2)),(xs,np.array(line3))

def get_model(n_particles,density,average_of=1,min=1,max=1000,step=1,repeats=100):

    uppers = []
    lowers = []
    middles = []

    #n_particles = 10
    for i in range(average_of):
        print("     ",round(100*(i/average_of)))
        raw = get_model_data(density,n_particles,rs=range(min,max,step),repeats=repeats)
        data = remove_outliers(raw)
        upper,lower,middle = get_error_mean(raw)
        uppers.append(upper)
        lowers.append(lower)
        middles.append(middle)
        #plt.plot(upper[0],upper[1])
        #plt.plot(lower[0],lower[1])
        #plt.scatter(data[0], data[1],s=0.1,c="grey",alpha=0.5)

    temp = np.zeros_like(uppers[0][1])
    for i in uppers:
        temp += i[1]
    upper = (uppers[0][0],temp/len(uppers))

    temp = np.zeros_like(lowers[0][1])
    for i in lowers:
        temp += i[1]
    lower = (lowers[0][0],temp/len(lowers))

    temp = np.zeros_like(middles[0][1])
    for i in middles:
        temp += i[1]
    middle = (middles[0][0],temp/len(middles))

    #plt.plot(upper[0],upper[1])
    #plt.plot(lower[0],lower[1])
    #plt.plot(middle[0],middle[1])

    def objective(x,a,b):
        return -(x**b)/a

    popt, _ = curve_fit(objective,middle[0],middle[1])
    a,b = popt
    a_actual = (a/n_particles)/((1/density)**2)
    print("Model: ", "-x**(((1/density)**2)*(" + str(a_actual) + " * n))" + "/(" + str(b) + ")")
    #y_new = objective(middle[0],a,b)

    return a_actual,b

    #print(y_new)

    #plt.plot(middle[0],y_new)

    #plt.show()

#print(get_model(10,500))
#print(get_model(10,1000))
#print(get_model(10,2000))

'''
with open("model_data_density_50.txt","w") as f:
    density = 500
    a_data = []
    b_data = []
    ns = []
    print("Density =",density)
    for n in range(10,110,10):
        print("Doing N =",n)
        a,b = get_model(n,density,average_of=1)
        a_data.append(a)
        b_data.append(b)
        ns.append(n)
        f.write(str(n) + "," + str(a) + "," + str(b) + "\n")

fig, (a_plot,b_plot) = plt.subplots(2,1)
a_plot.plot(ns,a_data)
a_plot.plot(ns,[np.mean(a_data)]*len(ns))
b_plot.plot(ns,b_data)
b_plot.plot(ns,[np.mean(b_data)]*len(ns))
plt.show()
'''

with open("test_model_data_good_2.txt","w") as f:
    a_data = []
    ns = []
    max = 200
    for i in range(10,2000,10):
        print((i/max)*100)
        a,b = get_model(20,i,average_of=10)
        f.write(str(i) + "," + str(a) + "," + str(b) + "\n")
        a_data.append(a)
        ns.append(i)

plt.plot(ns,a_data)
plt.semilogy(base=10)
plt.show()
