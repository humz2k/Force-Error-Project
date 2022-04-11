# %% codecell

from calculations import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model import *

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
    stds = []
    for i in xs:
        mean,std = np.mean(y[x == i]),np.std(y[x == i])
        line1.append(mean + m*std)
        line2.append(mean - m*std)
        line3.append(mean)
        stds.append(std)
    return (xs,np.array(line1)),(xs,np.array(line2)),(xs,np.array(line3)),(xs,np.array(stds))

def get_model(n_particles,density,average_of=1,min=1,max=1000,step=1,repeats=100,make_plot=True,plot_type="normal"):

    uppers = []
    lowers = []
    middles = []
    std_data = []

    #n_particles = 10
    for i in range(average_of):
        print("     ",round(100*(i/average_of)))
        raw = get_model_data(density,n_particles,rs=range(min,max,step),repeats=repeats)
        data = remove_outliers(raw)
        upper,lower,middle,stds = get_error_mean(raw)
        std_data.append(stds)
        uppers.append(upper)
        lowers.append(lower)
        middles.append(middle)
        if make_plot:
            if plot_type == "std":
                pass
            else:
                plt.scatter(data[0], data[1],s=0.1,c="grey",alpha=0.5)

    if plot_type != "std":
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
    else:
        temp = np.zeros_like(std_data[0][1])
        for i in std_data:
            temp += i[1]
        std = (std_data[0][0],temp/len(std_data))

    if make_plot:
        if plot_type == "std":
            pass
        else:
            plt.plot(upper[0],upper[1],label="upper_error")
            plt.plot(lower[0],lower[1],label="lower_error")
    #plt.plot(middle[0],middle[1])

    if plot_type == "std":
        pass
    else:

        def objective(x,a,b):
            return -(x**b)/a

        popt, _ = curve_fit(objective,middle[0],middle[1])
        a,b = popt
        a_actual = (a/n_particles)/((1/density)**2)

        y_new = objective(middle[0],a,b)

    if make_plot:
        if plot_type == "std":
            #print(std_data)
            plt.plot(std[0],std[1])
            plt.show()
        else:
            plt.plot(middle[0],y_new,label="Model")

            plt.legend(loc = "lower left")

            plt.show()

    if plot_type != "std":
        return a_actual,b

    return std

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
'''

from_file = True
shit = False

density = 500

if from_file:

    if shit:
        std = get_model(10,density,average_of=10,step=10,plot_type="std",make_plot=True)
    else:

        with open("test_model_data_500.txt","r") as f:
            temp = np.array([[float(j) for j in i.split(",")] for i in f.read().splitlines()])

        def objective(x,a,b):
            return a*x + b

        popt, _ = curve_fit(objective,temp[:,0],temp[:,1])
        a,b = popt
        print(a,b)
        y_new = objective(temp[:,0],a,b)
        #plt.plot(temp[:,0],temp[:,1])
        plt.plot(temp[:,0],temp[:,1],label="data")
        plt.plot(temp[:,0],y_new,label="model")

else:
    with open("test_model_data_100_fixed.txt","w") as f:
        a_data = []
        xs = []
        for i in range(10,50,5):
            print((i/50)*100)
            xs.append(i)
            std = get_model(i,density,average_of=40,step=10,plot_type="std",make_plot=False)

            def objective(x,a,b):
                return (x**b)/a

            popt, _ = curve_fit(objective,std[0],std[1])
            a,b = popt
            a_actual = (a/i)/((1/density)**2)
            a_data.append(a)
            f.write(str(i) + "," + str(a) + "," + str(b) + "\n")

            y_new = objective(std[0],a,b)

            plt.plot(std[0],std[1],label=str(i))

            plt.plot(std[0],y_new,label=str(i)+"model")

        #plt.plot(xs,a_data)



plt.legend()
plt.show()


'''
for n in range(20,41,10):
    density = 500
    #n = 20

    std = get_model(n,density,average_of=5,step=10,plot_type="std",make_plot=False)

    plt.plot(std[0],std[1],label=str(n))

    data = np.array([std_model(n,density,radius) for radius in range(1,1000)])

    plt.plot(range(1,1000),data,label=str(n)+"model")

plt.legend(loc="upper right")

plt.show()
'''

#plt.plot(xs,a_data)
