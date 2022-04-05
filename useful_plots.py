# %% codecell

from calculations import *
from model import *
import matplotlib.pyplot as plt

# %% codecell

def plot_radius_potential(density,n_particles,model=potential_model,step=10,repeats=1,start=10,upper_limit=2000,point="radius"):
    diffs = []
    theorys = []
    programs = []
    xs = []
    for i in range(start,upper_limit,step):
        xs.append(i)
        theorys.append(get_phi(n_particles=n_particles,density=density,radius=i))
        diff = 0
        programmatics = 0
        for j in range(repeats):
            temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=i,point=point)
            programmatics += programmatic
            diff += temp
        programmatics = programmatics/repeats
        programs.append(programmatics)
        diff = abs(diff/repeats)
        diffs.append(diff)

    modeled = np.array([model(n_particles,density,radius) for radius in range(start,upper_limit)])

    fig, (overlay_plot) = plt.subplots(1,1)

    overlay_plot.plot(list(range(start,upper_limit)),modeled,alpha=0.8, label = "Modeled")
    overlay_plot.plot(xs,programs,alpha=0.8, label = "Calculated")
    overlay_plot.plot(xs,theorys,alpha=0.8, label = "Theoretical")
    overlay_plot.set_xlabel('Radius')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="upper right")
    overlay_plot.set_title('Variable Radius, Density = ' + str(density) + ', N = ' + str(n_particles) + ', Average of = ' + str(repeats) + ', Measured from ' + point, pad=12)

    fig.suptitle('Theoretical vs Calculated vs Modeled',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_calculated_modeled_diff_radius_potential(density,n_particles,model=potential_model,step=10,repeats=1,start=10,upper_limit=2000,point="radius"):
    diffs = []
    theorys = []
    programs = []
    xs = []
    for i in range(start,upper_limit,step):
        xs.append(i)
        theorys.append(get_phi(n_particles=n_particles,density=density,radius=i))
        diff = 0
        programmatics = 0
        for j in range(repeats):
            temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=i,point=point)
            programmatics += programmatic
            diff += temp
        programmatics = programmatics/repeats
        programs.append(programmatics)
        diff = abs(diff/repeats)
        diffs.append(diff)

    modeled = np.array([model(n_particles,density,radius) for radius in xs])

    fig, (overlay_plot) = plt.subplots(1,1)

    #overlay_plot.plot(list(range(start,upper_limit)),modeled,alpha=0.4, label = "Modeled")
    overlay_plot.plot(xs,programs-modeled,alpha=0.4, label = "Calculated - Modeled")
    #overlay_plot.plot(xs,theorys,alpha=0.8, label = "Theoretical")
    overlay_plot.set_xlabel('Radius')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="upper right")
    overlay_plot.set_title('Variable Radius, Density = ' + str(density) + ', N = ' + str(n_particles) + ', Average of = ' + str(repeats) + ', Measured from ' + point, pad=12)

    fig.suptitle('Calculated/Modeled Difference',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_n_potential(density,radius,model=potential_model,repeats=1,step=10,start=10,upper_limit=3000,point="radius"):
    diffs = []
    theorys = []
    programs = []
    xs = []
    n = upper_limit
    for i in range(start,n,step):
        xs.append(i)
        theorys.append(get_phi(n_particles=i,density=density,radius=radius))
        diff = 0
        programmatics = 0
        for j in range(repeats):
            temp,theory,programmatic = get_diff(n_particles=i,density=density,radius=radius,point=point)
            programmatics += programmatic
            diff += temp
        programmatics = programmatics/repeats
        programs.append(programmatics)
        diff = abs(diff/repeats)
        diffs.append(diff)

    modeled = np.array([model(n,density,radius) for n in range(start,n)])

    fig, (overlay_plot) = plt.subplots(1,1)

    overlay_plot.plot(list(range(start,n)),modeled,alpha=0.5, label = "Modeled",zorder=1,linewidth=5)
    overlay_plot.plot(xs,programs,alpha=1, label = "Calculated",zorder=1)
    overlay_plot.plot(xs,theorys,alpha=0.8, label = "Theoretical",zorder=0)
    overlay_plot.set_xlabel('N Particles')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="upper right")
    overlay_plot.set_title('Variable N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Average of = ' + str(repeats) + ', Step = ' + str(step) + ', Measured from ' + point, pad=12)

    fig.suptitle('Theoretical vs Calculated vs Modeled',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_calculated_modeled_diff_n_potential(density,radius,repeats=1,step=10,start=10,upper_limit=3000,bar=False,point="radius"):
    diffs = []
    theorys = []
    programs = []
    xs = []
    n = upper_limit
    for i in range(start,n,step):
        xs.append(i)
        theorys.append(get_phi(n_particles=i,density=density,radius=radius))
        diff = 0
        programmatics = 0
        for j in range(repeats):
            temp,theory,programmatic = get_diff(n_particles=i,density=density,radius=radius,point=point)
            programmatics += programmatic
            diff += temp
        programmatics = programmatics/repeats
        programs.append(programmatics)
        diff = abs(diff/repeats)
        diffs.append(diff)

    modeled = np.array([potential_model(n,density,radius) for n in xs])

    fig, (diff_plot) = plt.subplots(1,1)

    if not bar:
        #diff_plot.plot(list(range(start,n)),modeled,alpha=0.4, label = "Modeled")
        diff_plot.plot(xs,programs-modeled,label = "Calculated - Modeled")
        #diff_plot.plot(xs,theorys,alpha=0.8, label = "Theoretical")

        diff_plot.set_xlabel('N Particles')
        diff_plot.set_ylabel('Potential')
        diff_plot.legend(loc ="upper right")
        diff_plot.set_title('Variable N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Average of = ' + str(repeats) + ', Step = ' + str(step) + ', Measured from ' + point, pad=12)

        fig.suptitle('Calculated/Modeled Difference',fontsize=16)
        fig.tight_layout()
        plt.show()
    else:
        to_plot = np.abs(programs-modeled)
        max = np.amax(to_plot)
        n = np.array(xs)[to_plot == max]
        print(max,"at",n)
        diff_plot.bar(xs,np.abs(programs-modeled))
        plt.show()

def get_rms_data(density,n_particles,radius,repeats=500):

    expected = potential_model(n_particles,density,radius)

    data = []
    for rep in range(repeats):
        temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=radius)
        data.append(programmatic)

    return expected,np.array(data)

def find_outliers(data,m=1.5):
    mean,std = np.mean(data),np.std(data)
    return abs(data - mean) < abs(std*m)

def get_rms(expected,data):
    squares = (expected-data)**2
    rm = np.sum(squares)/len(squares)
    return np.sqrt(rm)

def plot_rms_n_potential(density,radius,repeats=1,step=10,start=10,upper_limit=500,point="radius"):

    rms = []
    ns = list(range(start,upper_limit,step))
    for n in ns:
        temp_rms = 0
        for i in range(repeats):
            expected,data = get_rms_data(density,n,radius)

            data = data[find_outliers(data)]

            temp_rms += get_rms(expected,data)

        temp_rms /= repeats

        rms.append(temp_rms)

    data = [rms_model(n,density,radius) for n in ns]

    fig, (rms_plot) = plt.subplots(1,1)

    plt.plot(ns,data,label="Modeled")
    plt.plot(ns,rms,label="Calculated")
    plt.legend()

    rms_plot.set_title('Variable N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Average of = ' + str(repeats) + ', Step = ' + str(step) + ', Measured from ' + point, pad=12)
    rms_plot.set_xlabel('N Particles')
    rms_plot.set_ylabel('RMS Error')

    fig.suptitle('Calculated vs Modeled RMS Error',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_rms_diff_n_potential(density,radius,repeats=1,step=10,start=10,upper_limit=500,point="radius"):

    rms = []
    ns = list(range(start,upper_limit,step))
    for n in ns:
        temp_rms = 0
        for i in range(repeats):
            expected,data = get_rms_data(density,n,radius)

            data = data[find_outliers(data)]

            temp_rms += get_rms(expected,data)

        temp_rms /= repeats

        rms.append(temp_rms)

    data = np.array([rms_model(n,density,radius) for n in ns])

    fig, (rms_plot) = plt.subplots(1,1)

    plt.plot(ns,rms-data,label="Calculated-Modeled")
    plt.legend()

    rms_plot.set_title('Variable N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Average of = ' + str(repeats) + ', Step = ' + str(step) + ', Measured from ' + point, pad=12)
    rms_plot.set_xlabel('N Particles')
    rms_plot.set_ylabel('Delta RMS Error')

    fig.suptitle('Calculated/Modeled RMS Error Difference',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_particles(radius,n_particles,scale=20):
    radius,particles = get_particles(radius=radius,n_particles=n_particles)

    x = particles[:,0]
    y = particles[:,1]
    z = particles[:,2]

    fig = plt.figure()
    plot3d = fig.add_subplot(2,2,1, projection='3d')

    plot3d.scatter(x, y, z, c=z+radius/2)
    plot3d.set_xlabel('x')
    plot3d.set_ylabel('y')
    plot3d.set_zlabel('z')
    plot3d.set_xlim(-radius,radius)
    plot3d.set_ylim(-radius,radius)
    plot3d.set_zlim(-radius,radius)

    plotabove = fig.add_subplot(2,2,2)
    circle = plt.Circle(( 0 , 0 ), radius,fill=False,zorder=0)
    plotabove.add_artist(circle)
    plotabove.set_xlim(-radius,radius)
    plotabove.set_ylim(-radius,radius)
    plotabove.set(adjustable='box', aspect='equal')
    plotabove.scatter(x, y,s = (((z+radius/2)/radius)**2)*scale,c=z+radius/2,zorder=1)
    plotabove.set_xlabel('x')
    plotabove.set_ylabel('y')

    plotside = fig.add_subplot(2,2,3)
    circle = plt.Circle(( 0 , 0 ), radius,fill=False,zorder=0)
    plotside.add_artist(circle)
    plotside.set_xlim(-radius,radius)
    plotside.set_ylim(-radius,radius)
    plotside.set(adjustable='box', aspect='equal')
    plotside.scatter(x, z, c=z+radius/2,s = (((y+radius/2)/radius)**2)*scale,zorder=1)
    plotside.set_xlabel('x')
    plotside.set_ylabel('z')

    plotside2 = fig.add_subplot(2,2,4)
    circle = plt.Circle(( 0 , 0 ), radius,fill=False,zorder=0)
    plotside2.add_artist(circle)
    plotside2.set_xlim(-radius,radius)
    plotside2.set_ylim(-radius,radius)
    plotside2.set(adjustable='box', aspect='equal')
    plotside2.scatter(y, z, c=z+radius/2,s = (((x+radius/2)/radius)**2)*scale,zorder=1)
    plotside2.set_xlabel('x')
    plotside2.set_ylabel('z')

    fig.suptitle("Particles Generated by r = " + str(radius) + " & n = " + str(n_particles))
    fig.tight_layout(w_pad=5)

    plt.show()
