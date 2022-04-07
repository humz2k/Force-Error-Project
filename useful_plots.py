# %% codecell

from calculations import *
from model import *
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

# %% codecell

def plot_r_potential(density=None,n_particles=None,radius=None,model=potential_model,file=None):

    if not (type(radius) is list or isinstance(radius, np.ndarray)):
        radius = [radius]

    for idx,a in enumerate(radius):

        rs,phis = get_program_for_particles(density=density,radius=a,n_particles=n_particles)
        cont_r = list(range(floor(np.min(rs)),ceil(np.max(rs))+1))
        #print([r/a for r in cont_r])
        analytic = [get_phi(density=density,radius=a,point=r/a) for r in cont_r]
        model = [potential_model(density=density,n_particles=n_particles,radius=a,point=r/a) for r in cont_r]
        plt.plot(cont_r,model,zorder=1,color="red",alpha=0.5,label="model")
        #plt.plot(cont_r,analytic,zorder=1,color="blue",alpha=0.5,label="analytic")
        plt.scatter(rs,phis,zorder=0,s=0.5)

    plt.xlabel("radius of particle")
    plt.ylabel("potential at particle")
    plt.title("Density="+str(density)+",N Particles="+str(n_particles)+",Radius of sphere="+str(radius[0]))
    plt.legend()
    if file != None:
        plt.savefig(file)
    else:
        plt.show()

p = 500
n = 2000
a = 100
file = "p" + str(p) + "n" + str(n) + "a" + str(a)
plot_r_potential(density=p,n_particles=n,radius=a,file=file)

def plot_radius_potential(density=None,n_particles=None,point=1,model=potential_model,show_theory=False,step=10,repeats=1,start=10,upper_limit=2000):
    if not (type(density) is list or isinstance(density, np.ndarray)):
        density = [density]

    if not (type(n_particles) is list or isinstance(n_particles, np.ndarray)):
        n_particles = [n_particles]

    if not (type(point) is list or isinstance(point, np.ndarray)):
        point = [point]
    legend = True
    fig, (overlay_plot) = plt.subplots(1,1)
    for r in point:
        for n in n_particles:
            for p in density:
                programs = []
                xs = range(start,upper_limit,step)
                theorys = []
                for i in xs:
                    if show_theory:
                        theorys.append(get_phi(density=p,radius=i,point=r))
                    programmatics = 0
                    for j in range(repeats):
                        programmatic = get_programmatic(density=p,radius=i,n_particles=n,point=r)
                        programmatics += programmatic
                    programmatics /= repeats
                    programs.append(programmatics)

                label = "("
                if len(n_particles) > 1:
                    label += "n=" + str(n)
                    if len(density) > 1:
                        label += ","
                if len(density) > 1:
                    label += "p=" + str(p)
                    if len(point) > 1:
                        label += ","
                if len(point) > 1:
                    label += "r="+str(r) +"a"
                label += ")"
                if len(n_particles) == 1 and len(density) == 1 and len(point) == 1:
                    label = ""
                    legend = False

                if model != None:
                    modeled = np.array([model(n_particles=n,density=p,radius=radius,point=r) for radius in xs])
                    overlay_plot.plot(xs,modeled,alpha=1, label = "MODL" + label,zorder=1)

                if model != None or show_theory:
                    mctslabel = "MCTS" + label
                else:
                    mctslabel = label
                overlay_plot.plot(xs,programs,alpha=0.8, label = mctslabel,zorder=0)
                if show_theory:
                    overlay_plot.plot(xs,theorys,alpha=0.8, label = "THRY" + label,zorder=0)

    if model != None or show_theory:
        legend = True
    overlay_plot.set_xlabel('Radius')
    overlay_plot.set_ylabel('Potential')
    if legend:
        overlay_plot.legend(loc ="lower left")
    title = 'varying a,sims=' + str(repeats)
    if len(n_particles) == 1:
        title += ",n=" + str(n_particles[0])
    if len(density) == 1:
        title += ",p=" + str(density[0])
    if len(point) == 1:
        title += ',r=' + str(point[0]) + 'a'
    overlay_plot.set_title(title,fontsize=10, pad=12)

    fig.suptitle('Radius/Potential',fontsize=16)
    fig.tight_layout()
    plt.show()

#plot_radius_potential(density=100,n_particles=10,point=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],repeats=100,model=None,start=400,upper_limit=500)

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

def plot_n_potential(density=None,radius=None,point=1,show_theory=False,model=potential_model,repeats=1,step=10,start=10,upper_limit=3000):
    if not (type(density) is list or isinstance(density, np.ndarray)):
        density = [density]

    if not (type(radius) is list or isinstance(radius, np.ndarray)):
        radius = [radius]

    if not (type(point) is list or isinstance(point, np.ndarray)):
        point = [point]

    fig, (overlay_plot) = plt.subplots(1,1)

    for r in point:
        for a in radius:
            for p in density:
                programs = []
                xs = range(start,upper_limit,step)
                if show_theory:
                    theorys = [get_phi(density=p,radius=a,point=r)] * len(xs)
                for n in xs:
                    programmatics = 0
                    for j in range(repeats):
                        programmatic = get_programmatic(density=p,radius=a,n_particles=n,point=r)
                        programmatics += programmatic
                    programmatics /= repeats
                    programs.append(programmatics)

                label = "("
                if len(radius) > 1:
                    label += "a=" + str(a)
                    if len(density) > 1:
                        label += ","
                if len(density) > 1:
                    label += "p=" + str(p)
                    if len(point) > 1:
                        label += ","
                if len(point) > 1:
                    label += "r="+str(r) +"a"
                label += ")"
                if len(radius) == 1 and len(density) == 1 and len(point) == 1:
                    label = ""

                if model != None:
                    modeled = np.array([model(n_particles=n,density=p,radius=a,point=r) for n in xs])
                    #print(r,np.mean(np.array(programs)/modeled))
                    overlay_plot.plot(xs,modeled,alpha=0.8, label = "MODL" + label)

                if model != None or show_theory:
                    mctslabel = "MCTS" + label
                else:
                    mctslabel = label
                overlay_plot.plot(xs,programs,alpha=0.8, label = mctslabel)
                if show_theory:
                    overlay_plot.plot(xs,theorys,alpha=0.8, label = "THRY" + label)

    overlay_plot.set_xlabel('N Particles')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="lower right")
    title = 'varying n,sims=' + str(repeats)
    if len(radius) == 1:
        title += ",a=" + str(radius[0])
    if len(density) == 1:
        title += ",p=" + str(density[0])
    if len(point) == 1:
        title += ',r=' + str(point[0]) + 'a'
    overlay_plot.set_title(title, pad=12)

    fig.suptitle('N Particles/Potential',fontsize=16)
    fig.tight_layout()
    plt.show()

#plot_n_potential(density=[100],radius=[100],point=[0.1,0.3,0.6,0.9,1],repeats=30,start=10,upper_limit=50,step=2,model=potential_model)

def plot_calculated_modeled_diff_n_potential(density=None,radius=None,point=1,repeats=1,step=10,start=10,upper_limit=3000,ylim=None,square=True):

    if not (type(repeats) is list or isinstance(repeats, np.ndarray)):
        repeats = [repeats]

    fig, (diff_plot) = plt.subplots(1,1)

    for zorder,repeat in enumerate(repeats):
        programs = []
        xs = list(range(start,upper_limit,step))
        theorys = [get_phi(density=density,radius=radius,point=point)] * len(xs)
        for n in xs:
            programmatics = 0
            for j in range(repeat):
                programmatic = get_programmatic(density=density,radius=radius,n_particles=n,point=point)
                programmatics += programmatic
            programmatics = programmatics/repeat
            programs.append(programmatics)
        programs = np.array(programs)
        modeled = np.array([potential_model(n,density,radius,point=point) for n in xs])
        diff = programs-modeled
        if square:
            diff = diff**2
        diff_plot.plot(xs,diff,label="sims="+str(repeat),alpha=1,zorder=zorder)
    #diff_plot.plot(list(range(start,n)),modeled,alpha=0.4, label = "Modeled")
    #diff_plot.plot(xs,theorys,alpha=0.8, label = "Theoretical")

    diff_plot.set_xlabel('N Particles')
    diff_plot.set_ylabel('Potential')
    if ylim != None:
        if square:
            diff_plot.set_ylim([-0.01*ylim,ylim])
        else:
            diff_plot.set_ylim([-ylim,ylim])
    diff_plot.legend(loc ="upper right")
    diff_plot.set_title('varying n,p= ' + str(density) + ',a= ' + str(radius) + ',r=' + str(point) + 'a', pad=12)

    fig.suptitle('Calculated/Modeled Difference for N Particles/Potential',fontsize=16)
    plt.legend()
    fig.tight_layout()
    plt.show()

#plot_calculated_modeled_diff_n_potential(density=100,radius=100,point=1,repeats=[10,50,100,300],upper_limit=100,step=1,ylim=1000)

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
    particles = get_particles(radius=radius,n_particles=n_particles)

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

#plot_particles(100,10
