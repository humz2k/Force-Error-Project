# %% codecell
from calculations import *
from model import *
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]
# %% codecell
def plot_radius_potential(density,n_particles,model=potential_model,step=10,repeats=1,start=10,upper_limit=2000):
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
            temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=i)
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
    overlay_plot.set_title('Variable Radius, Density = ' + str(density) + ', N = ' + str(n_particles) + ', Average of = ' + str(repeats), pad=12)

    fig.suptitle('Theoretical vs Calculated vs Modeled',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_calculated_modeled_diff_radius_potential(density,n_particles,model=potential_model,step=10,repeats=1,start=10,upper_limit=2000):
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
            temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=i)
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
    overlay_plot.set_title('Variable Radius, Density = ' + str(density) + ', N = ' + str(n_particles) + ', Average of = ' + str(repeats), pad=12)

    fig.suptitle('Calculated/Modeled Difference',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_n_potential(density,radius,model=potential_model,repeats=1,step=10,start=10,upper_limit=3000):
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
            temp,theory,programmatic = get_diff(n_particles=i,density=density,radius=radius)
            programmatics += programmatic
            diff += temp
        programmatics = programmatics/repeats
        programs.append(programmatics)
        diff = abs(diff/repeats)
        diffs.append(diff)

    modeled = np.array([model(n,density,radius) for n in range(start,n)])

    fig, (overlay_plot) = plt.subplots(1,1)

    overlay_plot.plot(list(range(start,n)),modeled,alpha=0.4, label = "Modeled")
    overlay_plot.plot(xs,programs,alpha=0.4, label = "Calculated")
    overlay_plot.plot(xs,theorys,alpha=0.8, label = "Theoretical")
    overlay_plot.set_xlabel('N Particles')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="upper right")
    overlay_plot.set_title('Variable N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Average of = ' + str(repeats) + ', Step = ' + str(step), pad=12)

    fig.suptitle('Theoretical vs Calculated vs Modeled',fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_calculated_modeled_diff_n_potential(density,radius,repeats=1,step=10,start=10,upper_limit=3000,bar=False):
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
            temp,theory,programmatic = get_diff(n_particles=i,density=density,radius=radius)
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
        diff_plot.set_title('Variable N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Average of = ' + str(repeats) + ', Step = ' + str(step), pad=12)

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
# %% codecell 
plot_radius_potential(100,10,repeats=1)
# %% codecell
plot_n_potential(100,100,upper_limit=3000)
# %% codecell
plot_n_potential(100,10,upper_limit=3000)
# %% codecell
plot_n_potential(1000,10,upper_limit=3000)
# %% codecell
plot_calculated_modeled_diff_n_potential(100,100,upper_limit=500,repeats=10)
# %% codecell
plot_calculated_modeled_diff_n_potential(100,100,upper_limit=500,repeats=20)
# %% codecell
plot_calculated_modeled_diff_n_potential(100,100,upper_limit=500,repeats=1000)
