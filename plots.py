# %% codecell
from calculations import *
import matplotlib.pyplot as plt

# %% codecell
def plot_n_potential(density,radius,repeats=1):
    diffs = []
    theorys = []
    programs = []
    xs = []
    n = 2000
    start = 10
    for i in range(start,n,10):
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

    fig, (overlay_plot,diff_plot) = plt.subplots(2,1)

    overlay_plot.plot(xs,programs,alpha=0.4, label = "Calculated")
    overlay_plot.plot(xs,theorys,alpha=0.4, label = "Theoretical")
    overlay_plot.set_xlabel('N Particles')
    overlay_plot.set_ylabel('Potential')
    overlay_plot.legend(loc ="upper right")
    overlay_plot.set_title('Theoretical/Calculated Overlay',pad=12)

    diff_plot.plot(xs,diffs)
    diff_plot.set_xlabel('N Particles')
    diff_plot.set_ylabel('Delta Potential')
    diff_plot.set_title('Theoretical/Calculated Difference',pad=12)

    fig.suptitle('Constant N, Density = ' + str(density) + ', Radius = ' + str(radius) + ', Repeats = ' + str(repeats), fontsize=16)
    fig.tight_layout()
    plt.show()

# %% codecell

def get_spread(n_particles,density,radius,repeats):
    theory = get_phi(density=density,radius=radius)
    data = []
    for i in range(repeats):
        temp,theory,programmatic = get_diff(n_particles=n_particles,density=density,radius=radius)
        data.append(programmatic)

    fig, (scatter_plot,box_plot) = plt.subplots(1,2)

    scatter_plot.scatter([1]*len(data),data,s=5, label = "Calculated")
    scatter_plot.scatter([1],[theory],s=5, label = "Theoretical")
    scatter_plot.legend(loc ="upper right",prop={'size': 6})
    scatter_plot.set_ylabel('Potential')
    scatter_plot.get_xaxis().set_visible(False)
    scatter_plot.set_title('Theoretical/Calculated Constant N',pad=12)

    box_plot.set_title('Calculated Box Plot',pad=12)
    box_plot.boxplot(data)
    box_plot.set_ylabel('Potential')
    box_plot.get_xaxis().set_visible(False)

    fig.suptitle('N = ' + str(n_particles) + ', Density = ' + str(density) + ', Radius = ' + str(radius) + ', Repeats = ' + str(repeats), fontsize=16)
    fig.tight_layout()
    plt.show()

# %% codecell

plot_n_potential(100,100)

# %% codecell

plot_n_potential(10000,100)


# %% codecell

plot_n_potential(10000,10)

# %% codecell

plot_n_potential(5,10)

# %% codecell
get_spread(100,100,100,1000)

# %% codecell
get_spread(1000,100,100,1000)

# %% codecell
get_spread(2000,10,1000,1000)

# %% codecell
get_spread(2500,1000,10,1000)
