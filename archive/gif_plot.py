from calculations import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython import display
plt.rcParams['figure.figsize'] = [8, 8]

def plot_r_potential(n_particles,density,repeats,radiuses):
    theorys = [get_phi(density=density,radius=radius) for radius in radiuses]
    theory_xs = radiuses
    data = []
    xs = []

    for idx1,n in enumerate(n_particles):
        #temp_theory = []
        #temp_theory_xs = []
        temp_data = []
        temp_xs = []
        for idx2,radius in enumerate(radiuses):
            #this_theory = get_phi(density=density,radius=radius)
            #print(this_theory)
            #temp_theory.append(this_theory)
            #temp_theory_xs.append(radius)
            for i in range(repeats):
                temp,theory,programmatic = get_diff(n_particles=n,density=density,radius=radius)
                temp_data.append(programmatic)
                temp_xs.append(radius)
        #theorys.append(temp_theory.copy())
        #theory_xs.append(temp_theory_xs)
        data.append(temp_data.copy())
        xs.append(temp_xs)

    radiuses = np.array(radiuses)
    theorys = np.array(theorys)
    theory_xs = np.array(theory_xs)
    data = np.array(data)
    xs = np.array(xs)

    fig, ax = plt.subplots()

    data_plot = plt.scatter(xs[0], data[0])
    plt.plot(theory_xs,theorys)
    fig.suptitle('Vary Radius, N = ' + str(n_particles[0]) + ', Density = ' + str(density) + ', Repeats = ' + str(repeats), fontsize=16)
    ax.set_ylabel('Potential')
    ax.set_xlabel('Radius')
    plt.subplots_adjust(left=0.1,bottom=0.25)
    #ax.set_xlabel('Time [s]')

    axslid = plt.axes([0.2, 0.1, 0.7, 0.03])
    n_slider = Slider(ax=axslid, label='N Particles', valmin=0, valmax=len(n_particles)-1,valinit=0,valstep=1,orientation="horizontal",valfmt='%0.1f')
    axslid.add_artist(axslid.xaxis)
    xticks = [str(l) for l in n_particles]
    axslid.set_xticks([l for l in range(len(n_particles))])
    axslid.set_xticklabels(xticks)

    def update(val):
        temp = np.hstack([xs[n_slider.val][:,np.newaxis],data[n_slider.val][:,np.newaxis]])
        fig.suptitle('Vary Radius, N = ' + str(n_particles[n_slider.val]) + ', Density = ' + str(density) + ', Repeats = ' + str(repeats), fontsize=16)
        data_plot.set_offsets(temp)
        fig.canvas.draw_idle()

    n_slider.on_changed(update)

    #update(1)

    plt.show()

def gif_r_potential(n_particles,density,repeats,radiuses):
    theorys = [get_phi(density=density,radius=radius) for radius in radiuses]
    theory_xs = radiuses
    data = []
    xs = []

    for idx1,n in enumerate(n_particles):
        #print(idx1/len(n_particles))
        #temp_theory = []
        #temp_theory_xs = []
        temp_data = []
        temp_xs = []
        for idx2,radius in enumerate(radiuses):
            #print("     ",idx2/len(radiuses))
            print((idx2+len(radiuses)*idx1)/(len(radiuses) * len(n_particles)))
            #this_theory = get_phi(density=density,radius=radius)
            #print(this_theory)
            #temp_theory.append(this_theory)
            #temp_theory_xs.append(radius)
            for i in range(repeats):
                temp,theory,programmatic = get_diff(n_particles=n,density=density,radius=radius)
                temp_data.append(programmatic)
                temp_xs.append(radius)
        #theorys.append(temp_theory.copy())
        #theory_xs.append(temp_theory_xs)
        data.append(temp_data.copy())
        xs.append(temp_xs)

    radiuses = np.array(radiuses)
    theorys = np.array(theorys)
    theory_xs = np.array(theory_xs)
    data = np.array(data)
    xs = np.array(xs)

    fig, ax = plt.subplots()

    plt.ylim([-8e6,0.5e6])
    data_plot = plt.scatter(xs[0], data[0],zorder=1,s=0.1,label='MTCS')
    plt.plot(theory_xs,theorys,zorder=0,color='red',label='Theory')
    plt.legend(loc='lower left')
    fig.suptitle('Vary Radius, N = ' + str(n_particles[0]) + ', Density = ' + str(density) + ', Repeats = ' + str(repeats), fontsize=16)
    ax.set_ylabel('Potential')
    ax.set_xlabel('Radius')
    plt.subplots_adjust(left=0.1,bottom=0.25)
    #ax.set_xlabel('Time [s]')

    '''
    axslid = plt.axes([0.2, 0.1, 0.7, 0.03])
    n_slider = Slider(ax=axslid, label='N Particles', valmin=0, valmax=len(n_particles)-1,valinit=0,valstep=1,orientation="horizontal",valfmt='%0.1f')
    axslid.add_artist(axslid.xaxis)
    xticks = [str(l) for l in n_particles]
    axslid.set_xticks([l for l in range(len(n_particles))])
    axslid.set_xticklabels(xticks)
    '''


    def update(val):
        temp = np.hstack([xs[val-1][:,np.newaxis],data[val-1][:,np.newaxis]])
        fig.suptitle('Vary Radius, N = ' + str(n_particles[val-1]) + ', Density = ' + str(density) + ', Repeats = ' + str(repeats), fontsize=16)
        data_plot.set_offsets(temp)
        #fig.canvas.draw_idle()

    anim =FuncAnimation(fig,update,frames=len(n_particles))

    writergif = PillowWriter(fps=15)
    anim.save("test.gif",writer=writergif)
    #html = display.HTML(video)
    #display.display(html)

    #plt.close()
    #n_slider.on_changed(update)

    #update(1)

    #plt.show()

gif_r_potential(list(range(1,3001,50)),100,100,list(range(1,1000,10)))
