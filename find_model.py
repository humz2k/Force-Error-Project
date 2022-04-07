from calculations import *
from scipy.optimize import curve_fit

def get_radius_potential_model(density=None,n_particles=None,point=1,start=10,step=10,upper_limit=2000,repeats=1):
    programs = []
    xs = range(start,upper_limit,step)
    for i in xs:
        programmatics = 0
        for j in range(repeats):
            programmatic = get_programmatic(density=density,radius=i,n_particles=n_particles,point=point)
            programmatics += programmatic
        programmatics /= repeats
        programs.append(programmatics)
    xs = np.array(list(xs))
    programs = np.array(programs)

    def objective(x,a):
        return -(x**5)/a

    popt, _ = curve_fit(objective,xs,programs)
    a, = popt

    def model_func(x=None,param='y'):

        if param == 'a':
            return a

        if x == None:
            return "-(x**5)/" + str(a)

        return objective(x,a)

    return model_func
