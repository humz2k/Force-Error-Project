# %% codecell
from calculations import *
from model import *
import matplotlib.pyplot as plt
from useful_plots import *

plt.rcParams['figure.figsize'] = [8, 8]

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
