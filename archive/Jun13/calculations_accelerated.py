from particles_generator import *
from theoretical_stuff import *
import numpy as np
import math
from scipy import spatial
from scipy import constants
import struct
import moderngl
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_pretty(dpi=150,fontsize=10):
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    plt.rc('legend',fontsize=5)
    plt.rcParams['figure.figsize'] = [5, 5]

plot_pretty()

ctx = moderngl.create_context(standalone=True)

prog = ctx.program(
    vertex_shader="""
    #version 330

    // Output values for the shader. They end up in the buffer.
    out float d;
    in vec3 pos;
    in vec3 check;

    void main() {
        d = distance(pos,check);
    }
    """,
    # What out varyings to capture in our buffer!
    varyings=["d"],
)

def get_dists(particles,f_size=4):
    n_particles = particles.shape[0]

    f_string = "f" + str(f_size)

    buffer1 = ctx.buffer(particles.astype(f_string).tobytes())
    outbuffer = ctx.buffer(reserve=n_particles*n_particles*(f_size))

    vao = ctx.vertex_array(prog, [(buffer1, "3" + f_string, "pos"),(buffer1, "3f /i", "check")])

    vao.transform(outbuffer,instances=n_particles)

    a = outbuffer.read()

    out = np.ndarray((n_particles*n_particles),f_string,a)

    vao.release()
    buffer1.release()
    outbuffer.release()

    return np.reshape(out,(n_particles,n_particles))

def get_dists_scipy(particles,**kwargs):
    spatials = []
    for i in particles:
        spatials.append(spatial.distance.cdist(particles,[i],**kwargs))
    return np.array(spatials).T[0]

def run_test(func,particles,**args):
    t1 = time.perf_counter()
    a = func(particles,**args)
    t2 = time.perf_counter()
    return t2-t1, a

xs = np.arange(2,22)**3
gpu = []
scipy = []
scipy_4byte = []

gpu_error = []
scipy_4byte_error = []

for n_particles in xs:
    print(n_particles)
    f8 = get_particles(radius=10,n_particles=n_particles).astype("f8")
    f4 = f8.astype("f4")

    scipy_8f_time,master = run_test(get_dists_scipy,f8)
    scipy.append(scipy_8f_time)

    gpu_4f_time,results = run_test(get_dists,f4,f_size=4)
    gpu.append(gpu_4f_time)
    gpu_error.append(np.nanmean(np.abs(results-master)))

    scipy_4f_time,results = run_test(get_dists_scipy,f4)
    scipy_4byte.append(scipy_4f_time)
    scipy_4byte_error.append(np.nanmean(np.abs(results-master)))

# %% codecell

fig, ((ax1), (ax2))  = plt.subplots(2, 1, sharex='col')

#fig.supxlabel("N Particles",fontsize=15)

ax1.plot(xs,gpu,label="ModernGL float32",linewidth=1)
ax1.plot(xs,scipy,label="Scipy float64",linewidth=1)
ax1.plot(xs,scipy_4byte,label="Scipy float32",linewidth=1)
ax1.set_ylabel("Runtime (s)",fontsize=15)
ax1.legend(loc="upper left", prop={'size': 8},frameon=True)

#ax2.set_yscale('log')
ax2.scatter(xs,gpu_error,label="ModernGL float32",s=5)
ax2.scatter(xs,scipy_4byte_error,label="Scipy float32",s=5)
ax2.legend(prop={'size': 8},frameon=True)
ax2.set_ylabel("Diff Scipy float64",fontsize=15)
start, end = ax2.get_ylim()
ax2.yaxis.set_ticks(np.linspace(start, end, 5))
ax2.set_xlabel("N Particles")

#plt.savefig("GPU vs Scipy with float64 diff")
plt.show()
