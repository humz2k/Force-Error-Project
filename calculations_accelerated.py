from particles_generator import *
from theoretical_stuff import *
import numpy as np
import math
from scipy import spatial
from scipy import constants
import struct
import moderngl
import time

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

n_particles = 10000
print("gen particles")
particles = get_particles(radius=10,n_particles=n_particles)
print("done")

def get_dists(particles):
    n_particles = particles.shape[0]

    print(n_particles)

    buffer1 = ctx.buffer(particles.astype('f4').tobytes())
    outbuffer = ctx.buffer(reserve=n_particles*n_particles*(4))

    vao = ctx.vertex_array(prog, [(buffer1, "3f", "pos"),(buffer1, "3f /i", "check")])

    vao.transform(outbuffer,instances=n_particles)

    a = outbuffer.read()

    print(len(a))

    out = np.ndarray((n_particles*n_particles),"f4",a)

    vao.release()
    buffer1.release()
    outbuffer.release()

    return np.reshape(out,(n_particles,n_particles))

def get_dists_scipy(particles):
    spatials = []
    for i in particles:
        spatials.append(spatial.distance.cdist(particles,[i]))
    return np.array(spatials).T[0]

t1 = time.perf_counter()
test = get_dists(particles)
t2 = time.perf_counter()
print(t2-t1)

t1 = time.perf_counter()
test2 = get_dists_scipy(particles)
t2 = time.perf_counter()
print(t2-t1)

print(np.mean(np.abs(test-test2)))
