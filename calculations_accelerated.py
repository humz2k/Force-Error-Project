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

n_particles = 3000
print("gen particles")
particles = get_particles(radius=10,n_particles=n_particles)
print("done")

def get_distance(particles,start_point,prog=prog):
    n_particles = len(particles)
    to_check = np.full_like(particles,start_point)
    vertices = np.reshape(np.concatenate((particles,to_check),axis=1),(1,n_particles,6))
    vbo = ctx.buffer(vertices.astype('f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'pos', 'check')
    buffer = ctx.buffer(reserve=n_particles * 4)
    vao.transform(buffer)
    out = np.array(struct.unpack(str(n_particles)+"f",buffer.read()))
    buffer.release()
    vao.release()
    vbo.release()
    return out

def get_distances(particles,prog=prog):
    n_particles = particles.shape[0]

    #print(np.dstack([particles[:,0],particles[:,1],particles[:,2]]))

    t3 = time.time()

    checks = np.reshape(np.repeat(particles,n_particles,axis=0),(1,n_particles*n_particles,3))

    particles = np.reshape(particles,(1,) + particles.shape)
    #print(particles)

    parts = np.concatenate([particles]*n_particles,axis=1)

    buffer1 = ctx.buffer(parts.astype('f4').tobytes())

    buffer2 = ctx.buffer(checks.astype('f4').tobytes())

    t4 = time.time()
    print(t4-t3)

    vao = ctx.vertex_array(prog, [(buffer1, "3f", "pos"),(buffer2,"3f","check")])

    outbuffer = ctx.buffer(reserve=n_particles*n_particles*(4))

    vao.transform(outbuffer,vertices=n_particles*n_particles)

    out = np.array(struct.unpack(str(n_particles*n_particles)+"f",outbuffer.read()))

    outbuffer.release()
    buffer2.release()
    buffer1.release()
    vao.release()

    return out


t1 = time.time()
out = get_distances(particles)
t2 = time.time()
print(t2-t1)
#print(first)
#print(second)

t1 = time.time()
spatials = np.zeros((len(particles),len(particles)))
for i in range(len(particles)):
    spatials[i] = spatial.distance.cdist(particles,np.reshape(particles[i],(1,3))).T
t2 = time.time()
spatials = spatials.flatten()
print(t2-t1)


#t12 = time.time()
#test = spatial.distance.cdist(particles,np.reshape(start_point,(1,3)))
#print(len(test))
#t22 = time.time()
#print(t22-t12)

'''
to_check = np.full_like(particles[1:],particles[0])

particles = particles[1:]

test = np.reshape(np.concatenate((particles,to_check),axis=1),(1,n_particles-1,6))
print(test.shape)
print(test)

vbo = ctx.buffer(test.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'pos', 'check')
buffer = ctx.buffer(reserve=(n_particles-1) * 4)
vao.transform(buffer)

out = np.array(struct.unpack(str(n_particles-1)+"f",buffer.read()))
print(out)

to_check = np.reshape(to_check[0],(1,3))

print(spatial.distance.cdist(particles,to_check))
'''
