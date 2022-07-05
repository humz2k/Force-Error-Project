#DO UNITS

import numpy as np
import struct
import moderngl
import time
from astropy import constants
from math import ceil

ctx = moderngl.create_context(standalone=True)

prog = ctx.program(

    vertex_shader="""
    #version 330

    in vec3 part;
    in vec3 eval_pos;

    out float d;

    void main() {
        d = distance(part,eval_pos);
    }

    """,

    varyings=["d"],

)

def do_batch(part_buffer,n_particles,eval_pos):
    first = time.perf_counter()

    n_evals = eval_pos.shape[0]

    eval_buffer = ctx.buffer(eval_pos.astype("f4").tobytes())

    outbuffer = ctx.buffer(reserve=n_particles*n_evals*4*4)

    vao = ctx.vertex_array(prog, [(part_buffer, "3f", "part"),(eval_buffer, "3f /i", "eval_pos")])

    vao.transform(outbuffer,instances=n_evals)

    out = np.ndarray((n_particles*n_evals),"f4",outbuffer.read())
    out = np.reshape(out,(n_evals,n_particles))

    vao.release()
    outbuffer.release()
    eval_buffer.release()

    second = time.perf_counter()

    return out,second-first

class DirectSum(object):

    @staticmethod
    def acc_and_phi(pos,particles,masses,dists,eps=0):
        masses = masses[dists != 0]
        parts = particles[dists != 0]
        dists = dists[dists != 0]
        if eps == 0:
            potentials = (-1) * constants.G.value * (masses)/dists
            muls = (constants.G.value * ((masses) / (dists**3)))
            accelerations = (parts - pos) * np.reshape(muls,(1,) + muls.shape).T
        else:
            potentials = (-1) * constants.G.value * (masses)/((dists**2+eps**2)**(1/2))
            muls = (constants.G.value * masses / (((dists**2+eps**2)**(1/2))**3))
            accelerations = (parts - pos) * np.reshape(muls,(1,) + muls.shape).T
        return np.sum(accelerations,axis=0),np.sum(potentials)

    @staticmethod
    def acc_func(positions,particles,masses,dists,eps=0):
        acc = np.zeros((positions.shape[0],3),dtype=float)
        phi = np.zeros(positions.shape[0],dtype=float)
        for idx,pos in enumerate(positions):
            temp_acc,temp_phi = DirectSum.acc_and_phi(pos,particles,masses,dists[idx],eps)
            phi[idx] = temp_phi
            acc[idx] = temp_acc
        return acc,phi

def evaluate(particles,masses,evaluate_at,eps=0):

    first = time.perf_counter()

    max_output_size = int((ctx.info["GL_MAX_TEXTURE_BUFFER_SIZE"]*2))/4
    n_particles = particles.shape[0]
    n_evals = evaluate_at.shape[0]
    max_input = int(max_output_size/n_particles)
    out_acc = np.zeros_like(evaluate_at,dtype=float)
    out_phi = np.zeros(len(evaluate_at),dtype=float)

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()
    part_buffer = ctx.buffer(particles.astype("f4").tobytes())
    
    start = 0
    n_batches = ceil(n_evals/max_input)
    times = np.zeros(n_batches,dtype=float)
    for i in range(n_batches):
        if n_evals - start < max_input:
            end = n_evals
        else:
            end = start + max_input
        dist,batch_time = do_batch(part_buffer,n_particles,evaluate_at[start:end])
        acc,phi = DirectSum.acc_func(evaluate_at[start:end],particles,masses,dist,eps)


        out_phi[start:end] = phi
        out_acc[start:end] = acc
        start = end
        times[i] = batch_time

    fbo.release()
    part_buffer.release()

    second = time.perf_counter()
    
    return out_acc,out_phi,{"eval_time":second-first,"n_batches":n_batches,"batch_times":times}