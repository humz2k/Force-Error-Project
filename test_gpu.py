import numpy as np
import struct
import moderngl
import time
import PyCC
from astropy import constants

ctx = moderngl.create_context(standalone=True)
print("MAX OUT LENGTH",int((ctx.info["GL_MAX_TEXTURE_BUFFER_SIZE"]*2)))
    

dists_prog = ctx.program(
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

prog = ctx.program(

    vertex_shader="""
    #version 330

    const float G =""" + str(constants.G.value) + """;

    out vec3 acc;
    out float phi;

    in vec3 part;
    in vec3 eval_pos;
    in float mass;
    in float eps;

    float d;
    float acc_mul;

    void main() {
        d = distance(part,eval_pos);
        if (d == 0){
            acc[0] = 0.;
            acc[1] = 0.;
            acc[2] = 0.;
            phi = 0.;
        } else{

            if (eps != 0){
                d = sqrt(pow(d,2) + pow(eps,2));
            }
            phi = (-1) * G * mass / d;
            acc_mul = G * mass / pow(d,3);
            acc[0] = (part[0] - eval_pos[0]) * acc_mul;
            acc[1] = (part[1] - eval_pos[1]) * acc_mul;
            acc[2] = (part[2] - eval_pos[2]) * acc_mul;
         }
    }

    """,

    varyings=["phi","acc"],

)

def phi_acc(particles,masses,eval_pos,eps):
    n_particles = particles.shape[0]
    n_evals = eval_pos.shape[0]
    
    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()

    part_buffer = ctx.buffer(particles.astype("f4").tobytes())
    mass_buffer = ctx.buffer(masses.astype("f4").tobytes())
    eval_buffer = ctx.buffer(eval_pos.astype("f4").tobytes())
    eps_buffer = ctx.buffer(np.array([[eps]]).astype("f4").tobytes())

    outbuffer = ctx.buffer(reserve=n_particles*n_evals*4*4)

    vao = ctx.vertex_array(prog, [(part_buffer, "3f", "part"),(eval_buffer, "3f /i", "eval_pos"), (mass_buffer, "1f", "mass"), (eps_buffer, "1f /r", "eps")])

    vao.transform(outbuffer,instances=n_evals)

    out = np.ndarray((n_particles*n_evals*4),"f4",outbuffer.read())
    x = np.sum(np.reshape(out[1::4],(n_evals,n_particles)),axis=1)
    y = np.sum(np.reshape(out[2::4],(n_evals,n_particles)),axis=1)
    z = np.sum(np.reshape(out[3::4],(n_evals,n_particles)),axis=1)
    phis = np.sum(np.reshape(out[::4],(n_evals,n_particles)),axis=1)
    acc = np.column_stack((x,y,z))

    vao.release()
    fbo.release()
    outbuffer.release()
    part_buffer.release()
    mass_buffer.release()
    eval_buffer.release()
    eps_buffer.release()

    return acc,phis

def get_dists(particles,f_size=4):
    n_particles = particles.shape[0]

    f_string = "f" + str(f_size)

    fbo = ctx.simple_framebuffer((1,1))
    fbo.use()


    buffer1 = ctx.buffer(particles.astype(f_string).tobytes())
    outbuffer = ctx.buffer(reserve=n_particles*n_particles*(f_size))

    vao = ctx.vertex_array(prog, [(buffer1, "3" + f_string, "pos"),(buffer1, "3f /i", "check")])

    vao.transform(outbuffer,instances=n_particles)

    a = outbuffer.read()

    out = np.ndarray((n_particles*n_particles),f_string,a)

    vao.release()
    buffer1.release()
    outbuffer.release()
    fbo.release()

    return np.reshape(out,(n_particles,n_particles))

n = 100
df = PyCC.Distributions.Uniform(r=100,n=n,p=100)

pos = df.loc[:,["x","y","z"]].to_numpy()
masses = df.loc[:,["mass"]].to_numpy()
evaluate_at = pos

acc,phi = phi_acc(pos,masses,evaluate_at,0)

out,stats = PyCC.evaluate(df= df, save=False)
acc2 = out.loc[:,["ax","ay","az"]].to_numpy()
phi2 = out.loc[:,"phi"].to_numpy()
print(np.mean(np.abs(acc2[0:n]-acc)))
print(np.mean(np.abs(phi2[0:n] - phi)))