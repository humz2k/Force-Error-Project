import struct
import moderngl
import numpy as np

ctx = moderngl.create_context(standalone=True)

prog = ctx.program(
    vertex_shader="""
    #version 330

    // Output values for the shader. They end up in the buffer.
    out float first;
    out float second;
    in float pos;
    in float check;

    void main() {
        first = pos;
        second = check;

    }
    """,
    # What out varyings to capture in our buffer!
    varyings=["first","second"],
)

a = np.reshape(np.arange(0,10),(1,10)).T

print(a)

buffer1 = ctx.buffer(a.astype('f4').tobytes())

outbuffer = ctx.buffer(reserve=20*2*(4))

print("THING")

vao = ctx.vertex_array(prog, [(buffer1, "1f", "pos"),(buffer1, "1f /i", "check")])

print("YEE")

vao.transform(outbuffer,first=0,instances=2)

print("OOH")

out = np.array(struct.unpack(str(2*20)+"f",outbuffer.read()))

print(out)

vao.release()
buffer1.release()
outbuffer.release()
