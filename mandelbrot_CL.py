import pyopencl as cl
import numpy as np
import time, matplotlib.pyplot as plt

KERNEL_SRC = """

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;
    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;
    double zr = 0.0, zi = 0.0;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""

def run_kernel(kernel, N, dtype):
    image = np.zeros((N, N), dtype=np.int32)
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    X0 = dtype(X_MIN)
    X1 = dtype(X_MAX)
    Y0 = dtype(Y_MIN)
    Y1 = dtype(Y_MAX)

    kernel(queue, (N, N), None, buf,
           X0, X1, Y0, Y1,
           np.int32(N), np.int32(MAX_ITER))
    queue.finish()

    t0 = time.perf_counter()
    kernel(queue, (N, N), None, buf,
           X0, X1, Y0, Y1,
           np.int32(N), np.int32(MAX_ITER))
    queue.finish()

    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, buf)
    queue.finish()

    return elapsed, image

ctx   = cl.create_some_context(interactive=False)

# device = ctx.devices[0]
# print("Device:", device.name)

# extensions = device.extensions.split()
# if "cl_khr_fp64" in extensions:
#     print("FP64 supported")
# else:
#     print("FP64 NOT supported")

queue = cl.CommandQueue(ctx)
prog  = cl.Program(ctx, KERNEL_SRC).build()

MAX_ITER = 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

t32_1024, img32_1024 = run_kernel(prog.mandelbrot, 1024, np.float32)
t32_2048, img32_2048 = run_kernel(prog.mandelbrot, 2048, np.float32)

t64_1024, img64_1024 = run_kernel(prog.mandelbrot_f64, 1024, np.float64)
t64_2048, img64_2048 = run_kernel(prog.mandelbrot_f64, 2048, np.float64)

print("1024x1024:")
print("f32:", t32_1024, "s")
print("f64:", t64_1024, "s")
print("ratio f64/f32:", t64_1024 / t32_1024)

print("\n2048x2048:")
print("f32:", t32_2048, "s")
print("f64:", t64_2048, "s")
print("ratio f64/f32:", t64_2048 / t32_2048)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("float32")
plt.imshow(img32_2048, cmap='hot', origin='lower')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("float64")
plt.imshow(img64_2048, cmap='hot', origin='lower')
plt.axis('off')
plt.show()

diff = np.abs(img64_2048 - img32_2048)
plt.imshow(diff, cmap='viridis')
plt.title("Difference (f64 - f32)")
plt.colorbar()
plt.show()