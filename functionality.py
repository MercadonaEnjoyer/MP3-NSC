import matplotlib.pyplot as plt

def plot_performance():
    impl = ["Naive", "Numpy", "Numba32", "Numba64",
            "Parallel", "Dask", "Cluster", "CL32", "CL64"]

    time_s = [10.545, 1.031, 0.062, 0.062,
              0.012, 0.050, 0.699, 0.0002, 0.0028]

    speedup = [1, 10.2, 169, 169,
               958, 195, 1397, 47931, 3766]

    # --- Time plot ---
    plt.figure()
    plt.bar(impl, time_s)
    plt.yscale('log')  # log scale helps compare very different times
    plt.title("Execution Time (log scale)")
    plt.xlabel("Implementation")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Speedup plot ---
    plt.figure()
    plt.bar(impl, speedup)
    plt.yscale('log')
    plt.title("Speedup (log scale)")
    plt.xlabel("Implementation")
    plt.ylabel("Speedup (x)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_performance()