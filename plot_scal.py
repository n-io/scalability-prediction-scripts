import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np


def get_scalability_data(bench, size):
    x = []
    y_min = []
    y_max = []
    y = []
    num = []
    with open("benchmark-marconi-updated.csv") as csv_file:
        bench_reader = csv.reader(csv_file)
        # data processing
        for entry in bench_reader:
            if entry[0] == bench and entry[2] == size:
                xs = float(entry[3])
                ys = entry[4:8]
                ys = [t for t in ys if t.strip()]  # remove empty string
                ys = [float(i) for i in ys]  # cast to float
                if not ys:  # skip if empty
                    print("empty", ys)
                    break
                x.append(xs)
                y.append(statistics.median(ys))
                y_max.append(max(ys))
                y_min.append(min(ys))
                num.append(int(entry[3]))
    # calculate speedup from y
    s = []
    e = []
    #print(num)
    if num and num[0] == 1:
        baseline = y[0]
        for idx, val in enumerate(y):
            speedup = (baseline / val)
            efficiency = speedup / float(num[idx])
            s.append(speedup)
            e.append(efficiency)
    #print("  runtime", y)
    #print("  speedup", s)
    #print("  p.eff. ", e)
    return x, y, y_min, y_max, s, e


def get_speedup(bench, size):
    (x, y, y_min, y_max, s, e) = get_scalability_data(bench, size)
    return s


def get_efficiency(bench, size):
    (x, y, y_min, y_max, s, e) = get_scalability_data(bench, size)
    return e


def plot_scalability(bench, size):
    num_flt = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    num_str = ["1", "2", "4", "8", "16", "32", "64"]
    (x, y, y_min, y_max, s, e) = get_scalability_data(bench, size)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3.5), tight_layout=True)
    # fig 1, runtime
    ax1.set_xscale("log", basex=2)
    ax1.set_xticks(num_flt)
    ax1.set_xticklabels(num_str)
    ax1.set_title(bench + " " + size)
    ax1.plot(x, y, '-')
    ax1.fill_between(x, y_min, y_max, alpha=0.2)
    ax1.plot(x, y, 'o', color='tab:red')
    # fig 2, speedup
    ax2.bar(num_str, s)
    ax2.set_xticklabels(num_str)
    ax2.set_title("speedup")
    # fig 3, parallel efficiency
    ax3.axhline(1.0, color="gray")
    ax3.set_xscale("log", basex=2)
    ax3.set_xticks(num_flt)
    ax3.set_xticklabels(num_str)
    ax3.plot(x, e, '-')
    ax3.set_ylim(0, 1.05)
    ax3.set_title("par. efficiency")
    plt.show()
    plt.savefig('scalability.pdf', bbox_inches='tight', dpi = 300)


def main():
    sel_bench = [
        "2dconv",  # "3dconv",
        "jacobi1d", "jacobi2d",
        "seidel",
        # "fdtd",

        "sobel3", "sobel5", "sobel7",
        "median",
        # "moldyn",

        # "mvt_kernel1",
        # "mvt_kernel2",
        "matmul",
        "gemm", "gesummv",
        # "gramschmidt_kernel3",
        "syrk_kernel2",
        # "syr2k_kernel2",

        "fdtd2d_kernel4",
        "covariance_kernel1", "vecadd", "covariance_kernel2",
        "correlation_kernel2",
        # "gramschmidt_kernel2",
        "correlation_kernel3",

        "fdtd2d_kernel2", "fdtd2d_kernel3",
        "bicg_kernel1", "bicg_kernel2",
        "atax_kernel2", "atax_kernel3",

        # "gramschmidt_kernel1",
        # "syr2k_kernel1",

        "correlation_kernel1",
        "fdtd2d_kernel1",
        "syrk_kernel1",
        "atax_kernel1",
        "correlation_kernel5",

        "correlation_kernel4", "covariance_kernel3"
    ]
    sel_sizes = ["67108864", "268435456"]
    # sel_sizes = ["8192", "16384"]
    for bench in sel_bench:
        for size in sel_sizes:
            print("plotting", bench, size)
            plot_scalability(bench, size)


if __name__ == "__main__":
    main()

