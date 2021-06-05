import matplotlib.pyplot as plt
import numpy as np
import math
from plot_scal import plot_scalability, get_scalability_data



def plot_multi_prediction(bench, size, single_kernels, single_size_weights, comm_overhead):
    num_flt = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    num_str = ["1", "2", "4", "8", "16", "32", "64"]
    (x, y, y_min, y_max, s, e) = get_scalability_data(bench, size)

    # compute raw scalability prediction
    #print (s)
    #pred1 = [0.0] * len(num_str)
    pred1 = np.zeros(len(num_str))
    for idx, kernel in enumerate(single_kernels):
        w = single_size_weights[idx]
        #print('kernel:', kernel, ' w:', w)
        (t1, t2, t3, t4, sk, t5) = get_scalability_data(kernel, size)
        sk = np.array(sk)
        #print(sk)
        pred1 = pred1 + sk * w

    #print('pred1:', pred1)
    # compute scalability prediction corrected with communication overhead
    pred2 = pred1 * (1.0 - comm_overhead)
    #print('pred2:', pred2)
    l1pred2 = np.sum(np.power((s - pred1), 2))
    l2pred2 = np.sum(np.power((s - pred2), 2))
    print('L2 norms - pred1:', l1pred2, 'pred2:', l2pred2)

    fig, (ax1) = plt.subplots(1, 1, figsize=(3.5, 3.5), tight_layout=True)

    width = 0.20
    x = np.arange(len(num_str))
    cm = plt.get_cmap('gray')
    ax1.bar(x , s,     label='measured speedup',        width=width, color=cm(0))
    ax1.bar(x + width, pred1, label='raw prediction',        width=width, color=cm(0.55))
    ax1.bar(x + width*2, pred2, label='prediction with comm.', width=width, color=cm(0.75))
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_str)
    ax1.legend(labels=['measured speedup', 'raw prediction', 'prediction with comm.'], loc="upper left")
    ax1.set_ylim([0, 55])
    title = str(bench) + str(" ") + str(int(math.sqrt(int(size)))) + "^2"
    ax1.set_title(title)
    #ax1.set_title(bench + str(" ") + str(math.sqrt(size)) ) + str("^2"))
    plt.show()
    filename = str(bench) + str("_") + str(int(math.sqrt(int(size))))
    fig.savefig(filename)


def main():
    sel_bench = [
        # multi-task
        #"2mm", "matmulchain", "mvt", "syrk", "bicg", "fdtd2d",
        # individual tasks
        #"matmul",
        #"mvt_kernel1", "mvt_kernel2",
        "syrk_kernel1", "syrk_kernel2",
        #"bicg_kernel1", "bicg_kernel2",
        #"fdtd2d_kernel1", "fdtd2d_kernel2", "fdtd2d_kernel3", "fdtd2d_kernel4"
    ]
    sel_sizes = ["67108864", "268435456"]
    # sel_sizes = ["8192", "16384"]
    #    for bench in sel_bench:
    #    for size in sel_sizes:
    #        print("plotting", bench, size)
    #        plot_scalability(bench, size)
    for size in  sel_sizes:
        plot_multi_prediction("syrk", size, ["syrk_kernel1", "syrk_kernel2"], [0.03, 0.97], 0.001)
        plot_multi_prediction("2mm", size, ["matmul", "matmul"], [0.5, 0.5], 0.117)
        plot_multi_prediction("matmulchain", size, ["matmul", "matmul", "matmul"], [0.33, 0.33, 0.33], 0.763)
        plot_multi_prediction("bicg", size, ["bicg_kernel1", "bicg_kernel2"], [0.5, 0.5], 0.0)
        plot_multi_prediction("fdtd2d", size, ["fdtd2d_kernel1", "fdtd2d_kernel2", "fdtd2d_kernel1", "fdtd2d_kernel4"], [0.25, 0.25, 0.25, 0.25], 0.112)



if __name__ == "__main__":
    main()