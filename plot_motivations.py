import numpy as np
import matplotlib.pyplot as plt

from plot_scal import get_scalability_data


def plot_scalability(bench, size1, size2):
    num_flt = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    num_str = ["1", "2", "4", "8", "16", "32", "64"]
    (x1, y1, t1, tt1, s1, e1) = get_scalability_data(bench, size1)
    (x2, y2, t2, tt2, s2, e2) = get_scalability_data(bench, size2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3.5), tight_layout=True)

    # fig 1, runtime
    ax1.set_xscale("log", basex=2)
    ax1.set_xticks(num_flt)
    ax1.set_xticklabels(num_str)
    ax1.set_title(bench)
    ax1.plot(x1, y1, '-', color='tab:orange')
    ax1.plot(x2, y2, '-', color='tab:red')

    # fig 2, speedup
    x = np.arange(len(x1))  # the label locations
    width = 0.35  # the width of the bars
    ax2.bar(x - width / 2, s1, width, label='8192^2', color='black')
    ax2.bar(x + width / 2, s2, width, label='16384^2', color='gray')
    ax2.set_xticklabels(num_str)
    ax2.set_title("Speedup")

    # fig 3, parallel efficiency
    ax3.axhline(1.0, color="gray")
    ax3.set_xscale("log", basex=2)
    ax3.set_xticks(num_flt)
    ax3.set_xticklabels(num_str)
    ax3.plot(x1, e1, '-',  color='tab:orange')
    ax3.plot(x2, e2, '-',  color='tab:red')
    ax3.set_ylim(0, 1.05)
    ax3.set_title("par. efficiency")
    plt.show()

    # new figure, only speedup
    fig, (ax) = plt.subplots(1, 1, figsize=(4.0, 3.5), tight_layout=True)
    ax.bar(x - width / 2, s1, width, label='8192^2', color='black')
    ax.bar(x + width / 2, s2, width, label='16384^2', color='gray')
    ax.set_ylim(0, 40)
    ax.set_xticks(x)
    ax.set_xticklabels(num_str)
    ax.set_ylabel('Speedup', fontsize=16)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    ax.set_title(bench, fontsize=16)
    ax.legend(loc='upper left')
    plt.show()
    #plt.savefig('mot_matmulchain.png', bbox_inches='tight', dpi = 300)


sel_bench = ["matmul", "2mm", "matmulchain"] #, 
sel_sizes = ["67108864", "268435456"]
# sel_sizes = ["8192", "16384"]
for bench in sel_bench:
    plot_scalability(bench, sel_sizes[0], sel_sizes[1])


