import math
import csv
import numpy as np
import matplotlib.pyplot as plt
#from plot_scal import get_speedup, get_efficiency
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_kernels



static_feature_list = [
    "input", "output",  # buffers: 0, 1
    "bitwise", "int_addsub", "int_mul",  # int: 2, 3, 4
    "f32.addsub", "f32.div", "f32.mul",  # f32: 5, 6, 7
    "f64.addsub", "f64.div", "f64.mul",  # f64: 8, 9, 10
    "load", "store", "other",   # mem and others: 11, 12, 13
    "coalescing"  # coalescing mem access: 14
]  # 15 features

comp_feature_list = [
    "total_instr"  # 15
]  # 1 additional computed feature

runtime_feature_list = [
    "size", "procs"  # runtime: 16, 17
]  # 2 additional runtime features


def dist(x, y):
    return cosine_similarity([x],[y])
    #return math.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


# improved static features with % of total instructions and +1 feature
def get_static_features_ext(bench_name):
    feature_vec = get_static_features(bench_name)
    # postprocessing instruction normalization
    total_instruction = float(0)
    for f_id in range(2, 14):  # from 2 to 13, we are skipping input, output and coalescing on purpose
        total_instruction = total_instruction + feature_vec[f_id]
    for f_id in range(2, 14):
        feature_vec[f_id] = feature_vec[f_id] / total_instruction
    feature_vec.append(total_instruction)
    #print(feature_vec)
    return feature_vec


def plot_static_features_ext(bench_names, features):
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(static_feature_list)))
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    # ax.set_xlim(0, np.sum(data, axis=1).max())
    sel_features = static_feature_list[2:13]
    for i, bench_name in enumerate(bench_names):
        # print(len(features[i]))
        widths = features[i][2:13]
        print('w', widths)
        starts = np.zeros(len(widths))
        for wid, w in enumerate(widths):
            if wid == 0:
                continue
            starts[wid] = starts[wid-1] + w + 0.001
        print('s', starts)
        ax.barh(bench_name, widths, left=starts, height=0.5, label=sel_features)  # , color=color)
        #centers = starts + widths / 2
        #r, g, b, _ = color
        #text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #for feature_id in range(2, 13):
        #    ax.text(x, y, str(int(c)), ha='center', va='center') #, color=text_color)
    #ax.legend(ncol=len(bench_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    plt.show()


def get_static_features(bench_name):
    feature_index = []
    feature_vec = []
    with open("staticFeatures.csv") as csv_file:
        feature_reader = csv.reader(csv_file)
        # extract static  using the same order in static_feature_list
        row1 = next(feature_reader)  # gets the first line
        for f in static_feature_list:
            feature_index.append(row1.index(f))
        # print(feature_list)
        # print(feature_index)
        for entry in feature_reader:
            if entry[0] == bench_name:
                for fid in feature_index:
                    feature_vec.append(float(entry[fid]))
    return feature_vec


def get_full_features(bench_name, sizes, procs):
    static_features = get_static_features_ext(bench_name)
    features = []
    for s_count, s_value in enumerate(sizes):
        for px in procs:
            feature_vec = static_features.copy()
            #feature_vec = []
            # feature encoding note: as we only have two sizes, we encode it either as 0 or 1
            feature_vec.append(float(s_count))
            # feature encoding note: we encode the log base2 of the number of processors
            log_proc = math.log2(float(px))
            feature_vec.append(log_proc)
            #print(feature_vec)
            features.append(feature_vec)
    return features


def minmax_normalization_by_feature(feature_matrix, feature_index):
    # search the max
    max_val = 0
    for feature_vec in feature_matrix:
        max_val = max(max_val, feature_vec[feature_index])
    # normalize
    if max_val == 0:
        raise AssertionError('Error: the feature has 0 as max value', feature_index)
    for feature_vec in feature_matrix:
        value = feature_vec[feature_index]
        feature_vec[feature_index] = value / max_val
    print("max", max_val, " for index", feature_index)


def get_full_feature_set_norm():
    return


def plot_distance_matrix(feature_vec, benchmarks):
    num_bench = len(benchmarks)
    distances = np.zeros((num_bench, num_bench))
    for x in range(num_bench):
        for y in range(num_bench):
            distances[x][y] = dist(feature_vec[x], feature_vec[y])
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(distances, cmap='gray_r')
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_yticks(np.arange(len(benchmarks)))
    # fix benchmark names
    benchmarks = [b.replace('kernel', 'k') for b in benchmarks]
    ax.set_xticklabels(benchmarks)
    ax.set_yticklabels(benchmarks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #for i in range(len(benchmarks)):
    #    for j in range(len(benchmarks)):
    #        text = ax.text(j, i, f'{distances[i, j]:.2f}', ha="center", va="center", color="w")
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5
    plt.tight_layout()
    plt.show()
    plt.savefig('features.png', bbox_inches='tight', dpi = 300)
    plt.savefig('features.pdf', bbox_inches='tight', dpi = 300)


def main():
    # only single kernel benchmark, no multi-task
    sel_bench = [
        "2dconv", 
        #"3dconv",
        "jacobi1d", "jacobi2d",
        "seidel",
        #"fdtd",

        "sobel3", "sobel5", "sobel7",
        "median",
        #"moldyn",

        #"mvt_kernel1", "mvt_kernel2",
        "matmul",
        "gemm", "gesummv",
        #"gramschmidt_kernel3",
        "syrk_kernel2",
        #"syr2k_kernel2",

         "fdtd2d_kernel4",
        "covariance_kernel1", "vecadd", "covariance_kernel2",
         "correlation_kernel2", 
         #"gramschmidt_kernel2", 
         "correlation_kernel3",

        "fdtd2d_kernel2", "fdtd2d_kernel3",
        "bicg_kernel1", "bicg_kernel2",
        "atax_kernel2", "atax_kernel3",

        #"gramschmidt_kernel1",
        #"syr2k_kernel1",

        "correlation_kernel1",
        "fdtd2d_kernel1",
        "syrk_kernel1",
        "atax_kernel1",
        "correlation_kernel5",

        "correlation_kernel4", "covariance_kernel3"

    ]
    sel_sizes = ["67108864", "268435456"]
    sel_procs = ["1", "2", "4", "8", "16", "32", "64"]
    #feat_num = 17  # 14+1 static + 2 runtime features
    bench_num = len(sel_bench)
    print("bench_num", bench_num)
    size_num = len(sel_sizes)
    proc_num = len(sel_procs)
    sample_num = bench_num * size_num * proc_num
    #print("feat_num", feat_num, " bench_num", bench_num, " size_num", size_num, " proc_num", proc_num)
    print("sample_num", sample_num)
    np.printoptions(precision=2, suppress=True)

    # static features
    static_features_matrix = []
    for bench in sel_bench:
        static_features_matrix.append(get_static_features_ext(bench))
        #print(bench, get_static_features_ext(bench))

    plot_static_features_ext(sel_bench, static_features_matrix)
    print("static feature matrix", static_features_matrix)

    # apply min-max normalization to selected features 0 (input), 1 (output), 15 (total_instr)
    # other feature are normalized by instruction %
    minmax_normalization_by_feature(static_features_matrix, 0)
    minmax_normalization_by_feature(static_features_matrix, 1)
    minmax_normalization_by_feature(static_features_matrix, 15)
    plot_distance_matrix(static_features_matrix, sel_bench)


if __name__ == "__main__":
    main()
