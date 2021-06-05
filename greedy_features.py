import math

import numpy as np
from sklearn.svm import SVR

from feature_analysis import get_full_features, minmax_normalization_by_feature
from plot_scal import get_speedup


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
    sel_procs = ["1", "2", "4", "8", "16", "32", "64"]
    feat_num = 18  # 15+1 static + 2 runtime features
    bench_num = len(sel_bench)
    size_num = len(sel_sizes)
    proc_num = len(sel_procs)
    sample_num = bench_num * size_num * proc_num
    print("feat_num", feat_num, " bench_num", bench_num, " size_num", size_num, " proc_num", proc_num)
    print("sample_num", sample_num)
    np.printoptions(precision=2, suppress=True)

    print("full features + speedup values")
    X = np.zeros((sample_num, feat_num), dtype=float)
    Y = np.zeros(sample_num, dtype=float)
    labels = []
    sample_x = 0
    sample_y = 0
    for bx, bench in enumerate(sel_bench):
        # print("features for", bench)
        features = get_full_features(bench, sel_sizes, sel_procs)
        # print(features)
        for fx, f_vec in enumerate(features):
            X[sample_x, :] = f_vec
            sample_x = sample_x + 1
        # print("speedup for ", bench)
        for sx, size in enumerate(sel_sizes):
            labels.append(bench + " " + str(int(math.sqrt(float(size)))) + "^2")
            for px, speedup_value in enumerate(get_speedup(bench, size)):
                Y[sample_y] = speedup_value
                sample_y = sample_y + 1
                # print(speedup_value)
        if not sample_x == sample_y:
            raise AssertionError('Error: the feature vector does not match the speedup vector', sample_x, sample_y)

    np.printoptions(precision=2, suppress=True)
    print("X", X.shape)
    print(X)

    # apply min-max normalization to selected features:
    # 0 (input), 1 (output), 15 (total_instr), 17 (proc_num)
    minmax_normalization_by_feature(X, 0)
    minmax_normalization_by_feature(X, 1)
    minmax_normalization_by_feature(X, 14)
    minmax_normalization_by_feature(X, 17)
    print("X AFTER NORMALIZATION", X.shape)
    np.set_printoptions(precision=2, suppress=True, linewidth=300)
    print(X)
    # print("Y", Y.shape)
    # print(Y)

    base_model = SVR(C=1000000, epsilon=1.0, gamma='auto')
    base_model.fit(X, Y)
    print("base model score:", base_model.score(X, Y))


    # ranking here
    X = np.delete(X, -1, 1)  # delete procs num
    X = np.delete(X, -1, 1)  # delete input size
    X = np.delete(X, 8, 1)  # delete f32.mul
    X = np.delete(X, 2, 1)  # delete input

    # let's remove one feature only and check the score
    max = 0
    max_id = 0
    for i in range(np.shape(X)[1]):
        XR = np.copy(X)
        XR = np.delete(XR, i, 1)  # delete second row of A
        Rmodel = SVR(C=1000000, epsilon=1.0, gamma='auto')
        Rmodel.fit(XR, Y)
        prediction = Rmodel.predict(XR)  # prediction on the training data
        # prediction = abs(prediction)
        loss_score = 1.0 - Rmodel.score(XR, Y)
        print("removed feature:", i, " out of ", np.shape(XR)[1], " -> score:", Rmodel.score(XR, Y), " Score loss:",loss_score )
        if loss_score > max:
            max = loss_score
            max_id = i
    print("max id", max_id, " - loss score:",max)


if __name__ == "__main__":
    main()
