import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from plot_scal import get_speedup, get_efficiency, get_scalability_data
from feature_analysis import get_static_features_ext, get_full_features, minmax_normalization_by_feature
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


def plot_prediction(sel_bench, sel_size, sel_index, speedup, predicted):
    plot("ERROR: this function has a bug!")
    num_flt = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    num_str = ["1", "2", "4", "8", "16", "32", "64"]
    num_bench = 12  # len(sel_bench) number of bench in the training data
    num_size = len(sel_size)
    num_proc = len(num_flt)
    group = np.arange(num_proc)
    width = 0.35
    for bx, bench in enumerate(sel_bench):
        base_index = sel_index[bx] * num_size * num_proc  #num_bench
        fig, temp = plt.subplots(1, num_size, tight_layout=True)  # figsize=(9, 3.5)
        axes = fig.axes
        for sx, size in enumerate(sel_size):
            ax = axes[sx]
            index = base_index + sx * num_size
            ax.bar(group - width/2, speedup[index:index+num_proc],   width, label="Measured")
            ax.bar(group + width/2, predicted[index:index+num_proc], width, label="Predicted")
            ax.set_ylabel("Speedup")
            ax.set_title(bench + " " + size)
            #ax.set_xscale("log", basex=2)
            ax.set_xticks(group)
            ax.set_xticklabels(num_str)
            ax.legend()
    plt.show()


def training_validation(model, X, Y, labels):
    num_flt = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    num_str = ["1", "2", "4", "8", "16", "32", "64"]
    num_proc = len(num_flt)
    group = np.arange(num_proc)
    width = 0.20
    block = int(len(Y) / len(labels))
    print("block = ", block, "len(labels) = ", len(labels), "len(Y) = ", len(Y), "len(X) = ", len(X))
    
    predicted_y = np.zeros(len(Y), dtype=float)
        
    for i in range(len(labels)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        left = block * i
        right = block * (i+1)
        x_test = X[left:right,:]
        y_test = Y[left:right]
        
        x_train = np.delete(X, slice(left, right), 0)
        y_train = np.delete(Y, slice(left, right))
        
        model.fit(x_train, y_train)
        
        prediction = abs(model.predict(x_test)) # prediction on the training data
        
        predicted_y[left:right] = prediction
        
        #model.score(prediction, y_test)
        
        #print("x_test = ", x_test)
        #print("y_test = ", y_test)
        #print("prediction = ", prediction)
        
        fig, ax = plt.subplots(1, 1, tight_layout=True)  # figsize=(9, 3.5)
        ax.set_ylim(0, 40)
        ax.bar(group - width / 2, y_test, width, label="Measured", color='black')
        ax.bar(group + width / 2, prediction, width, label="Predicted", color='gray')
        ax.set_ylabel("Speedup",fontsize=16)
        ax.set_title(labels[i], fontsize=16)
        # ax.set_xscale("log", basex=2)
        ax.set_xticks(group)
        ax.set_xticklabels(num_str)
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16, rotation=0)
        ax.legend()
        filename = "single_" + labels[i][:-2].replace(' ','_')
        if "kernel" in filename:
            continue
        plt.savefig(filename)
        #plt.show()
   
    print("mean_squared_error = ", mean_squared_error(Y, predicted_y))
    print("mean_squared_log_error = ", mean_squared_log_error(Y, predicted_y))
    print("mean_absolute_percentage_error = ", mean_absolute_percentage_error(Y, predicted_y))

def cross_validation(X, Y, labels):
    model = SVR(C=1000000, epsilon=1.0, gamma='scale')
    fold = KFold(n_splits=len(labels), shuffle=False)
    fig, temp = plt.subplots(6, 4, figsize=(24, 16))
    print("fold num:", len(labels))
    num_flt = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    num_str = ["1", "2", "4", "8", "16", "32", "64"]
    num_proc = len(num_flt)
    group = np.arange(num_proc)
    axes = fig.axes
    ax_id = 0
    width = 0.35
    for train_index, test_index in fold.split(X):
        print(ax_id)
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)  # prediction on the training data
        ax = axes[ax_id]
        ax.bar(group - width / 2, y_test, width, label="Measured")
        ax.bar(group + width / 2, prediction, width, label="Predicted")
        ax.set_ylabel("Speedup")
        ax.set_title(labels[ax_id])
        # ax.set_xscale("log", basex=2)
        ax.set_xticks(group)
        ax.set_xticklabels(num_str)
        ax.legend()
        ax_id = ax_id + 1
    plt.show()

def usage():
    print('Use: scal_model.py <model choice>')
    print('SVR = 0, RF = 1, KNN = 2')

def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit()
    
    sel_bench = [
        "2dconv",  # "3dconv",
        "jacobi1d", "jacobi2d",
        "seidel",
        #"fdtd",

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
        #print('Bench = ', bench) 
        #print('features = ', features)
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

    np.printoptions(threshold=np.inf)
    #print("X", X.shape)
    print("X BEFORE NORMALIZATION", X.shape)
    np.set_printoptions(threshold=np.inf)
    #print (X)

    # apply min-max normalization to selected features:
    # 0 (input), 1 (output), 15 (total_instr), 17 (proc_num)
    minmax_normalization_by_feature(X, 0)
    minmax_normalization_by_feature(X, 1)
    minmax_normalization_by_feature(X, 14)
    minmax_normalization_by_feature(X, 17)
    np.set_printoptions(threshold=np.inf)
    
    print("X AFTER NORMALIZATION", X.shape)
    np.set_printoptions(precision=2, suppress=True, linewidth=300)
    #print("X = ", X)
    # print("Y", Y.shape)
    # print(Y)

    pca = PCA(n_components=6)
    pca.fit(X)
    XT = pca.transform(X)
    #XT = X

    # modeling on training data
    tested_models = [
        # SVR(C=1.0, epsilon=0.1),
        # SVR(C=1.0, epsilon=0.01),
        # SVR(C=1.0, epsilon=1.0),
        #SVR(C=1000, epsilon=0.1),  # score: 0.920
        #SVR(C=100, epsilon=0.1),  # score: 0.812
        #  SVR(C=10, epsilon=0.1),
        #SVR(C=1000, epsilon=1.0),  # score: 0.901
        #  SVR(C=1000, epsilon=10.0),
        #  SVR(C=1000, epsilon=100.0),
        #SVR(C=10000, epsilon=1.0),
        # SVR(C=10000, epsilon=10.0),
        # SVR(C=10000, epsilon=100.0),
        #SVR(C=100000, epsilon=1.0, gamma='scale'),
        #SVR(C=100000, epsilon=1.0, gamma='auto'),  # score: 0.961
        SVR(C=1000000, epsilon=1.0, gamma='auto'),  # score: 0.968 BEST -> pca (10) 0.970 / pca (8) 0.971 / / pca (6) 0.975
        #SVR(kernel='linear', C=1000000, epsilon=1.0)
        #SVR(C=1000000, epsilon=1.0, gamma='scale')  # score: 0.788
        RandomForestRegressor(n_estimators=100, criterion="mse", max_features ="auto", bootstrap = False, max_depth=7, min_samples_split=2, random_state=0), # Best MSE with max_depth = 7
        KNeighborsRegressor(n_neighbors=3) # Best MSE for n_neighbors = 3, tested from 1 to 10
    ]
    
    model_choice = int(sys.argv[1])
    
    #for im, model in enumerate(tested_models):
    model = tested_models[model_choice]
    
    model.fit(XT, Y)
    prediction = model.predict(XT)  # prediction on the training data
    #prediction = abs(prediction)
    #print(im, "C", model.C, "eps", model.epsilon)
    print("score:", model.score(XT, Y))
    #print("gamma:", model.gamma)
    plt.plot(prediction, Y, '.', color='gray')
    plt.xlabel("Predicted speedup", fontsize=16)
    plt.ylabel("Measured speedup", fontsize=16)
    # plt.title(type(model).__name__ + " C:" + str(model.C) + " eps:" + str(model.epsilon))
    plt.xlim(0,64)
    plt.ylim(0, 64)
    plt.xticks (fontsize=16, rotation=0)
    plt.yticks (fontsize=16, rotation=0)
    #plt.axes().set_xlim(plt.ylim()) # ensure the plot is not skewed
    plt.show()
    plt.savefig("correlation.png")

    #print ("labels = ", labels)
    #print ("Y = ", Y)
    training_validation(model, XT, Y, labels)

    # modeling with leave-one-out cross validation
    #cross_validation(XT, Y, labels)
    return

if __name__ == "__main__":
    main()

