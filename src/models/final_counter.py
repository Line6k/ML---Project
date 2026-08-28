import sys
import os
import pandas as pd
import pickle
import numpy as np
from glob import glob
import numpy as np
from glob import glob
import scipy
import math
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

csv_path = "../../data/raw/MetaMotion/A-row-heavy_MetaWear_2019-01-14T15.04.06.123_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
model_path = "../../models/rf_model"


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        clean_filename = os.path.basename(f)
        participant = clean_filename.split("-")[0]
        label = clean_filename.split("-")[1]
        category = clean_filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


def mark_outliers_chauvenet(dataset, col, C=2):

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    dataset = dataset.copy()
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    deviation = abs(dataset[col] - mean) / std

    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    for i in range(0, len(dataset.index)):
        prob.append(
            1.0
            - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
        )
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def count_reps_and_exercise(csv_path, model_path):

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    files = sorted(glob(csv_path))
    acc_df, gyr_df = read_data_from_files(files)

    data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

    data_merged.columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "label",
        "category",
        "set",
    ]

    # Accelerometer:    12.500HZ
    # Gyroscope:        25.000Hz

    sampling = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set": "last",
    }

    data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

    data_merged.columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "label",
        "category",
        "set",
    ]

    data_merged[:].resample(rule="200ms").apply(sampling)

    # split by day

    days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

    data_resampled = pd.concat(
        [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
    )

    data_resampled["set"] = data_resampled["set"].astype("int")

    df = data_resampled

    outlier_columns = list(df.columns[:6])

    outlier_removed_df = df.copy()
    for col in outlier_columns:
        for label in df["label"].unique():
            dataset = mark_outliers_chauvenet(df[df["label"] == label], col)

            dataset.loc[dataset[col + "_outlier"], col] = np.nan

            outlier_removed_df.loc[(outlier_removed_df["label"] == label), col] = (
                dataset[col]
            )

            n_outliers = len(dataset) - len(dataset[col].dropna())

    df = outlier_removed_df

    predictor_columns = list(df.columns[:6])

    for col in predictor_columns:
        df[col] = df[col].interpolate()

    for s in df["set"].unique():
        start = df[df["set"] == s].index[0]
        stop = df[df["set"] == s].index[-1]

        duration = stop - start

        df.loc[(df["set"] == s), "duration"] = duration.seconds

    duration_df = df.groupby(["category"])["duration"].mean()

    duration_df.iloc[0] / 5
    duration_df.iloc[1] / 10

    df_lowpass = df.copy()
    LowPass = LowPassFilter()

    fs = 1000 / 200
    cutoff = 1.4

    for col in predictor_columns:
        df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
        df_lowpass[col] = df_lowpass[col + "_lowpass"]
        del df_lowpass[col + "_lowpass"]

    df_pca = df_lowpass.copy()
    PCA = PrincipalComponentAnalysis()
    pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

    df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

    df_squared = df_pca.copy()

    acc_r = (
        df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
    )
    gyr_r = (
        df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2
    )

    df_squared["acc_r"] = np.sqrt(acc_r)
    df_squared["gyr_r"] = np.sqrt(gyr_r)

    df_temporal = df_squared.copy()
    NumAbs = NumericalAbstraction()

    predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

    ws = int(1000 / 200)

    df_temporal_list = []

    for s in df_temporal["set"].unique():
        subset = df_temporal[df_temporal["set"] == s].copy()

        for col in predictor_columns:
            subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
            subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
        df_temporal_list.append(subset)

    df_temporal = pd.concat(df_temporal_list)

    df_freq = df_temporal.copy().reset_index()
    FreqAbs = FourierTransformation()

    fs = int(1000 / 200)
    ws = int(2800 / 200)

    df_freq_list = []
    for s in df_freq["set"].unique():
        subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
        subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
        df_freq_list.append(subset)

    df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

    df_freq = df_freq.dropna()

    df_freq = df_freq.iloc[::2]

    df_cluster = df_freq.copy()

    cluster_columns = ["acc_x", "acc_y", "acc_z"]
    k_values = range(2, 10)
    inertias = []

    for k in k_values:
        subset = df_cluster[cluster_columns]
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        cluster_labels = kmeans.fit_predict(subset)
        inertias.append(kmeans.inertia_)

    kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
    subset = df_cluster[cluster_columns]
    df_cluster["cluster"] = kmeans.fit_predict(subset)

    expected_features = model.feature_names_in_
    X_new = df_cluster[expected_features]

    df_cluster["predicted_exercise"] = model.predict(X_new)

    final_predictions = {}
    for s in df_cluster["set"].unique():
        subset = df_cluster[df_cluster["set"] == s]

        # Get the most common prediction in this set
        majority_vote = subset["predicted_exercise"].mode()[0]
        final_predictions[s] = majority_vote

        print(f"Set {s} classified as: {majority_vote}")

    return final_predictions


print(count_reps_and_exercise(csv_path, model_path))
