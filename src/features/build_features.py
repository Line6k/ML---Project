import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df[["acc_x", "label"]].boxplot(by="label", figsize=(20, 10))

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))

# Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py


def plot_binary_outliers(dataset, col, outlier_col, reset_index):

    dataset = dataset.copy()

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()
    plt.xlabel("samples")
    plt.ylabel("value")

    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
        label="no outlier" + col,
    )

    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
        label="outlier" + col,
    )

    plt.legend(
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
        frameon=True,
        facecolor="white",
        framealpha=1,
    )

    plt.show()


# --------------------------------------------------------------
# Interquartile range
# --------------------------------------------------------------


def mark_outliers_iqr(dataset, col):

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


col = "acc_x"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# We need to check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)

# Rest acc data might pose a problem


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


for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# The rest data doesn't seem to be normally distributed,
# so there are more than a few outliers


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------


def mark_outliers_lof(dataset, columns, n=20):
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# Loop over all columns

dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------
label = "bench"

for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

dataset, outliers, X_scores = mark_outliers_lof(
    df[df["label"] == label], outlier_columns
)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)
dataset[dataset["gyr_z_outlier"]]

dataset.loc[dataset["gyr_z_outlier"], col] = np.nan

# Create a loop

outlier_removed_df = df.copy()  # I will not make the mistake of leaving this out again
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_iqr(df[df["label"] == label], col)
        dataset.loc[dataset[col + "_outlier"], col] = np.nan
        outlier_removed_df.loc[(outlier_removed_df["label"] == label), col] = dataset[
            col
        ]

        n_outliers = len(df) - len(outlier_removed_df[col].dropna())
        print(f"remove {n_outliers} from {col} for {label}")


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
