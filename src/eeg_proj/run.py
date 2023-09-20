import tensorflow as tf
import numpy as np
import pandas as pd
from EEGModels import EEGNet
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    read_raw_bids,
)
from sklearn.model_selection import train_test_split

ELECTRODES = [
    "P8",
    "P7",
    "CP1",
    "CP2",
    "P6",
    "O2",
    "P4",
    "F4",
]


def run() -> None:
    data_dir = "/home/halli/Documents/ELE670/eeg-proj/ds004584"
    raw_paths: list[BIDSPath] = find_matching_paths(
        data_dir,
        extensions=[".set"],
    )
    df = pd.read_csv(
        f"{data_dir}/participants.tsv",
        sep="\t",
    )

    # Select based on MoCA score
    #   Below 26 => cognitive impairment
    moca_threshold = 26
    df["COGNITIVE_IMPAIRMENT"] = df.apply(lambda row: int(row.MOCA < moca_threshold), axis=1)

    # raw_paths = [raw_paths[0], raw_paths[-1]]
    # df = df.drop(df.index[1:-1])

    n_raws = len(raw_paths)
    groups = df["GROUP"].values

    raws = [
        read_raw_bids(
            path,
            verbose=False,
        )
        for path in raw_paths
    ]

    # Get eeg channel names (some subjects have extra chs)
    all_channels = {}
    for raw in raws:
        chs = raw.ch_names
        for ch in chs:
            if ch not in all_channels:
                all_channels[ch] = 1
            else:
                all_channels[ch] += 1
    drop_channels = []
    for k, v in all_channels.items():
        if v != n_raws:
            drop_channels.append(k)
    for raw in raws:
        raw.drop_channels(
            drop_channels,
            on_missing="ignore",
        )
    # Try with
    for raw in raws:
        raw.pick(ELECTRODES)
    times = []
    data = None
    y = []
    sample_rate = 128
    n_samples = sample_rate * 10
    n_patient_samples = 10
    for i, raw in enumerate(raws):
        raw.resample(sample_rate)
        raw_data, time = raw.get_data(stop=n_samples * n_patient_samples, return_times=True)
        # TODO: Try to use psd?
        # raw_data = raw.compute_psd()
        print(raw_data.shape)
        exit()
        times.append(time)
        group = 0
        if groups[i] == "PD":
            group = 1
        for _ in range(n_patient_samples):
            y.append(group)
        if i == 0:
            data = raw_data
        else:
            data = np.hstack([data, raw_data])
    n_channels = data.shape[0]
    print(len(y))
    y = pd.get_dummies(y)
    print(data.shape)

    X = data.reshape(n_raws * n_patient_samples, n_channels, n_samples, 1)
    X = X * 1000
    x_train, x_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y)
    x_val, y_val = x_tmp, y_tmp
    # x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True),
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=80),
        tf.keras.callbacks.TensorBoard(log_dir="log", histogram_freq=1),
    ]
    model = EEGNet(
        2,
        Chans=n_channels,
        Samples=n_samples,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="SpatialDropout2D",
        # dropoutType="Dropout",
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist = model.fit(
        X,
        y,
        batch_size=16,
        epochs=50,
        verbose=2,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    import matplotlib.pyplot as plt

    f, ax = plt.subplots(1, 2, figsize=(18, 8))

    ax[0].plot(hist.history['accuracy'])  # Type your solution
    ax[0].plot(hist.history['val_accuracy'])  # Type your solution
    ax[0].set_title('model accuracy')
    ax[0].set_label('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='upper left')

    ax[1].plot(hist.history['loss'])  # Type your solution
    ax[1].plot(hist.history['val_loss'])  # Type your solution
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'val'], loc='upper left')
    plt.show()
