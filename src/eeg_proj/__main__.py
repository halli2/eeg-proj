import sys
from argparse import ArgumentParser
from typing import Optional

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    read_raw_bids,
)
from scipy import signal

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

PD = 0
CONTROL = 1
MOCA_THRESHOLD = 26
FS = 500.0


def process_raw(raw: mne.io.Raw, info: pd.Series) -> mne.io.Raw:
    """Keep only a select few electrodes and set channel positions correctly"""
    raw.pick(ELECTRODES)

    ch_path: BIDSPath = info["CHPATH"]
    df = pd.read_csv(
        ch_path.fpath,
        sep="\t",
        index_col="name",
    )
    coords = {}
    for ch in ELECTRODES:
        xyz = df.loc[ch]
        coords[ch] = [-xyz["y"] / 10, xyz["x"] / 10, xyz["z"] / 10]
    mont = mne.channels.make_dig_montage(coords)
    raw.set_montage(mont)
    return raw


class Patient:
    def __init__(self, info: pd.Series, fmin: float = 0, fmax: float = 120):
        self.group: str = info.get("GROUP")
        self.moca: str = "COGNITIVE NORMAL"
        if info.get("MOCA") < MOCA_THRESHOLD:
            self.moca = "COGNITIVE IMPAIRED"
        raw: mne.io.Raw = read_raw_bids(
            info.get("BIDSPath"),
            verbose=False,
        )
        self.raw: mne.io.Raw = process_raw(raw, info)
        self.psd: mne.time_frequency.Spectrum = self.raw.compute_psd(fmin=fmin, fmax=fmax, method="welch")


Patients = list[Patient]


def load_data(data_dir: str) -> list[Patient]:
    raw_paths: list[BIDSPath] = find_matching_paths(
        data_dir,
        extensions=[".set"],
    )
    chs = find_matching_paths(
        data_dir,
        suffixes="electrodes",
        extensions=[".tsv"],
    )
    df = pd.read_csv(
        f"{data_dir}/participants.tsv",
        sep="\t",
    )
    df["BIDSPath"] = raw_paths
    df["CHPATH"] = chs

    # df = df.loc[[0, 20, 110, 111, 148]]  # TODO: Remove
    patients = [Patient(row) for _, row in df.iterrows()]
    return patients


def calculate_psd_metrics(patients: Patients) -> pd.DataFrame:
    """Calculates  average and std based on groups"""
    df = pd.DataFrame()
    df["freqs"] = patients[0].psd.freqs
    for channel in ELECTRODES:
        control = []
        pd_group = []
        for patient in patients:
            # TODO: Check diff between these?
            if patient.group == "PD":
                # if patient.moca == "COGNITIVE IMPAIRED":
                pd_group.append(patient.psd.get_data([channel]))
            else:
                control.append(patient.psd.get_data([channel]))
        control = np.array(control)
        pd_group = np.array(pd_group)
        control = 10 * np.log10(control)
        pd_group = 10 * np.log10(pd_group)
        df[f"{channel}_control_avg"] = np.average(control, axis=0)[0]
        df[f"{channel}_control_std"] = np.std(control, axis=0)[0]
        df[f"{channel}_pd_avg"] = np.average(pd_group, axis=0)[0]
        df[f"{channel}_pd_std"] = np.std(pd_group, axis=0)[0]
    return df


def plot_psd_metrics(metrics: pd.DataFrame, channel: str, ax: Optional[plt.Axes] = None) -> None:
    if ax is None:
        _, ax = plt.subplots()
    freqs = metrics["freqs"]
    avg = metrics[f"{channel}_control_avg"]
    std = metrics[f"{channel}_control_std"]
    pd_avg = metrics[f"{channel}_pd_avg"]
    pd_std = metrics[f"{channel}_pd_std"]

    ax.plot(freqs, avg, label="control")
    ax.plot(freqs, pd_avg, label="pd")
    # ax.plot(freqs, pd_avg, label="ci")
    ax.fill_between(freqs, avg + std, avg - std, color="C0", alpha=0.3)
    ax.fill_between(freqs, pd_avg + pd_std, pd_avg - pd_std, color="C1", alpha=0.3)
    ax.set_xlim(1, 50)
    ax.set_xlabel("Freq [Hz]")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 9, 15, 30])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylabel("μV^2 / Hz [dB]")
    ax.set_title(f"{channel}")
    ax.legend(loc="upper right")


def save_psd_metric_plots(metrics: pd.DataFrame) -> None:
    for electrode in ELECTRODES:
        f, ax = plt.subplots()
        plot_psd_metrics(
            metrics,
            electrode,
            ax,
        )
        plt.savefig(f"result/psd_{electrode}.svg")


def plot_all_psd(metrics: pd.DataFrame) -> None:
    f, ax = plt.subplots(3, 3)
    index = 0
    for i in range(3):
        for j in range(3):
            if index == len(ELECTRODES):
                break
            plot_psd_metrics(
                metrics,
                ELECTRODES[index],
                ax[i, j],
            )
            index += 1
    plt.show()


def plot_and_save(y: npt.NDArray, file: str, x: Optional[npt.NDArray] = None, ax: Optional[plt.Axes] = None) -> None:
    if ax is None:
        _, ax = plt.subplots()
    if x is None:
        x = np.linspace(0, 1, len(y))
    ax.plot(x, y)
    plt.savefig(file)


def normalize_energy(raw: mne.io.Raw) -> npt.NDArray[np.float32]:
    data = raw.get_data()
    for i in range(len(data)):
        data[i] = 2 * (data[i] - np.min(data[i])) / np.ptp(data[i]) - 1
    return data


def band_pass(
    data: npt.NDArray[np.float32], lowcut: float = 2.5, highcut: float = 14, order: int = 6
) -> npt.NDArray[np.float32]:
    """Butterworth zero-phase"""
    sos = signal.butter(
        order,
        [lowcut, highcut],
        btype="band",
        fs=FS,
        output="sos",
    )
    return signal.sosfiltfilt(sos, data)  # y[0 : 500 * 10])


def inspect_patient(patients: list[Patient]) -> None:
    """Inspect 2 second of a channel"""
    f, ax = plt.subplots(3, 1, sharex=True, layout="constrained")
    for patient in patients:
        raw = patient.raw
        y, x = raw.get_data(["P8"], 0, 5000, return_times=True)
        y = y[0]
        # Normalize
        y_norm = 2 * (y - np.min(y)) / np.ptp(y) - 1
        # Bandpass filter
        y_filtered = band_pass(y_norm)
        ax[0].plot(x, y, label=f"{patient.group}")
        ax[0].set_title("Raw signal [P8]")
        ax[0].set_ylabel("µV")
        ax[1].plot(x, y_norm)
        ax[1].set_title("Normalized signal [P8]")
        ax[2].plot(x, y_filtered)
        ax[2].set_title("Filtered signal [P8]")
        ax[2].set_xlabel("Time (s)")
    f.legend()
    plt.savefig("result/data_processing.svg")

    # Inspect raw signal
    pass


def main() -> int:
    _rng = np.random.RandomState(420)
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    patients = load_data(args.data_dir)

    # df = calculate_psd_metrics(patients)
    # plot_all_psd(df)
    # save_psd_metric_plots(df)

    # inspect_patient([patients[0], patients[-1]])

    y = patients[0].raw.get_data(picks=["P8"])
    y = y[0]  # [0:1000]
    y_n = normalize_energy(patients[0].raw)
    y_n = y_n[0]  # [0:1000]
    y_bp = band_pass(y_n)
    # plot_and_save(y[0][0:1000], "tmp.svg")
    # y = patients[0].raw.get_data(["P8"])[0]
    # Normalize
    # y = 2 * (y - np.min(y)) / np.ptp(y) - 1
    # Filter with a zero-phase butterworth 6th order filter
    # b, a = signal.butter(8, 0.5)

    order = 4
    lpc = librosa.lpc(y_bp, order=order)
    print(lpc)
    (f, y_psd) = signal.welch(y, FS, nperseg=1028 * 10, scaling="density")
    y_psd_db = 10 * np.log10(y_psd)

    (f, y_bp_psd) = signal.welch(y_bp, FS, nperseg=1028 * 10, scaling="density")
    y_bp_psd_db = 10 * np.log10(y_bp_psd)
    lpc = librosa.lpc(y_bp_psd, order=order)

    lpc = np.hstack([[0], -1 * lpc[1:]])
    y_hat = scipy.signal.lfilter(lpc, [1], y_bp_psd)
    # y_hat = scipy.signal.lfilter(lpc, [1], y_bp)

    # (f, y_hat_psd) = signal.welch(y_hat, FS, nperseg=1028, scaling="density")
    # y_hat_psd_db = 10 * np.log10(y_hat_psd)

    # (f, lpc_psd) = signal.welch(
    #     y_hat, FS, nperseg=4096, scaling="density"
    # )  # , FS, nperseg=4096 * 8, scaling="density")
    # lpc_psd_db = 10 * np.log10(lpc_psd)

    # _, ax = plt.subplots(2)
    # ax[0].plot(y_n)
    # ax[0].plot(y_hat)
    # plt.show()

    # exit()

    # sys.exit()
    # b = np.hstack([[0], -1 * a[1:]])
    # x_hat = scipy.signal.lfilter(b, [1], x)
    # a_psd = signal.psd
    _, ax = plt.subplots(3)
    ax[0].plot(f, y_psd_db)
    ax[0].set_xlim(1, 15)
    ax[0].set_ylim(-130, -100)

    ax[1].plot(f, y_bp_psd_db)
    ax[1].set_xlim(1, 15)
    ax[1].set_ylim(-70, 30)

    # ax[2].plot(y_bp[0][:500])
    # ax[2].plot(y_hat[:500])
    ax[2].plot(f, y_hat)
    ax[2].set_xlim(1, 15)
    ax[2].set_ylim(-60, 0)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
