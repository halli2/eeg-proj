import sys
from argparse import ArgumentParser
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    read_raw_bids,
)
from scipy import signal

# from EEGModels import EEGNet
# from run import run

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
        # print(self.raw.describe())
        # sys.exit()


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

    # df = df.loc[[0, 20, 110]]  # TODO: Remove
    patients = [Patient(row) for _, row in df.iterrows()]
    return patients


def calculate_psd_metrics(patients: Patients):
    pass


def get_psds(patients: Patients, fmin=0, fmax=120, method="welch") -> pd.DataFrame | list[mne.time_frequency.Spectrum]:
    psds = [p.raw.compute_psd(fmin=fmin, fmax=fmax, method=method) for p in patients]
    df = pd.DataFrame()
    for ch in psds[0].ch_names:
        all_channels = np.array([psd.get_data([ch]) for psd in psds])
        all_channels = 10 * np.log10(all_channels)
        avg = np.average(all_channels, axis=0)[0]
        std = np.std(all_channels, axis=0)[0]
        std_min = avg - std
        std_max = avg + std
        df[ch] = {"avg": avg, "min": std_min, "max": std_max}
    print(df.head())
    return df, psds


def plot_psd(data: npt.NDArray, ax: Optional[plt.Axes] = None) -> None:
    f, power = signal.welch(
        data,
        FS,
        "flattop",  # hann?
        scaling="density",
    )
    power_db = 10 * np.log10(power)

    if ax is None:
        _, ax = plt.subplots()
    ax.set_xlim(1, 60)
    ax.set_ylim(-190, -100)
    ax.plot(f, power_db)
    ax.set_xlabel("Freq [Hz]")
    ax.set_ylabel("Power / freq [dB]")


def main() -> int:
    _rng = np.random.RandomState(420)
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    patients = load_data(args.data_dir)
    df, psds = get_psds(patients)
    psds: list[mne.time_frequency.Spectrum] = psds

    _, freqs = psds[0].get_data(["P7"], return_freqs=True)
    f, ax = plt.subplots()
    p7 = df["P7"]
    avg_p7_db = p7["avg"]
    min_p7_db = p7["min"]
    max_p7_db = p7["max"]
    # avg_p7_db = 10 * np.log10(p7["avg"])
    # min_p7_db = 10 * np.log10(p7["min"])
    # max_p7_db = 10 * np.log10(p7["max"])
    # min_p7_db = max_p7_db + avg_p7_db
    ax.plot(freqs, avg_p7_db)
    ax.fill_between(freqs, min_p7_db, max_p7_db, color="C0", alpha=0.5)
    ax.set_xscale("symlog")
    ax.set_xlim(1, 55)
    ax.set_xticks([2, 4, 7, 13, 25, 50])
    plt.show()

    # print(avgs["P8"])
    # fig, ax = plt.subplots(1, 2)  # , sharex=True, sharey=True)
    # # plt.psd()
    # data, _times = patients[0].raw.get_data(["P7"], return_times=True)

    # plot_psd(data[0], ax[0])
    # plt.show()

    # f, Pxx_spec = signal.welch(
    #     data[0],
    #     FS,
    #     "flattop",
    #     512 * 4,
    #     scaling="spectrum",
    # )
    # Pxx_spec = 10 * np.log10(Pxx_spec)
    # plt.figure()
    # plt.semilogy(f, np.sqrt(Pxx_spec))
    # plt.xlim(2, 50)
    # plt.xlabel("frequency [Hz]")
    # plt.ylabel("Linear spectrum [V RMS]")
    # plt.title("Power spectrum")
    # plt.show()

    # ax[0].psd(data[0], Fs=FS)
    # plt.xlim(2, 50)
    # ax[0].plot(f, Pxx_spec)
    # ax[0].set_xticks(np.arange(0, 60, 10))
    # psds[0].plot(axes=ax[1])
    # plt.show()
    # psd = patients[0].raw.compute_psd(method="welch")
    # psd = patients[0].raw.compute_psd(method="multitaper")
    # p8 = psd.get_data(["P8", "P7"])
    # print(type(p8[0][0]))
    # print(p8)
    # psd.plot()
    # plt.show()
    # run(args.data_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
