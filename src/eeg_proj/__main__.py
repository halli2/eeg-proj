import sys
from argparse import ArgumentParser
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    read_raw_bids,
)

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

    # df = df.loc[[0, 20, 110, 111, 112]]  # TODO: Remove
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
    ax.fill_between(freqs, avg + std, avg - std, color="C0", alpha=0.3)
    ax.fill_between(freqs, pd_avg + pd_std, pd_avg - pd_std, color="C1", alpha=0.3)
    ax.set_xlim(1, 55)
    ax.set_xlabel("Freq [Hz]")
    ax.set_ylabel("μV^2 / Hz [dB]")
    ax.set_title(f"{channel}")
    ax.legend(loc="upper right")


def main() -> int:
    _rng = np.random.RandomState(420)
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    patients = load_data(args.data_dir)

    df = calculate_psd_metrics(patients)

    f, ax = plt.subplots(3, 3)
    index = 0
    for i in range(3):
        for j in range(3):
            if index == len(ELECTRODES):
                break
            plot_psd_metrics(
                df,
                ELECTRODES[index],
                ax[i, j],
            )
            index += 1
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
