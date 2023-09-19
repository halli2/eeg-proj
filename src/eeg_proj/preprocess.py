from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
import matplotlib.pyplot as plt


def preprocess_tsv(data_dir: Path):
    """Goes through all *channels.tsv files and adds signal type and saves s *channels_processed.tsv"""
    for file in data_dir.iterdir():
        if not file.is_dir():
            continue
        ch_file = next(iter(file.rglob("*channels.tsv")))
        ch_file_edited = f"{Path.joinpath(ch_file.parent, ch_file.stem)}_processed{ch_file.suffix}"
        df = pd.read_csv(ch_file, sep="\t")
        df["type"] = "eeg"
        df.to_csv(ch_file_edited, sep="\t")


def load_data(data_dir: Path) -> tuple[list[mne.io.Raw], list[str]]:
    raw_paths: list[BIDSPath] = find_matching_paths(
        data_dir,
        extensions=[".set"],
    )

    df = pd.read_csv(f"{data_dir}/participants.tsv", sep="\t")
    raws = [
        read_raw_bids(bids_path, verbose=False, extra_params={"preload": True}).crop(0, 60) for bids_path in raw_paths
    ]

    for raw in raws:
        # Set all channels to eeg channels
        for ch in raw.ch_names:
            raw.set_channel_types({ch: "eeg"})
    # raws[0].plot()
    raw.filter(1.0, None)
    raw.filter(None, 50.0)
    raws[0].plot_psd()
    plt.show()

    x = raws
    y = df["GROUP"].to_numpy()  # Control group or Parkinsons disease

    return x, y
