from argparse import ArgumentParser
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from numpy.random import RandomState

import preprocess

# from EEGModels import EEGNet
from run import run


SAMPLES = 128


def main() -> None:
    run()
    # rng = RandomState(420)
    # parser = ArgumentParser()
    # parser.add_argument(
    #     "-p",
    #     "--preprocess",
    #     help="Preprocess .tsv files.",
    #     action="store_true",
    # )
    # args = parser.parse_args()

    # data_dir = Path(__file__).parents[2].joinpath("bids/ds004584")
    # if args.preprocess:
    #     preprocess.preprocess_tsv(data_dir)

    # raws, y = preprocess.load_data(data_dir)

    # enc = OneHotEncoder()
    # enc.fit([['Control', 0], ['PD', 1]])
    # y = enc.transform(y)
    # print(y)
    # exit()
    # drop_samples = []
    # x = [v.get_data(stop=SAMPLES) for v in raws]
    # for i, v in enumerate(x):
    #     if len(v) != 63:
    #         drop_samples.append(i)
    # for i in sorted(drop_samples, reverse=True):
    #     del x[i]
    #     del y[i]

    # train_x, test_x, train_y, test_y = train_test_split(
    #     x,
    #     y,
    #     test_size=0.2,
    #     stratify=y,
    #     random_state=rng,
    # )
    # assert len(train_x) == len(train_y)
    # assert len(test_x) == len(test_y)
    # # train_x, validate_x, train_y, validate_y = train_test_split(
    # #     train_x,
    # #     train_y,
    # #     test_size=0.1,
    # #     stratify=y,
    # #     random_state=rng,
    # # )

    # model = EEGNet(2, 62, SAMPLES)
    # model.compile(loss="categorical_crossentropy", optimizer="adam")
    # fitted = model.fit(
    #     x=train_x,
    #     y=train_y,
    #     validation_split=0.1,
    #     # validation_data=(validate_x, validate_y),
    #     # loss="mse",
    # )


if __name__ == "__main__":
    main()
