"""
Script to split train and validation set with given ratio or size.
"""
import pyrootutils

# See: https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True
)

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def create_parser():
    parser = argparse.ArgumentParser(
        description="Split the dataset into training and validation sets and save them as separate CSV files."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV file containing the main dataset."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="Path to output directory where the split datasets will be saved."
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of the training set (between 0 and 1). Default is 0.9."
    )

    parser.add_argument(
        "--num_val",
        type=int,
        default=None,
        help="Number of samples to be used in the validation set. Default is None."
    )

    return parser


def split_and_save_dataset(input_path, output_path, train_ratio=0.9, num_val=None):
    """
    Splits the dataset into training and validation sets and saves them as new CSV files.

    Params:
        input_path: Path to the original CSV file.
        output_path: Directory where the split datasets will be saved.
        train_ratio: Ratio of the training set (between 0 and 1). Default is 0.9.
        num_val: Number of samples to be used in the validation set. Default is None.
    """
    data = pd.read_csv(input_path)

    if num_val is not None and train_ratio is not None:
        print("Both train_ratio and num_val are given. Using num_val to determine validation set size.")
        val_data = data.sample(n=num_val, random_state=42)
        train_data = data.drop(val_data.index)
    else:
        train_data, val_data = train_test_split(data, train_size=train_ratio, random_state=42)

    train_file = os.path.join(output_path, 'metadata_train.csv')
    val_file = os.path.join(output_path, 'metadata_val.csv')

    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)

    print(f"Datasets saved as '{train_file}' (size: {len(train_data)}) and '{val_file}' (size: {len(val_data)})")


def main(args):
    split_and_save_dataset(args.input_path, args.output_path, args.train_ratio, args.num_val)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
