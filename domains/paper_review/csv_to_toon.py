import argparse

import pandas as pd
from pandas.testing import assert_frame_equal

from domains._toon_io import read_toon, write_toon


def convert_csv_to_toon(input_path, output_path):
    df = pd.read_csv(input_path, dtype=str)
    write_toon(df, output_path)
    decoded = read_toon(output_path)
    assert_frame_equal(df, decoded, check_dtype=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a paper_review CSV dataset to TOON.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output TOON path")
    args = parser.parse_args()

    convert_csv_to_toon(args.input, args.output)
