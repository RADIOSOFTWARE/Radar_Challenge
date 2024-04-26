import numpy as np
import matplotlib
import argparse
import csv


"""
to-do
[-] unpacking
[x] packaging
[ ] +- brutforse
[ ] algorithm
"""


def algorithm(data: np.ndarray) -> np.ndarray:
    """
    method of seatching for global hypotheses
    data: np.ndarray - input compatibility matrix
    """

    return data


def packaging(patch_file: str, data: np.ndarray) -> None:
    """
    method of packing of global hypotheses into a csv table
    """
    quantity_column = len(data[0])
    with open(patch_file, "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        writer.writerow(list(range(quantity_column)) + ["W"])
        writer.writerows(data)


def main():
    # ---------------parser-config------------------
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--i", help="Input compatibility matrix file in csv format")
    arg_parser.add_argument(
        "--o", help="Output global hypotheses file in csv", default="output.csv"
    )

    # ---------------parsing-parameters-------------
    args = arg_parser.parse_args()
    input_file = args.i
    output_file = args.o

    # ---------------parsing-file-------------------
    data: np.ndarray
    with open(input_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        data = np.array(list(csv_reader), dtype=int)

    # ------------------start-----------------------
    output_data: np.ndarray = algorithm(data)
    packaging(output_file, output_data)


if __name__ == "__main__":
    main()
