#!/usr/bin/env python3
"""Compute the p-values on the accuracies of real and random pathways.
"""

__authors__ = ("Fabio Cumbo (fabio.cumbo@gmail.com)",
               "Valerio Ponzi (ponzi@diag.uniroma1.it)")

__version__ = "0.1.0"
__date__ = "Sep 14, 2023"

import argparse as ap
import errno
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

TOOL_ID = "biopathai_pvalue"


def read_params():
    p = ap.ArgumentParser(
        prog=TOOL_ID,
        description="Compute the p-values on the accuracies of real and random pathways",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dump",
        type=os.path.abspath,
        help="Path to the output file (optional)",
    )
    p.add_argument(
        "--real-accuracies",
        type=os.path.abspath,
        required=True,
        dest="real_accuracies",
        help="Path to the accuracy table computed on the real pathways",
    )
    p.add_argument(
        "--random-accuracies",
        type=os.path.abspath,
        required=True,
        dest="random_accuracies",
        help="Path to the accuracy table computed on the random shuffled pathways",
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version='"{}" version {} ({})'.format(TOOL_ID, __version__, __date__),
        help='Print the "{}" version and exit'.format(TOOL_ID),
    )
    return p.parse_args()


# https://stackoverflow.com/a/33532498
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def main() -> None:
    args = read_params()

    if not os.path.isfile(args.real_accuracies):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.real_accuracies)

    if not os.path.isfile(args.random_accuracies):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.random_accuracies)

    if args.dump and os.path.isfile(args.dump):
        raise Exception("The output file already exists!\n{}".format(args.dump))

    # Load accuracies tables
    real_data = pd.read_csv(args.real_accuracies)
    random_data = pd.read_csv(args.random_accuracies)

    data = dict()

    for _, row_real in real_data.iterrows():
        pathway = row_real["Name"]

        # Expected value
        real_accuracy = row_real["Accuracy"]

        # Count how many times the accuracy computed on a random pathway is higher than the expected value
        count = 0

        # Also count the total number of random pathways
        total = 0

        for _, row_random in random_data.iterrows():
            if row_random["Name"].startswith("{}__s".format(pathway)):
                if row_random["Accuracy"] >= real_accuracy:
                    count += 1

                total += 1

        data[pathway] = count / total

    if data:
        sorted_pathways = sorted(data.keys(), key=lambda pathway: data[pathway])

        sorted_pvalues = [data[pathway] for pathway in sorted_pathways]

        adjusted_pvalues = p_adjust_bh(sorted_pvalues)

        table = [["Pathway", "p-value", "FDR"]]

        for idx, pathway in enumerate(sorted_pathways):
            table.append([pathway, sorted_pvalues[idx], adjusted_pvalues[idx]])

        if args.dump:
            with open(args.dump, "w+") as outfile:
                for table_row in table:
                    outfile.write("{}\n".format(",".join(table_row)))

        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
