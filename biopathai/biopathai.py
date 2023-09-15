#!/usr/bin/env python3
"""A Python tool for the analysis of biological pathways with machine learning techniques
"""

import argparse as ap
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path

from biopathai import __date__, __version__
from biopathai.modules.evaluate import evaluate
from biopathai.modules.prepare import prepare

# Define the tool name
TOOL_ID = "BioPathAI"

# Control current Python version
# It requires Python 3.6 or higher
if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception(
        "{} requires Python 3.6 or higher. Your current Python version is {}.{}.{}".format(
            TOOL_ID, sys.version_info[0], sys.version_info[1], sys.version_info[2]
        )
    )


def read_params():
    p = ap.ArgumentParser(
        prog=TOOL_ID,
        description="{}: a Python tool for the analysis of biological pathways with machine learning techniques".format(TOOL_ID),
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--classifier-nproc",
        type=int,
        default=1,
        dest="classifier_nproc",
        help="Make the execution of the classification algorithms parallel"
    )
    p.add_argument(
        "--classifier",
        type=str,
        choices=["decisiontree", "gaussianprocess", "nearestneighbors", "neuralnet", "randomforest", "svm"],
        default="randomforest",
        help="Select a classification algorithm"
    )
    p.add_argument(
        "--evaluate-in-memory",
        action="store_true",
        default=False,
        dest="evaluate_in_memory",
        help="Do not dump results on file during the evaluation process"
    )
    p.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of folds for the cross validation"
    )
    p.add_argument(
        "--how-many",
        type=int,
        default=1000,
        dest="how_many",
        help="Number of random pathways"
    )
    p.add_argument(
        "--in-file",
        type=str,
        required=True,
        dest="in_file",
        help="Path to the input tab-separated-values file with the list of paths to the data files and classes (e.g. control/case)"
    )
    p.add_argument(
        "--in-key-pos",
        type=int,
        required=True,
        dest="in_key_pos",
        help="Position of keys in input data files (e.g., column with Gene Symbols)"
    )
    p.add_argument(
        "--in-sep",
        type=str,
        default="\t",
        dest="in_sep",
        help="Field separator of input data files"
    )
    p.add_argument(
        "--in-value-pos",
        type=int,
        required=True,
        dest="in_value_pos",
        help="Position of values in input data files (e.g., column with the expression values)"
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Make it parallel"
    )
    p.add_argument(
        "--pathways",
        type=str,
        required=True,
        help="Pathways definition file (e.g., CPDB_pathways_genes.tab)"
    )
    p.add_argument(
        "--prepare-in-memory",
        action="store_true",
        default=False,
        dest="prepare_in_memory",
        help="Do not dump the pathways matrices on file and keep them in memory"
    )
    p.add_argument(
        "--random-classes",
        action="store_true",
        default=False,
        dest="random_classes",
        help="Shuffle classes"
    )
    p.add_argument(
        "--random-pathways",
        action="store_true",
        default=False,
        dest="random_pathways",
        help="Generate random pathways"
    )
    p.add_argument(
        "--out-folder",
        type=str,
        required=True,
        dest="out_folder",
        help="Path to the output folder"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print results on the stdout"
    )
    p.add_argument(
        "-v",
        "--version",
        action="version",
        version="{} version {} ({})".format(TOOL_ID, __version__, __date__),
        help="Print {} version and exit".format(TOOL_ID),
    )
    return p.parse_args()


def main() -> None:
    # Load command line parameters
    args = read_params()

    if args.verbose:
        print("{} v{} ({})".format(TOOL_ID, __version__, __date__))

    output_filepath = os.path.join(args.out_folder, "accuracy.csv")

    if os.path.isfile(output_filepath):
        raise Exception("The output file already exists!\n{}".format(output_filepath))

    t0 = time.time()

    prepare_output_folder = os.path.join(args.out_folder, "matrices")

    # Generate data
    matrices = prepare(
        args.in_file,
        prepare_output_folder,
        args.pathways,
        in_key_pos=args.in_key_pos,
        in_value_pos=args.in_value_pos,
        in_sep=args.in_sep,
        random_pathways=args.random_pathways,
        random_classes=args.random_classes,
        how_many=args.how_many,
        in_memory=args.prepare_in_memory,
        nproc=args.nproc,
        verbose=args.verbose
    )

    if not matrices:
        raise Exception("An error has occurred while generating pathways matrices")

    evaluations_folder = os.path.join(args.out_folder, "evaluations")

    os.makedirs(evaluations_folder, exist_ok=True)

    accuracies = dict()

    with mp.Pool(processes=args.nproc) as pool:
        evaluate_partial = partial(
            evaluate,
            classifiers=[args.classifier],
            folds=args.folds,
            in_memory=args.evaluate_in_memory,
            nproc=args.classifier_nproc,
            verbose=(args.nproc == 1 and args.verbose)
        )

        jobs = [
            pool.apply_async(
                evaluate_partial,
                args=(
                    matrix_filepath,
                    matrices[matrix_filepath],
                    os.path.join(
                        evaluations_folder,
                        os.path.splitext(os.path.basename(matrix_filepath))[0]
                    ),
                )
            )
            for matrix_filepath in matrices
        ]

        for job in jobs:
            file_id, _, file_accuracies = job.get()
            accuracies[file_id] = file_accuracies

    if accuracies:
        with open(output_filepath, "w+") as outfile:
            outfile.write("Name,Accuracy\n")

            # Sort file IDs based on the accuracy in descending order
            sorted_acc = sorted(
                accuracies.keys(),
                key=lambda file_id: accuracies[file_id][args.classifier],
                reverse=True
            )

            for file_id in sorted_acc:
                outfile.write(
                    "{},{}\n".format(
                        os.path.splitext(os.path.basename(file_id))[0],
                        accuracies[file_id][args.classifier]
                    )
                )

    t1 = time.time()

    if args.verbose:
        print("Total elapsed time {}s".format(int(t1 - t0)))


if __name__ == "__main__":
    main()
