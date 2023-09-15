"""Generate the pathway matrices.
"""

import errno
import multiprocessing as mp
import os
import random
from functools import partial
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple, Union

random.seed(0)


def load_pathways(pathways_filepath: str, genrandom: bool) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Load a pathways definition file

    :param pathways_filepath:   Path to the pathways definition file
    :param genrandom:           True in case of processing random pathways
    :return:                    Dictionaries with mapping between pathways and genes
    """

    pathway2genes = dict()
    gene2pathways = dict()

    with open(pathways_filepath) as infile:
        for line in infile:
            line = line.strip()

            if line:
                if not line.startswith("#"):
                    line_split = line.split("\t")

                    # Get the list of genes
                    genes = line_split[-1].split(",")

                    # Get the pathway name
                    pathway = '{}__{}__{}'.format(line_split[0], line_split[1], line_split[2])

                    for gene in genes:
                        if gene not in gene2pathways:
                            gene2pathways[gene] = list()

                        gene2pathways[gene].append(pathway)

                    if genrandom:
                        pathway2genes[pathway] = genes

    return pathway2genes, gene2pathways


def load_input_data(filepath: str, key_pos: int = 0, value_pos: int = 1, sep: str = "\t") -> Dict[str, str]:
    """Load the input data

    :param filepath:    Path to the input file
    :param key_pos:     Position of keys
    :param value_pos:   Position of values
    :param sep:         Field separator
    :return:            Dictionary with data
    """

    data = dict()

    with open(filepath) as infile:
        for line in infile:
            line = line.strip()

            if line:
                line_split = line.split(sep)
                data[line_split[key_pos]] = line_split[value_pos]

    return data


def generate_random_pathways(genes: List[str], pathway_sizes: Set[str], maxnum: int = 100) -> Dict[str, List[str]]:
    """Generate random pathways

    :param genes:           List of genes
    :param pathway_size:    Size of pathway in terms of number of genes
    :param maximum:         Maximum number of random pathways
    :return:                Dictionary with mapping between genes and random pathways
    """

    gene2pathways = dict()

    for random_pathway in pathway_sizes:
        random_selection = list()

        while len(random_selection) < maxnum:
            random_genes = random.sample(genes, random_pathway)

            if random_genes not in random_selection:
                random_selection.append(random_genes)

        for num, selection in enumerate(random_selection):
            for gene in selection:
                if gene not in gene2pathways:
                    gene2pathways[gene] = list()

                gene2pathways[gene].append("{}_{}".format(random_pathway, num))

    return gene2pathways


def build_pathway_matrix(
    pathway: str,
    pathways_data: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    shuffled_classes: Optional[Dict[int, Dict[str, str]]] = None,
    pathway2genes_sorted: Optional[Dict[str, List[str]]] = None,
    out_folder: Optional[str] = None,
    in_memory: bool = False
) -> Union[Dict[str, str], Dict[str, StringIO]]:
    """Dump the pathway matrices to files

    :param pathway:                 Name of the pathway
    :param pathways_data:           Pathway data with genes
    :param shuffles_classes:        Dictionary with shuffled classes
    :param pathway2genes_sorted:    Pathway with sorted genes
    :param out_folder:              Path to the output folder
    :param in_memory:               Keep matrices in memory
    :return:                        Set with paths to the output matrices
    """

    matrices = dict()

    headers = list()

    for fileid in pathways_data:
        for shuffled_run in shuffled_classes:
            filename = pathway.translate({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "})

            if len(shuffled_classes) > 1:
                filename = "{}__s{}".format(filename, shuffled_run)

            out_filepath = os.path.join(out_folder, "{}.csv".format(filename))

            outfile = None

            if in_memory:
                outfile = StringIO()

            else:
                matrices[out_filepath] = out_filepath
                outfile = open(out_filepath, "a+")

            if filename not in headers:
                outfile.write("fileid,{},class\n".format(",".join(pathway2genes_sorted[pathway])))
                headers.append(filename)

            line = [fileid]

            for gene in pathway2genes_sorted[pathway]:
                if gene in pathways_data[fileid][pathway]:
                    line.append(pathways_data[fileid][pathway][gene])

                else:
                    line.append("0.0")

            line.append(shuffled_classes[shuffled_run][fileid])

            outfile.write("{}\n".format(",".join(line)))

            if in_memory:
                outfile.seek(0)
                matrices[out_filepath] = outfile

            else:
                outfile.close()

    return matrices


def prepare(
    in_file: str,
    out_folder: str,
    pathways: str,
    in_key_pos: int = 0,
    in_value_pos: int = 1,
    in_sep: str = "\t",
    random_pathways: bool = False,
    random_classes: bool = False,
    how_many: int = 100,
    in_memory: bool = False,
    nproc: int = 1,
    verbose: bool = False
) -> Union[Dict[str, str], Dict[str, StringIO]]:
    """Generate the pathway matrices

    :param in_file:         Path to the input file with the list of paths to the input data files
    :param out_folder:      Path to the output folder
    :param pathways:        Path to the input file with the list of pathways and their genes
    :param in_key_pos:      Position of keys in input data
    :param in_value_pos:    Position of values in input data
    :param in_sep:          Field separator in input data files
    :param random_pathways: Generate random pathway matrices
    :param random classes:  Shuffle classes
    :param how_many:        Generate up to this number of random matrices
    :param in_memory:       Keep matrices in memory and do not dump them on file
    :param nproc:           Make it parallel
    :param verbose:         Print messages on the stdout
    :return:                Dictionary with paths to the output matrices
                            Return a dictionary with StringIO objects in case of in_memory
    """

    matrices = dict()

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    if verbose:
        print("Loading pathways")

    pathway2genes, gene2pathways = load_pathways(pathways, random_pathways)        

    if random_pathways:
        pathway_sizes = set()

        for pathway in pathway2genes:
            pathway_sizes.add(len(pathway2genes[pathway]))

        if verbose:
            print("\t{} sizes found".format(len(pathway_sizes)))

    pathways_data = dict()
    fileid2class = dict()

    already_generated = False

    with open(in_file) as infile:
        for line in infile:
            line = line.strip()

            if line:
                line_split = line.split("\t")

                if len(line_split) != 2:
                    # The input file must contain two columns:
                    # (i) the first one with the path to the data file
                    # (ii) the second one with the class label (e.g. normal/tumoral)
                    raise Exception("Malformed input file")

                if not os.path.isfile(line_split[0]):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), line_split[0])

                if verbose:
                    print("Loading {}".format(line_split[0]))

                data = load_input_data(line_split[0], key_pos=in_key_pos, value_pos=in_value_pos, sep=in_sep)

                if random_pathways and not already_generated:
                    gene2pathways = generate_random_pathways(list(data.keys()), pathway_sizes, maxnum=how_many)
                    already_generated = True

                file_id = os.path.splitext(os.path.basename(line_split[0]))[0]
                pathways_data[file_id] = dict()

                for gene in data:
                    if gene in gene2pathways:
                        for pathway in gene2pathways[gene]:
                            if pathway not in pathways_data[file_id]:
                                pathways_data[file_id][pathway] = dict()

                            pathways_data[file_id][pathway][gene] = data[gene]

                fileid2class[file_id] = line_split[1]

    shuffled_classes = dict()

    if random_classes:
        if verbose:
            print("Shuffling classes")

        for i in range(how_many):
            fileid_arr = list(fileid2class.keys())
            class_arr = list(fileid2class.values())

            random.shuffle(class_arr)
            shuffled_classes[i] = dict()

            for j in range(len(fileid_arr)):
                shuffled_classes[i][fileid_arr[j]] = class_arr[j]

    else:
        shuffled_classes[0] = fileid2class

    if verbose:
        print("Sorting genes")

    pathway2genes = dict()

    for fileid in pathways_data:
        for pathway in pathways_data[fileid]:
            for gene in pathways_data[fileid][pathway]:
                if pathway not in pathway2genes:
                    pathway2genes[pathway] = list()

                pathway2genes[pathway].append(gene)

    pathway2genes_sorted = dict()

    for pathway in pathway2genes:
        pathway2genes_sorted[pathway] = sorted(list(set(pathway2genes[pathway])))

    if verbose:
        print("Building matrices")

    with mp.Pool(processes=nproc) as pool:
        build_pathway_matrix_partial = partial(
            build_pathway_matrix,
            pathways_data=pathways_data,
            shuffled_classes=shuffled_classes,
            pathway2genes_sorted=pathway2genes_sorted,
            out_folder=out_folder,
            in_memory=in_memory
        )

        jobs = [
            pool.apply_async(
                build_pathway_matrix_partial,
                args=(pathway,)
            )
            for pathway in pathway2genes
        ]

        for job in jobs:
            matrices = {**matrices, **job.get()}

    return matrices
