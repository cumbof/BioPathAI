#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)',
                'Giovanni Felici (giovanni.felici@iasi.cnr.it)',
                'Paola Bertolazzi (paola.bertolazzi@iasi.cnr.it)',
                'Gabriella Mavelli (gabriella.mavelli@iasi.cnr.it)' )
__version__ = '0.01'
__date__ = 'Jun 07, 2021'

import os, random, copy

basepath = "/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp"

pathways_filepath = os.path.join(basepath, "Draghici2019_pathways.tsv")
annotations_filepath = os.path.join(basepath, "geo", "annotations", "GPL96.txt")
dataset_filepath = os.path.join(basepath, "geo", "GSE9476_GPL96_AcuteMyeloidLeukemia", "GSE9476_GPL96_AcuteMyeloidLeukemia.txt")
out_matrix_filepath = os.path.join(basepath, "geo", "GSE9476_GPL96_AcuteMyeloidLeukemia", "GSE9476_GPL96_AcuteMyeloidLeukemia_matrix.txt")
out_annotations_filepath = os.path.join(basepath, "geo", "GSE9476_GPL96_AcuteMyeloidLeukemia", "GSE9476_GPL96_AcuteMyeloidLeukemia_annotation.txt")

def load_pathways(pathways_filepath):
    # Load pathways
    pathways = dict()
    with open(pathways_filepath) as pfile:
        for line in pfile:
            line = line.strip()
            if line:
                line_split = line.split("\t")
                pathway = '{}__{}__{}'.format(line_split[0], line_split[1], line_split[2])
                pathway = pathway.translate({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "})
                pathways[pathway] = line_split[-1].split(",")
    return pathways

def load_annotations(annotations_filepath):
    # Load annotations
    annotations = dict()
    with open(annotations_filepath) as afile:
        header = list()
        start = False
        for line in afile:
            line = line.strip()
            if line:
                if line.startswith("ID\t"):
                    header = line.split("\t")
                    start = True
                else:
                    if start:
                        line_split = line.split("\t")
                        try:
                            annotations[line_split[0]] = line_split[header.index("Gene Symbol")].split(" /// ")
                        except:
                            annotations[line_split[0]] = "None"
    return annotations

def load_dataset(dataset_filepath):
    # Load dataset
    samples = list()
    classes = list()
    matrix = dict()
    with open(dataset_filepath) as dfile:
        start = False
        for line in dfile:
            line = line.strip()
            if line:
                if not start:
                    if line.startswith("!Sample_source_name_ch1"):
                        classes = line.replace("\"", "").split("\t")[1:]
                    elif line.startswith("!series_matrix_table_begin"):
                        start = True
                else:
                    if line.startswith("\"ID_REF\""):
                        samples = line.replace("\"", "").split("\t")[1:]
                    elif line.startswith("!series_matrix_table_end"):
                        break
                    else:
                        line = line.replace("\"", "")
                        line_split = line.split("\t")
                        matrix[line_split[0]] = line_split[1:]
    return samples, classes, matrix

def dump_matrix(out_matrix_filepath, samples, classes, matrix):
    # Dump expression matrix
    with open(out_matrix_filepath, "w+") as outm:
        probes = list(matrix.keys())
        outm.write("Sample,{},Class\n".format(','.join(probes)))
        for sample_idx in range(len(samples)):
            sample = samples[sample_idx]
            expclass = classes[sample_idx]
            expdata = list()
            for probe in probes:
                expdata.append(matrix[probe][sample_idx])
            outm.write("{}\t{}\t{}\n".format(sample, ','.join(expdata), expclass))

def invert_annotation(out_annotations_filepath, annotations):
    # Dump inverted annotation 
    inverted_annotation = dict()
    with open(out_annotations_filepath, "w+") as outm:
        for probe in annotations:
            for syn in annotations[probe]:
                if syn not in inverted_annotation:
                    inverted_annotation[syn] = list()
                inverted_annotation[syn].append(probe)
                outm.write("{}\t{}\n".format(syn, probe))
    return inverted_annotation

if __name__ == '__main__':
    # Load pathways
    pathways = load_pathways(pathways_filepath)

    # Load annotations
    annotations = load_annotations(annotations_filepath)

    # Load dataset
    samples, classes, matrix = load_dataset(dataset_filepath)

    # Dump matrix
    dump_matrix(out_matrix_filepath, samples, classes, matrix)

    # Invert annotation
    inverted_annotation = invert_annotation(out_annotations_filepath, annotations)
    
    # Dump pathways expression data and generate random data by shuffling genes and classes
    types = ["real", "random_genes", "random_classes"]
    howmany = 300
    for mtype in types:
        data_basepath = "/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/geo/GSE9476_GPL96_AcuteMyeloidLeukemia/{}/".format(mtype)
        if not os.path.exists(data_basepath):
            os.mkdir(data_basepath)
        for pathway in pathways:
            pathname = os.path.join(data_basepath, pathway)
            interval = range(howmany) if mtype != "real" else range(1)
            for i in interval:
                classes_arr = copy.deepcopy(classes)
                if mtype.startswith("random_classes"):
                    random.shuffle(classes_arr)
                genes_in_pathway = pathways[pathway]
                probes_in_pathway = list()
                for gene in genes_in_pathway:
                    if gene in inverted_annotation:
                        for probe in inverted_annotation[gene]:
                            probes_in_pathway.append(probe)
                with open("{}__size{}__{}.csv".format(pathname, len(probes_in_pathway), i), "w+") as m:
                    genes = list()
                    if mtype.startswith("random_genes"):
                        probes_in_pathway = random.sample(list(annotations.keys()), len(probes_in_pathway))
                    for probe in probes_in_pathway:
                        genes.append("{}__{}".format(annotations[probe][0], probe))
                    m.write("Sample,{},Class\n".format(','.join(genes)))
                    for sample_idx in range(len(samples)):
                        sample = samples[sample_idx]
                        expclass = classes_arr[sample_idx]
                        expdata = list()
                        for probe in probes_in_pathway:
                            expdata.append(matrix[probe][sample_idx])
                        m.write("{},{},{}\n".format(sample, ','.join(expdata), expclass))

    # Acute Myeloid Leukemia
    '''
    var1="leukemia"
    var2="leukemai"
    var3="bone marrow cd34+ selected cells"
    var4="bone marrow"
    var5="pbsc cd34 selected cells"
    var6="peripheral blood mononuclear cells"
    find . -type f -name "*.csv" -exec sed -i -e "s/$var1/Condition/gI; s/$var2/Condition/gI; s/$var2/Normal/gI; s/$var5/Normal/gI; s/$var6/Normal/gI" {} \;
    find . -type f -name "*.csv" -exec sed -i -e "s/$var4/Normal/gI" {} \;
    '''

    # Colorectal Cancer
    '''
    var1="human colorectal adenoma"
    var2="human colonic normal mucosa"
    find . -type f -name "*.csv" -exec sed -i -e "s/$var1/Condition/gI; s/$var2/Normal/gI" {} \;
    '''

    # Parkinson's Disease
    '''
    var1="disease state: parkinsons disease"
    var2="disease state: control"
    find . -type f -name "*.csv" -exec sed -i -e "s/$var1/Condition/gI; s/$var2/Normal/gI" {} \;
    '''
