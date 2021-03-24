#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)' )
__version__ = '0.01'
__date__ = 'Mar 23, 2021'

import os, random
import argparse as ap
from pathlib import Path

def read_params():
    p = ap.ArgumentParser( description = ( "The prepare_data.py script generate the pathway matrices" ),
                           formatter_class = ap.ArgumentDefaultsHelpFormatter )
    p.add_argument( '--folder', 
                    type = str,
                    help = "Input file matrix" )
    p.add_argument( '--pathways', 
                    type = str,
                    help = "Pathways definition file (CPDB_pathways_genes.tab)" )
    p.add_argument( '--random',
                    action = 'store_true',
                    default = False,
                    help = "Generate random pathways" )
    p.add_argument( '--how_many', 
                    type = int,
                    default = 100,
                    help = "Number of random pathways for each detected size" )
    p.add_argument( '--verbose',
                    action = 'store_true',
                    default = False,
                    help = "Print results" )
    p.add_argument( '-v', 
                    '--version', 
                    action = 'version',
                    version = 'prepare_data.py version {} ({})'.format( __version__, __date__ ),
                    help = "Print the current prepare_data.py version and exit" )
    return p.parse_args()

TUMORS = [ "tcga-acc","tcga-blca","tcga-brca","tcga-cesc","tcga-chol",
           "tcga-coad","tcga-dlbc","tcga-esca","tcga-gbm","tcga-hnsc",
           "tcga-kich","tcga-kirc","tcga-kirp","tcga-laml","tcga-lgg",
           "tcga-lihc","tcga-luad","tcga-lusc","tcga-meso","tcga-ov",
           "tcga-paad","tcga-pcpg","tcga-prad","tcga-read","tcga-sarc",
           "tcga-skcm","tcga-stad","tcga-tgct","tcga-thca","tcga-thym",
           "tcga-ucec","tcga-ucs","tcga-uvm" ]

# OpenGDC data on polimi server
GEQ_MASK = "/home/cappelli/opengdc/bed/tcga/{}/gene_expression_quantification/"

def load_pathways( pathways_filepath, genrandom ):
    pathway2genes = {}
    gene2pathways = {}
    with open(pathways_filepath) as m:
        for line in m:
            line = line.strip()
            if line:
                if not line.startswith("#"):
                    line_split = line.split("\t")
                    genes = line_split[-1].split(",")
                    pathway = '{}__{}__{}'.format(line_split[0], line_split[1], line_split[2])
                    for g in genes:
                        if g not in gene2pathways:
                            gene2pathways[g] = []
                        gene2pathways[g].append(pathway)
                    if genrandom:
                        pathway2genes[pathway] = genes
    return pathway2genes, gene2pathways

def load_geq_file( geq_filepath ):
    geq_data = {}
    with open(filepath) as geq_file:
        for line in geq_file:
            line = line.strip()
            if line:
                line_split = line.split("\t")
                gene = line_split[6]
                value = line_split[10]
                geq_data[gene] = value
    return geq_data

def generate_random_pathways( genes, pathway_sizes, maxnum=100 ):
    gene2pathways = {}
    for random_pathway in pathway_sizes:
        random_selection = list()
        while len(random_selection) < maxnum:
            random_genes = random.sample(genes, random_pathway)
            if random_genes not in random_selection:
                random_selection.append( random_genes )
        for num, selection in enumerate(random_selection):
            for gene in selection:
                if gene not in gene2pathways:
                    gene2pathways[gene] = list()
                gene2pathways[gene].append( '{}_{}'.format(random_pathway, num) )
    return gene2pathways

if __name__ == '__main__':
    print( 'Prepare Data v{} ({})'.format( __version__, __date__ ) )
    # Load command line parameters
    args = read_params()

    if not os.path.exists(args.folder):
        os.mkdir( args.folder )

    # Load the list of pathways and their genes
    if args.verbos:
        print("Loading pathways")
    pathway2genes, gene2pathways = load_pathways(args.pathways, args.random)        
    if args.random:
        pathway_sizes = set()
        for pathway in pathway2genes:
            pathway_sizes.add(len(pathway2genes[pathway]))
        if args.verbose:
            print("\t{} sizes found".format(len(pathway_sizes)))

    already_generated = False
    for tumor in TUMORS:
        geq = GEQ_MASK.format(tumor)
        if os.path.exists(geq):
            tumor_folder = os.path.join(args.folder, tumor)
            if not os.path.exists(tumor_folder):
                os.mkdir( tumor_folder )
            
            pathways_data = dict()
            aliquot2class = dict()
            gen = Path(geq).glob("*.bed")
            for filepath in gen:
                if args.verbose:
                    print("Loading {}".format(filepath))
                geq_data = load_geq_file( str(filepath) )
                if args.random and not already_generated:
                    gene2pathways = generate_random_pathways( list(geq_data.keys()), pathway_sizes, maxnum=args.how_many )
                    already_generated = True
                aliquot = '-'.join(os.path.splitext(os.path.basename(str(filepath)))[0].split("-")[:-1])
                pathways_data[ aliquot ] = dict()
                for gene in geq_data:
                    if gene in gene2pathways:
                        for pathway in gene2pathways[gene]:
                            if pathway not in pathways_data[ aliquot ]:
                                pathways_data[ aliquot ][ pathway ] = {}
                            pathways_data[ aliquot ][ pathway ][ gene ] = geq_data[gene]
                exclass = "None"
                meta = str(filepath) + ".meta"
                with open(meta) as m:
                    for line in m:
                        line = line.strip()
                        if line:
                            line_split = line.split("\t")
                            if line_split[0] == "manually_curated__tissue_status":
                                exclass = line_split[1]
                                break
                aliquot2class[aliquot] = exclass

            if args.verbose:
                print("Sorting genes")
            pathway2genes = {}
            for aliquot in pathways_data:
                for pathway in pathways_data[aliquot]:
                    for gene in pathways_data[aliquot][pathway]:
                        if pathway not in pathway2genes:
                            pathway2genes[pathway] = []
                        pathway2genes[pathway].append(gene)
            pathway2genes_sorted = {}
            for pathway in pathway2genes:
                pathway2genes_sorted[pathway] = sorted(list(set(pathway2genes[pathway])))

            if args.verbose:
                print("Building matrices\n")
            headers = []
            for aliquot in pathways_data:
                for pathway in pathways_data[aliquot]:
                    filename = pathway.translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "})
                    with open(os.path.join(tumor_folder, '{}.csv'.format(filename)), "a+") as m:
                        if filename not in headers:
                            m.write('aliquot,{},class\n'.format(','.join(pathway2genes_sorted[pathway])))
                            headers.append(filename)
                        gvalues = [ aliquot ]
                        for gene in pathway2genes_sorted[pathway]:
                            if gene in pathways_data[aliquot][pathway]:
                                gvalues.append( pathways_data[aliquot][pathway][gene] )
                            else:
                                gvalues.append( "0.0" )
                        gvalues.append(aliquot2class[aliquot])
                        m.write( '{}\n'.format(','.join(gvalues)) )
