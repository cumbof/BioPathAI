import os
from pathlib import Path

tumors = [ "tcga-acc","tcga-blca","tcga-brca","tcga-cesc","tcga-chol",
           "tcga-coad","tcga-dlbc","tcga-esca","tcga-gbm","tcga-hnsc",
           "tcga-kich","tcga-kirc","tcga-kirp","tcga-laml","tcga-lgg",
           "tcga-lihc","tcga-luad","tcga-lusc","tcga-meso","tcga-ov",
           "tcga-paad","tcga-pcpg","tcga-prad","tcga-read","tcga-sarc",
           "tcga-skcm","tcga-stad","tcga-tgct","tcga-thca","tcga-thym",
           "tcga-ucec","tcga-ucs","tcga-uvm" ]

pathways_folder = "/home/cappelli/pathways/"
if not os.path.exists(pathways_folder):
    os.mkdir( pathways_folder )

# genes per pathway
gene2pathways = {}
with open("/home/cappelli/pathways/CPDB_pathways_genes.tab") as m:
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

matrices_folder = os.path.join(pathways_folder, "matrices")
if not os.path.exists(matrices_folder):
    os.mkdir( matrices_folder )

for t in tumors:
    folder = "/home/cappelli/opengdc/bed/tcga/{}/gene_expression_quantification/".format(t)
    tumor_folder = os.path.join(matrices_folder, t)
    if not os.path.exists(tumor_folder):
        os.mkdir( tumor_folder )
    if os.path.exists(folder):
        pathways_data = { }
        aliquot2class = {}
        gen = Path(folder).glob("*.bed")
        for filepath in gen:
            print("Loading {}".format(filepath))
            aliquot = '-'.join(os.path.splitext(os.path.basename(str(filepath)))[0].split("-")[:-1])
            pathways_data[ aliquot ] = { }
            with open(str(filepath)) as m:
                for line in m:
                    line = line.strip()
                    if line:
                        line_split = line.split("\t")
                        gene = line_split[6]
                        if gene in gene2pathways:
                            for pathway in gene2pathways[gene]:
                                if pathway not in pathways_data[ aliquot ]:
                                    pathways_data[ aliquot ][ pathway ] = {}
                                pathways_data[ aliquot ][ pathway ][ gene ] = line_split[10]
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
        headers = []
        print("Building matrices\n")
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
