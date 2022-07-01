import os, random, copy

pathways = {}
with open("/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/Draghici2019_pathways.tsv") as m:
    for line in m:
        line = line.strip()
        if line:
            line_split = line.split("\t")
            pathway = '{}__{}__{}'.format(line_split[0], line_split[1], line_split[2])
            pathway = pathway.translate({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+ "})
            pathways[pathway] = line_split[-1].split(",")

annotations = {}
with open("/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/geo/annotations/GPL96.txt") as m:
    header = []
    start = False
    for line in m:
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

samples = []
classes = []
matrix = {}
with open("/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/geo/GSE9476_GPL96_AcuteMyeloidLeukemia/GSE9476_GPL96_AcuteMyeloidLeukemia.txt") as m:
    mstart = False
    for line in m:
        line = line.strip()
        if line:
            if not mstart:
                if line.startswith("!Sample_source_name_ch1"):
                    classes = line.replace("\"", "").split("\t")[1:]
                elif line.startswith("!series_matrix_table_begin"):
                    mstart = True
            else:
                if line.startswith("\"ID_REF\""):
                    samples = line.replace("\"", "").split("\t")[1:]
                elif line.startswith("!series_matrix_table_end"):
                    break
                else:
                    line = line.replace("\"", "")
                    line_split = line.split("\t")
                    matrix[line_split[0]] = line_split[1:]

with open("/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/geo/GSE9476_GPL96_AcuteMyeloidLeukemia/GSE9476_GPL96_AcuteMyeloidLeukemia_matrix.txt", "w+") as m:
    probes = list(matrix.keys())
    m.write("Sample,{},Class\n".format(','.join(probes)))
    for sample_idx in range(len(samples)):
        sample = samples[sample_idx]
        expclass = classes[sample_idx]
        expdata = []
        for probe in probes:
            expdata.append(matrix[probe][sample_idx])
        m.write("{}\t{}\t{}\n".format(sample, ','.join(expdata), expclass))

inverted_annotation = {}
with open("/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/geo/GSE9476_GPL96_AcuteMyeloidLeukemia/GSE9476_GPL96_AcuteMyeloidLeukemia_annotation.txt", "w+") as m:
    for probe in annotations:
        for syn in annotations[probe]:
            if syn not in inverted_annotation:
                inverted_annotation[syn] = []
            inverted_annotation[syn].append(probe)
            m.write("{}\t{}\n".format(syn, probe))

types = ["real", "random_genes", "random_classes"]
howmany = 300
for mtype in types:
    basepath = "/shares/CIBIO-Storage/CM/scratch/users/fabio.cumbo/tmp/geo/GSE9476_GPL96_AcuteMyeloidLeukemia/{}/".format(mtype)
    if not os.path.exists(basepath):
        os.mkdir(basepath)
    for pathway in pathways:
        pathname = os.path.join(basepath, pathway)
        interval = range(howmany) if mtype != "real" else range(1)
        for i in interval:
            classes_arr = copy.deepcopy(classes)
            if mtype.startswith("random_classes"):
                random.shuffle(classes_arr)
            genes_in_pathway = pathways[pathway]
            probes_in_pathway = []
            for gene in genes_in_pathway:
                if gene in inverted_annotation:
                    for probe in inverted_annotation[gene]:
                        probes_in_pathway.append(probe)
            with open("{}__size{}__{}.csv".format(pathname, len(probes_in_pathway), i), "w+") as m:
                genes = []
                if mtype.startswith("random_genes"):
                    probes_in_pathway = random.sample(list(annotations.keys()), len(probes_in_pathway))
                for probe in probes_in_pathway:
                    genes.append("{}__{}".format(annotations[probe][0], probe))
                m.write("Sample,{},Class\n".format(','.join(genes)))
                for sample_idx in range(len(samples)):
                    sample = samples[sample_idx]
                    expclass = classes_arr[sample_idx]
                    expdata = []
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
