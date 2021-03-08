# Pathways evaluation

The `src` folder contains three Python 3 scripts:
- `prepare_data.py` generates matrices, one for each pathway, starting from Gene Expression Quantification data of all the 33 different types of cancers in TCGA retrieved with OpenGDC;
- `ripper.py` is a python implementation of the `RIPPER` algortihm with the same iterative logics of `CAMUR`;
- `evaluate.py` is a wrapper for `N` classfication algorithms of the scikit learn package. It starts with a pathway matrix in input and produce (i) a binary matrix (`1` if a sample in row `i` has been misclassified by the algorithm under the column `j`, otherwise `0`) and (ii) the confusion matrices for all the involved classifiers.

Folder `assets` contains the `CPDB_pathways_genes.tab` with the list of pathways and involved genes.

BZ2 compressed tarballs `pathways-tcga.tar.bz2` and `random-tcga.tar.bz2` contain a directory for each of the TCGA tumors with a matrix for each of the pathways generated with the `prepare_data.py` script.

Due to their size, both the compressed tarballs are available outside GitHub:
- [https://www.dropbox.com/s/e045x0a65aevrst/pathways-tcga.tar.bz2?dl=1](https://www.dropbox.com/s/e045x0a65aevrst/pathways-tcga.tar.bz2?dl=1)
- [https://www.dropbox.com/s/hfuhdzojzqiondn/random-tcga.tar.bz2?dl=1](https://www.dropbox.com/s/hfuhdzojzqiondn/random-tcga.tar.bz2?dl=1)

How to evaluate a pathway:
```
python evaluate.py 
       --input ../matrices/tcga-brca/Signal_Transduction__R_HSA_162582___Reactome_.csv 
       --output_prefix ../matrices/tcga-brca/Signal_Transduction__R_HSA_162582___Reactome_ 
       --folds 10 
       --verbose
```
