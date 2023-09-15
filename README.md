# BioPathAI

A Python tool for the analysis of biological pathways with machine learning techniques

## Requirements

We developed `BioPathAI` as a Python 3.6 package with the following requirements:

- `pandas` (version >=1.3.5)
- `scikit-learn` (version >=0.22.1)
- `tabulate` (version >=0.9.0)

## Install

The tool is available through `pip` on The Python Package Index at [https://pypi.org/project/BioPathAI/](https://pypi.org/project/BioPathAI/).

It can be installed by running the following command on your terminal:

```text
pip install biopathai
```

## Input

`BioPathAI` takes two different types of data in input.

### Pathways definition file

The first one is a tab-separated-values file with the list of patways with their ID, source, and genes, as shown in the following example:

```text
#pathway                                                external_id    source        hgnc_symbol_ids
Rosiglitazone Pharmacokinetic Pathway                   PA165816969    PharmGKB      SLCO1B1,CYP2C8,CYP2C9
spermine biosynthesis                                   ARGSPECAT-PWY  HumanCyc      SMS,AMD1
Plasma lipoprotein assembly, remodeling, and clearance  WP4129         Wikipathways  APOA4,APOA5,APOC2,CIDEC,FGF21
Signaling by Overexpressed Wild-Type EGFR in Cancer     R-HSA-5638302  Reactome      EGFR,EGF
...
```

> **Note**
> The previous example is part of the Consensus Pathway Database available at [http://cpdb.molgen.mpg.de/](http://cpdb.molgen.mpg.de/).
> 
> Please note that, since there could be multiple pathways with the same name but from different sources, all the four columns are required by `BioPathAI` in order to build a unique identifier for each pathway.

### Links to the expression data

`BioPathAI` also takes in input another tab-separated-values file with the list of links to the actual data with the gene expression values and their class (e.g., case or control), like in the example below:

```text
#path                                        class
~/acf1ec71-46a5-4e7b-84b1-55203bca29d8.bed   tumoral
~/429b50eb-316f-459c-bc3a-0aca6e6dba46.bed   tumoral
~/eb9e415f-0545-4e97-ad35-b56b0f3db79b.bed   normal
~/827431d3-e25a-45b6-9b0f-5732501e2a5b.bed   tumoral
~/09f570eb-a751-41db-9511-300be228f24e.bed   normal
...
```

> **Note**
> The files listed in the first colum of the previous example must contains at least two columns, one with the gene names in the same format as reported in the pathways definition file (i.e., Gene Symbols in this particular example), the other one with the actual gene expression values.
>
> In this example we reported a few free-BED files retrieved from the OpenGDC public FTP repository at [ftp://geco.deib.polimi.it/opengdc/](ftp://geco.deib.polimi.it/opengdc/).

## Usage

Here is a list of available options:

| Option                 | Default        | Mandatory | Description  |
|:-----------------------|:---------------|:---------:|:-------------|
| `--classifier-nproc`   | `1`            |           | Make the execution of the classification algorithms parallel |
| `--classifier`         | `randomforest` |           | Select a classification algorithm. Possible values: `decisiontree`, `gaussianprocess`, `nearestneighbors`, `neuralnet`, `randomforest`, and `svm` |
| `--evaluate-in-memory` | `False`        |           | Do not dump results on file during the evaluation process |
| `--folds`              | `10`           |           | Number of folds for the cross-validation |
| `--how-many`           | `1000`         |           | Number of random pathways |
| `--in-file`            |                | ⚑         | Path to the input tab-separated-values file with the list of paths to the data files and classes (e.g. control/case) |
| `--in-key-pos`         |                | ⚑         | Position of keys in the input data files (e.g., column with the Gene Symbols) |
| `--in-sep`             | `\t`           |           | Field separator of input data files |
| `--in-value-pos`       |                | ⚑         | Position of values in the input data files (e.g., column with the expression values) |
| `--nproc`              | `1`            |           | Make it parallel |
| `--out-folder`         |                | ⚑         | Path to the output folder |
| `--pathways`           |                | ⚑         | Path to the pathways definition file in tsv format |
| `--prepare-in-memory`  | `False`        |           | Do not dump the pathways matrices on file and keep them in memory |
| `--random-classes`     | `False`        |           | Shuffle classes |
| `--random-pathways`    | `False`        |           | Generate random pathways |
| `--verbose`            | `False`        |           | Print results on the stdout |

This is an example of command line:

```text
biopathai --pathways ~/<pathways-file>.tsv --in-file ~/<data-files>.tsv --in-key-pos 0 --in-value-pos 1 --prepare-in-memory --evaluate-in-memory --out-folder ~/<output-folder>
```

> **Note**
> Please note that this command will produce a table with the accuracies reached by the specified classification algorithm on the real pathway matrices with gene expression values. In order to produce the same table for the random pathway matrices, you can simply add the following arguments to the command line `--random-classes --how-many 1000`. This will shuffle the classes a thousand times. If you want to generate pathways with a random combination of genes, you can replace `--random-classes` with `--random-pathways`.

## Output

`BioPathAI` produces a CSV file with the list of pathways IDs with the accuracy reached by the specified machine learning algorithm on the cross-validated models built on the gene expression values.

In order to produce the p-values and rank the pathways according to their relevance in relation to a specific disease or condition, `BioPathAI` provides a subroutine called `biopathai_pvalue` that can be executed with the following command:

```text
python biopathai_pvalue.py --real-accuracies ~/<biopathai-real-results>.csv --random-accuracies ~/<biopathai-random-results>.csv --dump ~/<output-pvalues-file>.csv
```

Please note that you have to run `BioPathAI` twice. The first time on the real pathways, then on the random pathways matrices (by specifying `--random-classes` or `--random-pathways` as explained above).

## Credits

Please credit out work in your manuscript by citing:

> _Manuscript in preparation_

## Support and contributions

Long-term discussion and bug reports are maintained via [GitHub Issues](https://github.com/cumbof/BioPathAI/issues), while code review is managed via [GitHub Pull Requests](https://github.com/cumbof/BioPathAI/pulls).

Please, (i) be sure that there are no existing issues/PR concerning the same bug or improvement before opening a new issue/PR; (ii) write a clear and concise description of what the bug/PR is about; (iii) specifying the list of steps to reproduce the behavior in addition to versions and other technical details is highly recommended.
