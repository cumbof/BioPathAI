#!/bin/bash

INFOLDER=$1
OUTFOLDER=$2
XARGSNPROC=$3
NPROC=$4

NPATHWAYS=`ls "${INFOLDER}" | wc -l`
NEVAL=`ls "${OUTFOLDER}"/*__confusion.csv | wc -l`
while [ $NEVAL -lt $NPATHWAYS ]; do
    find $INFOLDER \
        -type f -name "*.csv" -follow | xargs -n 1 -P $XARGSNPROC -i sh -c \
            'INPUT={};
	         if [ ! -f '"${OUTFOLDER}"'/"$(basename "${INPUT}" .csv)__confusion.csv" ] ; then \
                echo "${INPUT}";
                python3 evaluate.py --input "${INPUT}" \
                                    --folds 10 \
                                    --output_prefix '"${OUTFOLDER}"'/"$(basename "${INPUT}" .csv)" \
                                    --nproc '"${NPROC}"' ; \
             fi'
    # Search for missing runs and process them one by one
    NEVAL=`ls "${OUTFOLDER}"/*__confusion.csv | wc -l`
    XARGSNPROC=1
    # Repeating the process will generate the missing runs only
done
