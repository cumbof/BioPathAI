#!/bin/bash

INFOLDER=$1
OUTFOLDER=$2
XARGSNPROC=$3

NPATHWAYS=`ls "${INFOLDER}" | wc -l`
NEVAL=`ls "${OUTFOLDER}/*__evaluation.csv" | wc -l`
while [ $NEVAL -lt $NPATHWAYS ]; do
    find $INFOLDER \
        -type f -name "*.csv" -follow | xargs -n 1 -P $XARGSNPROC sh -c \
        'if [ ! -f "${OUTFOLDER}"/"$(basename "$0" .csv)__evaluation.csv" ] ; then \
            python evaluate.py \
                        --input "$0" \
                        --folds 10 \
                        --output_prefix "${OUTFOLDER}"/"$(basename "$0" .csv)" \
                        --verbose ; \
         fi'
    # Search for missing evaluations and process them one by one
    NEVAL=`ls "${OUTFOLDER}/*__evaluation.csv" | wc -l`
    XARGSNPROC=1
    # Repeating the process will generate the missing evaluations only
done