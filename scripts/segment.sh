#!/bin/bash
# Run Chinese word segmenter based on CTB model and lowercase the input

SEGMENTDIR=`dirname $0`/../tools/segmenter

python2 ${SEGMENTDIR}/segment.py -m ${SEGMENTDIR}/ctbModel -l ${SEGMENTDIR}/lex.GB.txt \
    | perl `dirname $0`/detokenizer.perl -l en
