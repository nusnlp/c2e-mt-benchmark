#!/bin/bash
# Run Chinese word segmenter based on CTB model and lowercase the input
TOPDIR=`dirname $0`/..
SEGMENTDIR=${TOPDIR}/tools/segmenter

sed -f `dirname $0`/normalize-zh.sed \
    | python2 ${SEGMENTDIR}/segment.py -m ${SEGMENTDIR}/ctbModel -l ${SEGMENTDIR}/lex.GB.txt \
    | perl `dirname $0`/lowercase.perl
