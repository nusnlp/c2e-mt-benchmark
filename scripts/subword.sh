#!/bin/bash
TOPDIR=`dirname $0`/..
SWDIR=${TOPDIR}/tools/subword-nmt
EXPSET=$1 # experimental settings, unpc or nist

BPE_DIR=${TOPDIR}/models/subword/${EXPSET}
BPE_MODEL=${BPE_DIR}/zh.bpe
BPE_VCB=${BPE_DIR}/zh.vcb
VTH=50  # vocabulary threshold

python ${SWDIR}/apply_bpe.py -c ${BPE_MODEL} --vocabulary ${BPE_VCB} --vocabulary-threshold ${VTH}
