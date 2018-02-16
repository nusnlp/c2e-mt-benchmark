#!/bin/bash
TOPDIR=`dirname $0`/..
EXPSET=$1  # nist or unpc
INPUT=$2  # name of input file, word-segmented but not subword-fragmented
OUTPUT=$3  # name of output file
if [[ ! -z $4 ]]; then
    DEVICES=$4
else
    DEVICES=cpu
fi

NEMATUS_DIR=${TOPDIR}/tools/nematus
NEMATUS_DECODER=${NEMATUS_DIR}/nematus/translate.py

# Generate N-best list
BEAM_SIZE=25
MODELS=`ls ${TOPDIR}/models/translation/${EXPSET}/model-*.best.npz | xargs`

bash ${TOPDIR}/scripts/subword.sh ${EXPSET} < ${INPUT} > ${INPUT}.subword
python2 ${NEMATUS_DECODER} -m ${MODELS} -i ${INPUT}.subword -o ${OUTPUT}.subword \
    -dl ${DEVICES} -k ${BEAM_SIZE} -n -p 4

# Post-process N-best list for augmenting
cat ${OUTPUT}.subword \
    | sed "s:@@ ::g" > ${OUTPUT}
