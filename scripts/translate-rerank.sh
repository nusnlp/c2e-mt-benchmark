#!/bin/bash
TOPDIR=`dirname $0`/..
EXPSET=$1  # nist or unpc
INPUT=$2  # name of input file, word-segmented but not subword-fragmented
OUTPUT=$3  # name of output file
if [ ! -z $4 ]; then
    DEVICES=$4
else
    DEVICES=cpu0
fi

TMPDIR=/tmp  # feel free to change this

NEMATUS_DIR=${TOPDIR}/tools/nematus
NEMATUS_DECODER=${NEMATUS_DIR}/nematus/translate.py
RERANKER_DIR=${TOPDIR}/tools/nbest-reranker
RERANKER=${RERANKER}/rerank.py
AUGMENTER=${RERANKER}/augmenter.py

RERANKER_MODEL_DIR=${TOPDIR}/models/reranker/${EXPSET}

LM_FILE=${RERANKER_MODEL_DIR}/binlm
RERANKER_FEATSTR="LM('LM0','${LM_FILE}',normalize=True)"

# Generate N-best list
NBEST_SIZE=50
MODELS=`ls ${TOPDIR}/models/${EXPSET}/model-*.best.npz | xargs`
NBEST_TXT=${OUTPUT}.nbest

bash ${TOPDIR}/scripts/subword.sh ${EXPSET} < ${INPUT} > ${INPUT}.subword
python2 ${NEMATUS_DECODER} -m ${MODELS} -i ${INPUT}.subword -o ${NBEST_TXT} \
    -dl ${DEVICES} -k ${NBEST_SIZE} --n-best -n -p 4

# Post-process N-best list for augmenting
cat ${NBEST_TXT} \
    | sed "s:@@ ::g" | sed "s:|||\s*\([0-9\.]\+\)\s*$:||| NMT0= \1:" \
    > ${NBEST_TXT}.postproc

# Rerank N-best
BASE_IN=`basename ${NBEST_TXT}.augment`
python2 ${AUGMENTER} -s ${INPUT} -f "${RERANKER_FEATSTR}" -i ${NBEST_TXT}.postproc -o ${NBEST_TXT}.augment
python2 ${RERANKER} -i ${NBEST_TXT}.augment -w ${RERANKER_MODEL_DIR}/weights.txt -o /tmp/rerank.${NBEST_TXT}
mv /tmp/rerank.${NBEST_TXT}/${BASE_IN}.reranked.1best ${OUTPUT}
