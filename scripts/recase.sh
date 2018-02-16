#/bin/bash
BASEDIR=`dirname $0`

RECASEMODEL=${BASEDIR}/../models/recaser/nist/moses.ini
RECASER_BIN=${BASEDIR}/../tools/mosesdecoder/bin/moses

# Rejoin sub-word fragments into word tokens then recase the output accordingly
${BASEDIR}/recase-stdin.perl -model ${RECASEMODEL} -moses ${RECASER_BIN}
