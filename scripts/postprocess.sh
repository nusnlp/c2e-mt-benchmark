#/bin/bash
SCRIPTS_ROOTDIR=tools/mosesdecoder/scripts
RECASEMODEL=models/recaser/nist/moses.ini
RECASER_BIN=tools/mosesdecoder/bin/moses
INFILE=$1

# Rejoin sub-word fragments into word tokens then recase the output accordingly
sed 's/\@\@ //g' | \
$SCRIPTS_ROOTDIR/recaser/recase.perl -model ${RECASEMODEL} -in ${INFILE} -moses ${RECASER_BIN}
