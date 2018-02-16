#/bin/bash
SRC_LANG=zh
TGT_LANG=en

RECASEMODEL=models/recaser/nist/moses.ini
RECASER_BIN=tools/mosesdecoder/bin/moses

sed 's/\@\@ //g' | \
$SCRIPTS_ROOTDIR/recaser/recase-stdin.perl -model ${RECASEMODEL} -in - -moses ${RECASER_BIN}
