#!/bin/bash
# theano device, in case you do not want to compute on gpu, change it to cpu
device=cuda0

source vars

# Change these with actual training data
TRAIN_PREFIX=train
TUNE_PREFIX=tune
DICT_PREFIX=bpedict

echo Started training: `date`
mkdir -p model-dstack-gru
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,gpuarray.preallocate=0.8 python2 ${nematus}/nematus/nmt.py \
    --model model-dstack-gru/model.npz --reload \
    --datasets ${CORPUS_PREFIX}.${SRC} ${CORPUS_PREFIX}.${TGT} \
    --valid_datasets ${TUNE_PREFIX}.${SRC} ${TUNE_PREFIX}.${TGT} \
    --dictionaries ${DICT_PREFIX}.${SRC}.json ${DICT_PREFIX}.${TGT}.json \
    --encoder lstm --decoder lstm_cond --decoder_deep lstm \
    --dim_word 500 \
    --dim 1024 \
    --enc_recurrence_transition_depth 4 \
    --dec_base_recurrence_transition_depth 8 \
    --n_words_src ${VOCAB_SIZE_S} \
    --n_words ${VOCAB_SIZE_T} \
    --maxlen 50 \
    --optimizer adam \
    --decay_c 0 \
    --clip_c 1 \
    --lrate 0.0001 \
    --layer_normalisation --tie_decoder_embeddings \
    --batch_size 40 \
    --valid_batch_size 40 \
    --validFreq 10000 \
    --dispFreq 1000 \
    --saveFreq 10000 \
    --dropout_embedding 0.2 \
    --dropout_hidden 0.2 \
    --dropout_source 0.1 \
    --dropout_target 0.1 \
    --finish_after 1200000 \
    --patience 10 \
    --external_validation_script ./validate.sh \

echo Finished training: `date`
