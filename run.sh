#!/usr/bin/env bash

set -x
set -e

############
# baseline 
###########
export NAME=1pos_gc_epoch40_ls8_bs416
export OUTPUT_DIR=results/biencoder_$NAME
export AD_POS=0
export FN=0
export FULL_LOSS=False
export N_HARD=0
export DEDUP=0
export BS=416
bash scripts/train_biencoder_esci.sh  > biencoder_$NAME.out
bash scripts/encode_esci.sh $OUTPUT_DIR
bash scripts/search_esci.sh $OUTPUT_DIR test
############
# top k negatives
###########
export NAME=1pos_gc_epoch40_ls8_bs416_nh100
export OUTPUT_DIR=results/biencoder_$NAME
export N_HARD=100
bash scripts/train_biencoder_esci.sh  > biencoder_$NAME.out
bash scripts/encode_esci.sh $OUTPUT_DIR
bash scripts/search_esci.sh $OUTPUT_DIR test
############
# top k negatives w/ false negative removal
###########
export NAME=1pos_gc_epoch40_ls8_bs416_nh100_fnkk085
export OUTPUT_DIR=results/biencoder_$NAME
export AD_POS=5
export FN=0.85
bash scripts/train_biencoder_esci.sh  > biencoder_$NAME.out
bash scripts/encode_esci.sh $OUTPUT_DIR
bash scripts/search_esci.sh $OUTPUT_DIR test
############
# top k negatives w/ false negative removal and deduplication
###########
export NAME=1pos_gc_epoch40_ls8_bs416_nh100_fnkk085_dedup
export OUTPUT_DIR=results/biencoder_$NAME
export DEDUP=0.999
bash scripts/train_biencoder_esci.sh  > biencoder_$NAME.out
bash scripts/encode_esci.sh $OUTPUT_DIR
bash scripts/search_esci.sh $OUTPUT_DIR test
