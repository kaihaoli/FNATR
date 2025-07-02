export NAME=1pos_gc_epoch40_ls8_bs416
export OUTPUT_DIR=results/biencoder_$NAME
bash scripts/train_biencoder_esci.sh  > biencoder_$NAME.out
bash scripts/encode_esci.sh $OUTPUT_DIR
bash scripts/search_esci.sh $OUTPUT_DIR test


