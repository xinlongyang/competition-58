#!/bin/sh
PWD_DIR=$(cd `dirname $0`; pwd)

python -u run.py \
  --do_predict=true \
  --data_dir=$PWD_DIR/data \
  --vocab_file=$PWD_DIR/pre-train/vocab \
  --bert_config_file=$PWD_DIR/pre-train/bert_config.json \
  --init_checkpoint=$PWD_DIR/output/model \
  --max_seq_length=128 \
  --output_dir=$PWD_DIR/output \
