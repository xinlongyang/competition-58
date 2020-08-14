#!/bin/sh
PWD_DIR=$(cd `dirname $0`; pwd)

python -u run.py \
  --do_train=true \
  --data_dir=$PWD_DIR/data \
  --vocab_file=$PWD_DIR/pre-train/vocab \
  --bert_config_file=$PWD_DIR/pre-train/bert_config.json \
  --init_checkpoint=$PWD_DIR/pre-train/pretraining_output/model.ckpt-2000 \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=30 \
  --output_dir=$PWD_DIR/output \
  --model_save=$PWD_DIR/output/model
