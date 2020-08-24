# coding=utf-8

'''BERT finetuning runner.
支持训练过程中观测模型效果 
created by syzong
2020/4
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import random
import time
import os
import sys
import modeling
import optimization
import tokenization
import tensorflow as tf
import pickle
import numpy as np
from split_data import split_data
from metrics import mean, get_multi_metrics

#cpu模式下改为 -1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("bert_config_file", None, "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", None, "output directory.")
flags.DEFINE_string("model_save", None, "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_load_train", False, "Whether to run incremental training.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_bool("train_arm", False, "Whether arm training.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 30, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 500, "How often to save the model checkpoint.")
flags.DEFINE_integer("save_checkpoints_epoch", 5, "over the epoch start to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

def get_data(file_path, is_training=True):
    inputs = []
    labels = []
    line_num = 0
    with open(file_path, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            #过滤第一行的标签
            if line_num == 0:
                line_num = 1
                continue
            #item = line.split("\t")
            item = line.strip("\n").split("\t")
            labels.append(item[0])
            inputs.append(item[1])

    if is_training:
        uni_label = list(set(labels))
        label_to_index = dict(zip(uni_label, list(range(len(uni_label)))))
        with open(os.path.join(FLAGS.output_dir, "label_to_index.json"), "w", encoding="utf8") as fw:
            json.dump(label_to_index, fw, indent=0, ensure_ascii=False)
        
        index_to_label = dict(zip(list(range(len(uni_label))),uni_label))
        with open(os.path.join(FLAGS.output_dir, "index_to_label.json"), "w", encoding="utf8") as fw:
            json.dump(index_to_label, fw, indent=0, ensure_ascii=False)
    else:
        with open(os.path.join(FLAGS.output_dir, "label_to_index.json"), "r", encoding="utf8") as fr:
            label_to_index = json.load(fr)

    # 2，输入转索引
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    inputs_ids = []
    input_masks = []
    segment_ids = []
    for text in inputs:
        text = tokenization.convert_to_unicode(text)
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        inputs_ids.append(input_id)
        input_masks.append([1] * len(input_id))
        segment_ids.append([0] * len(input_id))


    pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
    for input_id, input_mask, segment_id in zip(inputs_ids, input_masks, segment_ids):
        if len(input_id) < FLAGS.max_seq_length:
            pad_input_ids.append(input_id + [0] * (FLAGS.max_seq_length - len(input_id)))
            pad_input_masks.append(input_mask + [0] * (FLAGS.max_seq_length - len(input_mask)))
            pad_segment_ids.append(segment_id + [0] * (FLAGS.max_seq_length - len(segment_id)))
        else:
            pad_input_ids.append(input_id[:FLAGS.max_seq_length])
            pad_input_masks.append(input_mask[:FLAGS.max_seq_length])
            pad_segment_ids.append(segment_id[:FLAGS.max_seq_length])

    labels_ids = [label_to_index[label] for label in labels]

    for i in range(5):
        print("line {}: *****************************************".format(i))
        print("input: ", inputs[i])
        print("input_id: ", pad_input_ids[i])
        print("input_mask: ", pad_input_masks[i])
        print("segment_id: ", pad_segment_ids[i])
        print("label_id: ", labels_ids[i])

    return inputs, pad_input_ids, pad_input_masks, pad_segment_ids, labels_ids, label_to_index

def next_batch(batch_size, input_ids, input_masks, segment_ids, label_ids, is_training=True):
    z = list(zip(input_ids, input_masks, segment_ids, label_ids))
    if is_training:
        random.shuffle(z)
    input_ids, input_masks, segment_ids, label_ids = zip(*z)
    totle_len = len(input_ids)
    num_batches = len(input_ids) // batch_size
    if_remain = False
    if totle_len > num_batches*batch_size :
        if_remain = True
        num_batches = num_batches + 1
    for i in range(num_batches):
        if if_remain and i == num_batches-1:
            start = i * batch_size
            end = totle_len
        else:
            start = i * batch_size
            end = start + batch_size
        batch_input_ids = input_ids[start: end]
        batch_input_masks = input_masks[start: end]
        batch_segment_ids = segment_ids[start: end]
        batch_label_ids = label_ids[start: end]

        yield dict(input_ids=batch_input_ids, input_masks=batch_input_masks, segment_ids=batch_segment_ids, label_ids=batch_label_ids)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    predictions = tf.argmax(logits, axis=-1, name="predictions")
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities, predictions)

def id_to_label(label_to_index , id):
    for k,v in label_to_index.items():
        if v == id:
            return k

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_predict', must be True.")

    tf.gfile.MakeDirs(FLAGS.output_dir)
    input_ids_holder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids_holder')
    input_masks_holder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask_holder')
    segment_ids_holder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids_holder')
    label_ids_holder = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids_holder")
    
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
   
    if FLAGS.do_train:
        split_data(FLAGS.data_dir, is_training=True)
        _, train_in_ids, train_in_masks, train_seg_ids, train_lab_ids, lab_to_idx = get_data(os.path.join(FLAGS.data_dir, "train.tsv"),True)
        _, dev_in_ids, dev_in_masks, dev_seg_ids, dev_lab_ids, dev_lab_to_idx = get_data(os.path.join(FLAGS.data_dir, "dev.tsv"),False)
        num_train_steps = None
        num_warmup_steps = None
        num_train_steps = int(len(train_in_ids) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        label_list = [value for key, value in lab_to_idx.items()]
        num_labels = len(label_list)
        print("label numbers: ", num_labels)

        loss, per_example_loss, logits, probabilities, predictions = create_model(bert_config, True, input_ids_holder, input_masks_holder,
                                                                        segment_ids_holder, label_ids_holder, num_labels)
        train_op = optimization.create_optimizer(loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
        #max_to_keep最多保存几个模型
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=3)
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

            sess.run(tf.variables_initializer(tf.global_variables()))
            max_acc=0
            current_step = 0
            start = time.time()
            for epoch in range(FLAGS.num_train_epochs):
                print("----- Epoch {}/{} -----".format(epoch + 1, FLAGS.num_train_epochs))

                for batch in next_batch(FLAGS.train_batch_size, train_in_ids, train_in_masks, train_seg_ids, train_lab_ids, True):
                    feed_dict = {input_ids_holder: batch["input_ids"], input_masks_holder: batch["input_masks"], 
                                segment_ids_holder: batch["segment_ids"], label_ids_holder: batch["label_ids"]}
                    # 训练模型
                    _, train_loss, train_predictions = sess.run([train_op, loss, predictions], feed_dict=feed_dict)
                    
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=train_predictions, true_y=batch["label_ids"], labels=label_list)
                    print("train: total_step: %d, current_step: %d, loss: %.4f, acc: %.4f, recall: %.4f, precision: %.4f, f_beta: %.4f"
                                %(num_train_steps, current_step, train_loss, acc, recall, prec, f_beta))

                    current_step += 1
                    if current_step % FLAGS.save_checkpoints_steps == 0 and epoch > FLAGS.save_checkpoints_epoch:
                        eval_losses = []
                        eval_predictions_all = []
                        label_ids_all = []
                        
                        for eval_batch in next_batch(FLAGS.eval_batch_size, dev_in_ids, dev_in_masks, dev_seg_ids, dev_lab_ids, True):
                            eval_feed_dict = {input_ids_holder: eval_batch["input_ids"], input_masks_holder: eval_batch["input_masks"], 
                                        segment_ids_holder: eval_batch["segment_ids"], label_ids_holder: eval_batch["label_ids"]}

                            eval_loss, eval_predictions = sess.run([loss, predictions], feed_dict=eval_feed_dict)
                            eval_losses.append(eval_loss)
                            eval_predictions_all.extend(eval_predictions)
                            label_ids_all.extend(eval_batch["label_ids"])
                            
                        print("\n")
                        eval_acc, eval_recall, eval_prec, eval_f_beta = get_multi_metrics(pred_y=eval_predictions_all, true_y=label_ids_all, labels=label_list)

                        print("eval:  loss: %.4f, acc: %.4f, recall: %.4f, precision: %.4f, f_beta: %.4f"
                                        %(mean(eval_losses), eval_acc, eval_recall, eval_prec, eval_f_beta))
                        print("\n")

                        if eval_acc >= max_acc:
                            print("********** save new model, step {} , dev_acc {}".format(current_step,eval_acc))
                            max_acc = eval_acc
                            saver.save(sess, FLAGS.model_save, global_step=current_step)

            end = time.time()
            print("total train time: ", end - start)

    if FLAGS.do_predict:
        test_inputs, test_in_ids, test_in_masks, test_seg_ids, test_lab_ids, lab_to_idx = get_data(os.path.join(FLAGS.data_dir, "test.tsv"),False)
        label_list = [value for key, value in lab_to_idx.items()]
        num_labels = len(label_list)
        print("****Test*****\n label numbers: ", num_labels)
        loss, per_example_loss, logits, probabilities, predictions = create_model(bert_config, False, input_ids_holder, input_masks_holder,
                                                                        segment_ids_holder, label_ids_holder, num_labels)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            model_file=tf.train.latest_checkpoint(FLAGS.output_dir)
            print("model_file:",model_file)
            saver.restore(sess,model_file)
            start = time.time()
            test_predictions_all = []
            label_ids_all = []
            test_prob_all = []
            test_losses = []

            for test_batch in next_batch(FLAGS.predict_batch_size, test_in_ids, test_in_masks, test_seg_ids, test_lab_ids, False):
                test_feed_dict = {input_ids_holder: test_batch["input_ids"], input_masks_holder: test_batch["input_masks"], 
                            segment_ids_holder: test_batch["segment_ids"], label_ids_holder: test_batch["label_ids"]}

                test_loss, test_predictions, test_prob = sess.run([loss, predictions, probabilities], feed_dict=test_feed_dict)
                test_predictions_all.extend(test_predictions)
                label_ids_all.extend(test_batch["label_ids"])
                test_prob_all.extend(test_prob.tolist())
                test_losses.append(test_loss)

            print("\n")
            test_acc, test_recall, test_prec, test_f_beta = get_multi_metrics(pred_y=test_predictions_all, true_y=label_ids_all, labels=label_list)

            print("Test:  loss: %.4f, acc: %.4f, recall: %.4f, precision: %.4f, f_beta: %.4f"
                        %(mean(test_losses), test_acc, test_recall, test_prec, test_f_beta))
            print("\n")
            ext_id = 0
            output_predict_file = os.path.join(FLAGS.output_dir, "sub.csv")
            with tf.gfile.GFile(output_predict_file, "w") as writer:
                output_line = "ext_id,std_id,prob\n"
                writer.write(output_line)
                test_len = len(test_in_ids)
                pred_len = len(test_predictions_all)
                print("test_len : %d , pred_len : %d\n"%(test_len,pred_len))
                for i in range(test_len):
                    prob = round(test_prob_all[i][test_predictions_all[i]],4)
                    output_line = str(ext_id) + "," + id_to_label(lab_to_idx, test_predictions_all[i]) + "," + str(prob) + "\n"
                    ext_id+=1
                    writer.write(output_line)

            end = time.time()
            print("total test time: ", end - start)

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
