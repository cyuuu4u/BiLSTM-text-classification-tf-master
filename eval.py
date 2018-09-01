#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# 命令行参数
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt_test/rt-polarity_test.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt_test/rt-polarity_test.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("positive_data_file", "./imdb_test/pos.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./imdb_test/neg.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)") # 一个batch的默认大小
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train: # 测试所有输入的训练样本
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)  # 返回每句数据标签中值为1的下标值，返回1是positive，返回0是negative
    print("Total number of test examples: {}".format(len(y_test)))

else: # 测试自己输入的样本
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print("\nEvaluating...\n")

# Evaluation
# ==================================================
# 找到最新的时间点应用模型
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    # 打开会话,准备传入构造好的graph,交给后台运算
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # 以上分别为保存,输出 输入占位符 的设置

        # Generate batches for one epoch
        # 取batches,每FLAGS.batch_size个list(x_test),作一个batch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        # 判断输出,用来和标签比对,即知道正确率,见下
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            # 循环相连接,把所有的x_test_batch跑出来的结果x_test_batch,首尾相连

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test)) # 正确预测的个数
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test)))) # 预测的正确率
    # y_test=所有的标签,correct_predictions=正确率求和,两者求商,为正确率

# Save the evaluation to a csv
# 准备数据,存入该路径下的prediction.csv文件
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
