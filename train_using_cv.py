#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation") # 限制可选参数必须是float 验证集的数量为10%
tf.flags.DEFINE_string("positive_data_file", "./data/rt_train/rt-polarity_train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt_train/rt-polarity_train.neg", "Data source for the negative data.")

# Model Hyperparameters
# 超参
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)") # 每训练100次测试下
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)") # 保存一次模型
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("k_fold", 5, "k-fold validation (default: 10)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement") # 加上一个布尔类型的参数，要不要自动分配
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices") # 加上一个布尔类型的参数，要不要打印日志

tf.flags.DEFINE_boolean("random", False, "Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static", False, "Keep the word embeddings static (default: False)")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file) # 返回[x_text,y]
test_text, test_labels = data_helpers.load_data_and_labels("./data/rt_test/rt-polarity_test.pos", "./data/rt_test/rt-polarity_test.neg")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text]) # 把每句话中的单词用空格隔开并计算最长
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) # tensorflow提供的工具，将数据填充为最大长度，默认0填充
x = np.array(list(vocab_processor.fit_transform(x_text))) # fit_transform:先拟合数据，然后转化它将其转化为标准形式.生成随机数矩阵每句话为一行后面补零
x_test = np.array(list(vocab_processor.fit_transform(test_text)))
y_test = np.argmax(test_labels, axis=1)

# build cv
revs = []
for i in range(x.shape[0]):
    data_and_cv = {"idx_of_sent":x[i], # 都是array
                   "label_of_sent":y[i],
                   "split":np.random.randint(0, FLAGS.k_fold)}
    revs.append(data_and_cv)
# Randomly shuffle data
# 随机打乱数据，生成固定随机数
# np.random.seed(10)  # seed用于指定随机数生成
# shuffle_indices = np.random.permutation(np.arange(len(y)))# arange(len(y))返回一个array([0,1,2...,10661]) shuffle_indiced返回一个array([7767,...10433])随机序列
# x_shuffled = x[shuffle_indices] # 打乱了数据 还是返回每个句子为一行的随机数矩阵后面补零
# y_shuffled = y[shuffle_indices] # 生成对应的label的矩阵每行为一个句子，是[1,0]或[0,1]

# Split train/test set
# TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y))) # 取得索引
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:] # 划分数据集，分为训练、验证
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# x_train每行是一个sentence的随机数向量

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev))) # 打印切分的比例


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x.shape[1],
            num_classes=y.shape[1],
            vocab_size=len(vocab_processor.vocabulary_), # 计算单词的数目
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),# 上面定义的filter_sizes拿过来，"3,4,5"按","分割
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,# l2正则化项
            trainable=not FLAGS.static)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False) # 首先将step定义为变量初始化为0
        optimizer = tf.train.AdamOptimizer(1e-3)# 定义优化器使用adam优化
        grads_and_vars = optimizer.compute_gradients(cnn.loss) # 将使用卷积神经网络计算出来的损失函数最小化。 该方法会返回list[(gradients,variable)]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) # 使用优化器更新参数，每进行一次参数更新就加一次global step

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        # 每一步都存参数,tensorboard可以看
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        # 损失函数和准确率的参数保存
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        # 训练数据保存
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        # 测试数据保存
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir): # 如果路径不存在就创建一个
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if not FLAGS.random:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            initW = data_helpers.load_embedding_vectors_word2vec(vocabulary, "./data/GoogleNews-vectors-negative300.bin", True)
            print("word2vec file has been loaded")
            sess.run(cnn.W.assign(initW))


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = { # 给placeholder喂数据
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 # 检测的时候不采用dropout
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, Test acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy

        def test_step(x_batch):
            feed_dict = {cnn.input_x: x_batch,
                         cnn.dropout_keep_prob: 1.0}
            predictions = sess.run(cnn.predictions, feed_dict)
            return predictions

        test_max_acc = []
        for cv in range(FLAGS.k_fold):
            # 根据十折交叉将数据分为一个cv要用的训练集和验证集
            print("cv: %d" % cv)
            x_train, y_train = [], []
            x_dev, y_dev = [], []
            for rev in revs:
                idx_of_sent = list(rev["idx_of_sent"])
                label_of_sent = list(rev["label_of_sent"])
                if rev["split"] == cv:
                    x_dev.append(idx_of_sent)
                    y_dev.append(label_of_sent)
                else:
                    x_train.append(idx_of_sent)
                    y_train.append(label_of_sent)
            x_train = np.array(x_train, dtype="int")
            y_train = np.array(y_train, dtype=np.float32)
            x_dev = np.array(x_dev, dtype="int")
            y_dev = np.array(y_dev, dtype=np.float32)
            # Generate batches
            batches_per_epoch = int(len(y_train) / FLAGS.batch_size)
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs) # zip是将两个list一一对应打包为元组形式，以短的为长度。这里又将各元组对转换为list了,但每个list里存的还是两个array
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch) # 可理解为解压，返回二维矩阵的形式
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % batches_per_epoch == 0: # 每训练一定数量的batch后就用验证集验证一下看验证集上的loss是否有在持续下降，EARLY STOPPING以解决overfitting的问题
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0: # 每训练一段就写一次checkpoint
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # Testing on each cv
            print("\nTesting:")
            predictions = test_step(x_test)
            correct_predictions = float(sum(predictions == y_test))
            test_acc = correct_predictions / float(len(y_test))
            test_max_acc.append(test_acc)
            print("Total number of test examples: {}".format(len(y_test)))
            print("Test Accuracy: {:g}".format(test_acc))
            print("")

        # Average Testing accuracy
        print("Average Testing Accuracy:", np.mean(test_max_acc))

        # Max Testing accuracy
        max_acc = 0.
        for acc in test_max_acc:
            if acc > max_acc:
                max_acc = acc
        print("Max Testing Accuracy: {:g}".format(max_acc))